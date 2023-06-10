# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loads an uncompressed pretrained model, compresses the model and evaluates its performance on imagenet"""
import logging
import math
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import classification_report
import torch.nn.functional as F

from .compression.model_compression import compress_model
from .dataloading.FI_loader import get_loaders
from .dataloading.PM_dataset import PM_dataset
from .permutation.model_permutation import permute_model
from .training.imagenet_utils import ImagenetTrainer, ImagenetValidator, get_imagenet_criterion
from .training.lr_scheduler import get_learning_rate_scheduler
from .training.optimizer import get_optimizer
from .training.training import TrainingLogger, train_one_epoch
from .training.validating import ValidationLogger, validate_one_epoch
from .utils.config_loader import load_config
from .utils.horovod_utils import initialize_horovod
from .utils.logging import get_tensorboard_logger, log_compression_ratio, log_config, setup_pretty_logging
from .utils.model_size import compute_model_nbits
from .utils.models import get_uncompressed_model
from .utils.state_dict_utils import save_state_dict_compressed, load_state_dict

# fmt: off
try:
    import horovod.torch as hvd
    HAVE_HOROVOD = True
except ImportError:
    HAVE_HOROVOD = False            #与多GPU训练有关
# fmt: on


_MODEL_OUTPUT_PATH_SUFFIX = "trained_models"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, test_loader, config):
    model.eval().to(DEVICE)
    test_loss = 0
    correct = 0
    pred_list = torch.Tensor()
    true_list = torch.Tensor()
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.long().squeeze().to(DEVICE)
        with torch.no_grad():
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        pred_list = torch.cat((pred_list, pred.squeeze().cpu()), 0)
        true_list = torch.cat((true_list, target.cpu()), 0)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # get target have the same shape of pred

    report = classification_report(true_list, pred_list, labels=range(config["dataloader"]["num_classes"]), digits=4)
    print(report)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return report


def main():
    setup_pretty_logging()
    verbose = initialize_horovod()
    print(verbose)
    start_timestamp = datetime.now()

    # specify config file to use in case user does not pass in a --config argument
    file_path = os.path.dirname(__file__)
    default_config = os.path.join(file_path, "../config/train_resnet34.yaml")
    config = load_config(file_path, default_config_path=default_config)
    # summary_writer = None
    if (HAVE_HOROVOD and hvd.rank == 0) or (not HAVE_HOROVOD):
        summary_writer = get_tensorboard_logger(config["output_path"])
    log_config(config, summary_writer)

    # Get the model, optimize its permutations, and compress it
    model_config = config["model"]
    compression_config = model_config["compression_parameters"]
    dataloader_config = config["dataloader"]

    model = get_uncompressed_model(model_config["arch"],
                                   pretrained=False,
                                   path=model_config["model_path"],
                                   num_classes=dataloader_config["num_classes"]).to(DEVICE)
    # 从网站上下载别人提供的预训练模型
    # C:\Users\LiuYan\.cache\torch\hub\checkpoints，下载到这个位置了

    if "permutations" in model_config and model_config.get("use_permutations", False):
        permute_model(
            model,
            compression_config["fc_subvector_size"],
            compression_config["pw_subvector_size"],
            compression_config["large_subvectors"],
            permutation_groups=model_config.get("permutations", []),
            layer_specs=compression_config["layer_specs"],
            sls_iterations=model_config["sls_iterations"],
        )

    uncompressed_model_size_bits = compute_model_nbits(model)
    model = compress_model(model, **compression_config).to(DEVICE)

    compressed_model_size_bits = compute_model_nbits(model)
    log_compression_ratio(uncompressed_model_size_bits, compressed_model_size_bits, summary_writer)

    if HAVE_HOROVOD:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # Create training and validation dataloaders

    # val_data_sampler, val_data_loader = load_imagenet_val(
    #     dataloader_config["imagenet_path"],
    #     dataloader_config["num_workers"],
    #     dataloader_config["batch_size"],
    #     shuffle=dataloader_config["validation_shuffle"],
    # )
    # train_sampler, train_data_loader = load_imagenet_train(
    #     dataloader_config["imagenet_path"],
    #     dataloader_config["num_workers"],
    #     dataloader_config["batch_size"],
    #     shuffle=dataloader_config["train_shuffle"],
    # )
    if dataloader_config["dataset"] == "sewage":
        train_data_loader, val_data_loader, _ = get_loaders(dataloader_config["imagenet_path"],
                                                            dataloader_config["batch_size"],
                                                            dataloader_config["image_shape"],
                                                            num_workers=dataloader_config["num_workers"])

    if dataloader_config["dataset"] == 'cifar10':
        train_data_loader = DataLoader(
            datasets.CIFAR10(dataloader_config["root"], train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Pad(4),
                                 transforms.RandomCrop(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ])),
            batch_size=dataloader_config["batch_size"],
            num_workers=dataloader_config["num_workers"],
            pin_memory=True,
            shuffle=True)
        val_data_loader = DataLoader(
            datasets.CIFAR10(dataloader_config["root"], train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=dataloader_config["test_batch_size"],
            num_workers=dataloader_config["num_workers"],
            pin_memory=True,
            shuffle=True)

    elif dataloader_config["dataset"] == "PM":
        kwargs = kwargs = {'num_workers': dataloader_config["num_workers"], 'pin_memory': True}
        train_dataset = PM_dataset(
            r"E:\LY\network-slimming\PM\PALM-Training400",
            r"E:\LY\network-slimming\PM\PALM-Validation400",
            train=True,
            transform=transforms.Compose([
                # transforms.Resize((1024, 1024)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0), ratio=(1.0, 1.0)),
                # transforms.RandomCrop(32),
                transforms.RandomGrayscale(),
                transforms.RandomRotation(degrees=30),
                # transforms.RandomAutocontrast(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
        )
        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=dataloader_config["batch_size"],
            shuffle=True,
            **kwargs,
        )
        test_dataset = PM_dataset(
            r"E:\LY\network-slimming\PM\PALM-Training400",
            r"E:\LY\network-slimming\PM\PALM-Validation400",
            train=False,
            transform=transforms.Compose([
                transforms.Resize((1024, 1024)),
                # transforms.RandomCrop(32),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
        )
        val_data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=dataloader_config["test_batch_size"],
            shuffle=True,
            **kwargs,
        )
    else:
        raise ValueError("No valid dataset is given.")

    # Get imagenet optimizer, criterion, trainer and validator
    optimizer = get_optimizer(model, config)
    criterion = get_imagenet_criterion()
    n_epochs = config["epochs"]
    assert n_epochs > 0
    n_batch_size = len(train_data_loader)
    lr_scheduler = get_learning_rate_scheduler(config, optimizer, n_epochs, n_batch_size)

    trainer = ImagenetTrainer(model, optimizer, lr_scheduler, criterion)
    training_logger = TrainingLogger(summary_writer)
    validator = ImagenetValidator(model, criterion)
    validation_logger = ValidationLogger(n_batch_size, summary_writer)

    # Keep track of the best validation accuracy we have seen to save the best model at the end of every epoch
    best_acc = -math.inf            #负无穷大
    best_acc_epoch = -1
    last_acc = -math.inf

    if not config.get("skip_initial_validation", False):
        last_acc = validate_one_epoch(0, val_data_loader, model, validator, validation_logger, verbose, DEVICE)
        best_acc = last_acc
        best_acc_epoch = 0

    if model_config["resume"] is not None:
        state = torch.load(model_config["resume"])
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])

    save_state_dict_compressed(model, os.path.join(config["output_path"], _MODEL_OUTPUT_PATH_SUFFIX, "0.pth"))

    training_start_timestamp = datetime.now()
    report = None
    for epoch in range(1, n_epochs + 1):
        train_one_epoch(epoch, train_data_loader, model, trainer, training_logger, verbose, DEVICE)

        last_acc = validate_one_epoch(epoch, val_data_loader, model, validator, validation_logger, verbose, DEVICE)

        # Save the current state of the model after every epoch
        state = {"model": model.state_dict(),
                 "optimizer": optimizer.state_dict()}
        torch.save(state, os.path.join(config["output_path"], _MODEL_OUTPUT_PATH_SUFFIX, "checkpoint.pth"))
        # save_state_dict_compressed(
        #     model, os.path.join(config["output_path"], _MODEL_OUTPUT_PATH_SUFFIX, "model.pth")
        # )
        if lr_scheduler.step_epoch():
            # last_acc is between 0 and 100. We need between 0 and 1
            lr_scheduler.step(last_acc / 100)

        if last_acc >= best_acc:
            save_state_dict_compressed(
                model, os.path.join(config["output_path"], _MODEL_OUTPUT_PATH_SUFFIX, "best.pth")
            )
            report = test(model, val_data_loader, config)
            print(report)
            best_acc = last_acc
            best_acc_epoch = epoch

    # Done training!
    if verbose:
        logging.info("Done training!")
        summary_writer.close()
        with open(os.path.join(config["output_path"], "results.txt"), "w") as f:
            print(f"{start_timestamp:%Y-%m-%d %H:%M:%S}", file=f)
            print(f"{training_start_timestamp:%Y-%m-%d %H:%M:%S}", file=f)
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S}", file=f)
            print(last_acc, file=f)
            print(best_acc, file=f)
            print(best_acc_epoch, file=f)
            print(report, file=f)


if __name__ == "__main__":
    main()
