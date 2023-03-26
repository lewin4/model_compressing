# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loads a pretrained compressed model and evaluate its performance on imagenet"""

import os
import torch
import torch.nn.functional as F
import time
from sklearn.metrics import classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.compression.model_compression import compress_model
from src.utils.config_loader import load_config
from src.utils.horovod_utils import initialize_horovod
from src.utils.logging import log_compression_ratio, setup_pretty_logging
from src.utils.model_size import compute_model_nbits
from src.utils.models import get_uncompressed_model
from src.utils.state_dict_utils import load_state_dict, save_state_dict_compressed
from src.dataloading.FI_loader import get_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    setup_pretty_logging()
    verbose = initialize_horovod()

    # specify config file to use in case user does not pass in a --config argument
    file_path = os.path.dirname(__file__)
    default_config = os.path.join(file_path, "../config/train_vgg.yaml")
    config = load_config(file_path, default_config_path=default_config)

    dataloader_config = config["dataloader"]
    model_config = config["model"]
    compression_config = model_config["compression_parameters"]
    model = get_uncompressed_model(model_config["arch"],
                                   pretrained=False,
                                   path=model_config["model_path"],
                                   dataset=dataloader_config["dataset"],
                                   num_classes=dataloader_config["num_classes"]).to(DEVICE)
    # 从网站上下载别人提供的预训练模型
    # C:\Users\LiuYan\.cache\torch\hub\checkpoints，下载到这个位置了

    uncompressed_model_size_bits = compute_model_nbits(model)
    model = compress_model(model, **compression_config).to(DEVICE)
    compressed_model_size_bits = compute_model_nbits(model)
    log_compression_ratio(uncompressed_model_size_bits, compressed_model_size_bits)

    save_state_dict_compressed(model, "./best_best.pth")

    if config["model"].get("state_dict_compressed", None) is not None:
        model = load_state_dict(model, config["model"]["state_dict_compressed"])

    # Create training and validation dataloaders
    if dataloader_config["dataset"] == 'cifar10':
        train_dataset = datasets.CIFAR10(
            dataloader_config["root"], train=True, download=True,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
             ]))
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=dataloader_config["batch_size"],
            num_workers=dataloader_config["num_workers"],
            pin_memory=True,
            shuffle=True)
        val_dataset = datasets.CIFAR10(
            dataloader_config["root"],
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]))
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=dataloader_config["test_batch_size"],
            num_workers=dataloader_config["num_workers"],
            pin_memory=True,
            shuffle=True)

    elif dataloader_config["dataset"] == 'sewage':
        train_data_loader, val_data_loader, _ = get_loaders(
            dataloader_config["imagenet_path"],
            dataloader_config["batch_size"],
            dataloader_config["image_shape"],
            num_workers=dataloader_config["num_workers"])

    elif dataloader_config["dataset"] == "miniimagenet":
        from MLclf import MLclf
        # Download the original mini-imagenet data:
        # only need to run this line before you download the mini-imagenet dataset for the first time.
        MLclf.miniimagenet_download(Download=True)
        # Transform the original data into the format that fits the task for classification:
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset, validation_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(
            ratio_train=0.6, ratio_val=0.2,
            seed_value=None, shuffle=True,
            transform=transform,
            save_clf_data=True)

        # The dataset can be transformed to dataloader via torch:
        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=dataloader_config["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=dataloader_config["num_workers"]
        )
        val_data_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=dataloader_config["test_batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=dataloader_config["num_workers"]
        )
    else:
        raise ValueError("No valid dataset is given.")

    def test_fps(start_epoch, epochs):
        total_times = 0
        model.eval()
        print("{} epochs will be test.".format(epochs - start_epoch))
        for epoch in range(start_epoch, epochs):
            print("{} epoch start......".format(epoch))
            times = 0
            epoch_time = time.time()
            for data, target in train_data_loader:
                if config["use_cuda"]:
                    data, target = data.cuda(), target.cuda()
                start_time = time.time()
                with torch.no_grad():
                    output = model(data)
                stop_time = time.time()
                times += (stop_time - start_time)
            total_times += times
            print("{} epoch finish. time: {}. Pure inference time: {}".format(epoch, time.time() - epoch_time, times))

        num = len(train_dataset) * (epochs - start_epoch)
        print(
            "\nAll time: {}, \nImage num: {}, \nTime per image: {}".format(total_times, num, total_times / float(num)))

    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        pred_list = torch.Tensor()
        true_list = torch.Tensor()
        for data, target in val_data_loader:
            data, target = data.cuda(), target.long().cuda()
            with torch.no_grad():
                output = model(data)
                test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred_list = torch.cat((pred_list, pred.squeeze().cpu()), 0)
            true_list = torch.cat((true_list, target.cpu()), 0)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # get target have the same shape of pred

        report = classification_report(true_list, pred_list, labels=range(10), digits=4)
        print(report)

        test_loss /= len(val_dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(val_dataset),
            100. * correct / len(val_dataset)))
        return report

    test(0)
    # test_fps(0, 120)
    # validator = ImagenetValidator(model, get_imagenet_criterion())
    # logger = ValidationLogger(1, None)
    # validate_one_epoch(0, val_data_loader, model, validator, logger, verbose)


if __name__ == "__main__":
    main()
