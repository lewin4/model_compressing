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

from .compression.model_compression import compress_model
from .dataloading.imagenet_loader import load_imagenet_val
from .training.imagenet_utils import ImagenetValidator, get_imagenet_criterion
from .training.validating import ValidationLogger, validate_one_epoch
from .utils.config_loader import load_config
from .utils.horovod_utils import initialize_horovod
from .utils.logging import log_compression_ratio, setup_pretty_logging
from .utils.model_size import compute_model_nbits
from .utils.models import get_uncompressed_model
from .utils.state_dict_utils import load_state_dict
from .dataloading.FI_loader import get_loaders


def main():
    setup_pretty_logging()
    verbose = initialize_horovod()

    # specify config file to use in case user does not pass in a --config argument
    file_path = os.path.dirname(__file__)
    default_config = os.path.join(file_path, "../config/train_resnet18.yaml")
    config = load_config(file_path, default_config_path=default_config)

    compression_config = config["model"]["compression_parameters"]

    model = get_uncompressed_model(config["model"]["arch"],
                                   pretrained=False,
                                   path=config["model"]["model_path"],
                                   num_classes=6
                                   ).cuda()
    uncompressed_model_size_bits = compute_model_nbits(model)
    model = compress_model(model, **compression_config)
    compressed_model_size_bits = compute_model_nbits(model)
    log_compression_ratio(uncompressed_model_size_bits, compressed_model_size_bits)
    if config["use_cuda"]:
        model.cuda()
    else:
        model.cpu()
    if config["model"].get("state_dict_compressed", None) is not None:
        model = load_state_dict(model, os.path.join(file_path, config["model"]["state_dict_compressed"]))
    # from utils.torchstat import stat
    # stat(model.cpu(), (3, 192, 256))
    dataloader_config = config["dataloader"]
    # val_data_sampler, val_data_loader = load_imagenet_val(
    #     dataloader_config["imagenet_path"],
    #     dataloader_config["num_workers"],
    #     dataloader_config["batch_size"],
    #     shuffle=dataloader_config["validation_shuffle"],
    # )
    kwargs = {'num_workers': 4, 'pin_memory': True}

    _, test_loader, _ = get_loaders(
        image_dir=dataloader_config["imagenet_path"],
        batch_size=dataloader_config["batch_size"],
        img_shape=dataloader_config["image_shape"],
        radio=[0.2, 0.7, 0.1],
        **kwargs
    )

    def test_fps(start_epoch, epochs):
        total_times = 0
        model.eval()
        print("{} epochs will be test.".format(epochs - start_epoch))
        for epoch in range(start_epoch, epochs):
            print("{} epoch start......".format(epoch))
            times = 0
            epoch_time = time.time()
            for data, target in test_loader:
                if config["use_cuda"]:
                    data, target = data.cuda(), target.cuda()
                start_time = time.time()
                with torch.no_grad():
                    output = model(data)
                stop_time = time.time()
                times += (stop_time - start_time)
            total_times += times
            print("{} epoch finish. time: {}. Pure inference time: {}".format(epoch, time.time() - epoch_time, times))

        num = len(test_loader.dataset) * (epochs - start_epoch)
        print(
            "\nAll time: {}, \nImage num: {}, \nTime per image: {}".format(total_times, num, total_times / float(num)))

    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        pred_list = torch.Tensor()
        true_list = torch.Tensor()
        for data, target in test_loader:
            data, target = data.cuda(), target.long().cuda()
            with torch.no_grad():
                output = model(data)
                test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred_list = torch.cat((pred_list, pred.squeeze().cpu()), 0)
            true_list = torch.cat((true_list, target.cpu()), 0)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # get target have the same shape of pred

        report = classification_report(true_list, pred_list, labels=range(6), digits=4)
        print(report)

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return report

    # test(0)
    test_fps(0, 120)
    # validator = ImagenetValidator(model, get_imagenet_criterion())
    # logger = ValidationLogger(1, None)
    # validate_one_epoch(0, val_data_loader, model, validator, logger, verbose)


if __name__ == "__main__":
    main()
