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
import time

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
    model = compress_model(model, **compression_config).cuda()
    compressed_model_size_bits = compute_model_nbits(model)
    log_compression_ratio(uncompressed_model_size_bits, compressed_model_size_bits)

    model = load_state_dict(model, os.path.join(file_path, config["model"]["state_dict_compressed"]))

    dataloader_config = config["dataloader"]
    # val_data_sampler, val_data_loader = load_imagenet_val(
    #     dataloader_config["imagenet_path"],
    #     dataloader_config["num_workers"],
    #     dataloader_config["batch_size"],
    #     shuffle=dataloader_config["validation_shuffle"],
    # )
    kwargs = {'num_workers': 4, 'pin_memory': True}

    _, train_loader, _ = get_loaders(
        image_dir=dataloader_config["imagenet_path"],
        batch_size=8,
        img_shape=dataloader_config["image_shape"],
        radio=[0.2, 0.7, 0.1],
        **kwargs
    )

    def test_fps(start_epoch, epochs):
        model.eval()
        print("{} epochs will be test.".format(epochs - start_epoch))
        start_time = time.time()
        for epoch in range(start_epoch, epochs):
            print("{} epoch start......".format(epoch))
            epoch_time = time.time()
            for data, target in train_loader:
                data, target = data.cuda(), target.cuda()
                with torch.no_grad():
                    output = model(data)
            print("{} epoch finish. time: {}".format(epoch, time.time() - epoch_time))
        stop_time = time.time()
        times = stop_time - start_time
        num = len(train_loader.dataset) * (epochs - start_epoch)
        print("\nAll time: {}, \nImage num: {}, \nTime per image: {}".format(times, num, times/float(num)))

    test_fps(0, 200)
    # validator = ImagenetValidator(model, get_imagenet_criterion())
    # logger = ValidationLogger(1, None)
    # validate_one_epoch(0, val_data_loader, model, validator, logger, verbose)


if __name__ == "__main__":
    main()
