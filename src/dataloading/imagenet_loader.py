# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Tuple
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils import data
from torch.utils.data.sampler import Sampler
from torchvision.datasets import ImageFolder

from ..utils.horovod_utils import get_distributed_sampler

IMAGENET_PATH_PLACEHOLDER = "<your_imagenet_path_here>"

# These correspond to the average and standard deviation RGB colours of Imagenet and are commonly used for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STDEV = [0.229, 0.224, 0.225]

IMAGENET_VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDEV),
    ]
)

IMAGENET_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDEV),
    ]
)


###################################
# CAT VS DOG DATASETS
###################################

class DogCatDataset(data.Dataset):

    def __init__(self, root, suffix: str, Transforms=None):

        super(DogCatDataset, self).__init__()

        #imgs保存所有图片文件的路径，成一个列表
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        np.random.seed(85)
        np.random.shuffle(imgs)         #打乱imgs的顺序

        '''
        #减少图片数量，小范围测试
        len_imgs = len(imgs)
        imgs = imgs[:int(0.1 * len_imgs)]
        '''
        len_imgs = len(imgs)

        if suffix == "test":
            self.test = True                    #创建一个test属性
            self.train = False
        elif suffix == "train":
            self.test = False
            self.train = True
        elif suffix == "val":
            self.test = False
            self.train = False
        else:
            raise ValueError(f"you should give suffix in one of 'test' 'train' 'val',but your commit is {suffix}?")




        # -----------------------------------------------------------------------------------------
        # 因为在猫狗数据集中，只有训练集和测试集，但是我们还需要验证集，因此从原始训练集中分离出30%的数据
        # 用作验证集。
        # ------------------------------------------------------------------------------------------
        if self.test:
            self.imgs = imgs
        elif self.train:
            self.imgs = imgs[: int(0.7 * len_imgs)]
        else:
            self.imgs = imgs[int(0.7 * len_imgs):]

        if Transforms is None:

            #归一化
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            #在测试或者验证的情况下执行简单的数据处理，使模型识别率提高
            if self.test or not self.train:
                self.transforms = transforms.Compose([
                    transforms.Resize(224),                   #改变图片尺寸
                    transforms.CenterCrop(224),              #中心取景
                    transforms.ToTensor(),                   #转换成tensor类型的数据
                    normalize                       #归一化
                ])
            #反之在训练的情况下进行数据增强
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize(246),
                    transforms.RandomCrop(224),              #随机取景
                    transforms.RandomHorizontalFlip(),       #随机水平翻转
                    transforms.ToTensor(),
                    normalize
                ])
        else:
            self.transforms = Transforms

    def __getitem__(self, index):

        # 当前要获取图像的路径
        img_path = self.imgs[index]
        #train\cat.123.jpg

        if self.test:
            img_label = int(img_path.split('.')[-2].split('/')[-1])
        else:
            img_label = 1 if 'dog' in img_path.split('/')[-1] else 0    #从图片名称中寻找标签0是猫,1是狗

        img_data = Image.open(img_path)             #打开图片得到Image类型的图片数据
        img_data = self.transforms(img_data)        #得到的是tensor类型的图片数据

        return img_data, img_label

    def __len__(self):
        return len(self.imgs)




def _load_imagenet(
    rootpath: str, suffix: str, transform: transforms.Compose, num_workers: int, batch_size: int, shuffle: bool = False
) -> Tuple[Sampler, DataLoader]:
    """Creates a sampler and dataloader for the imagenet dataset

    Parameters:
        roothpath: Path to the imagenet folder before `train` or `val` folder
        suffix: Either `train` or `val`. Will be appended to `rootpath`
        transform: Operations to apply to the data before passing it to the model (eg. for data augmentation)
        num_workers: Number of pytorch workers to use when loading data
        batch_size: Size of batch to give to the networks
        shuffle: Whether to randomly shuffle the data
    Returns:
        sampler: A PyTorch DataSampler that decides the order in which the data is fetched
        loader: A PyTorch DataLoader that fetches the data for the model
    """
    if rootpath == IMAGENET_PATH_PLACEHOLDER:
        raise ValueError(f"{IMAGENET_PATH_PLACEHOLDER} is not a valid path. Did you forget to update the config file?")

    # dirname = os.path.join(rootpath, suffix)
    # dataset = ImageFolder(dirname, transform)
    dataset = DogCatDataset(rootpath, suffix, transform)

    sampler = get_distributed_sampler(dataset, shuffle)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, pin_memory=True)
    return sampler, loader


def load_imagenet_val(
    rootpath: str, num_workers: int, batch_size: int, shuffle: bool = False
) -> Tuple[Sampler, DataLoader]:
    """Creates a sampler and dataloader for the training partition of Imagenet"""
    return _load_imagenet(rootpath, "val", IMAGENET_VAL_TRANSFORM, num_workers, batch_size, shuffle)


def load_imagenet_train(
    rootpath: str, num_workers: int, batch_size: int, shuffle: bool = True
) -> Tuple[Sampler, DataLoader]:
    """Creates a sampler and dataloader for the validation partition of Imagenet"""
    return _load_imagenet(rootpath, "train", IMAGENET_TRAIN_TRANSFORM, num_workers, batch_size, shuffle)
