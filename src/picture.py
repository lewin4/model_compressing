# -*- coding:utf-8 -*-
import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
from PIL import Image
from tqdm import tqdm
import os

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f,encoding='bytes')
        #X = datadict[b'data']
        #Y = datadict[b'labels']
        #X = X.reshape(10000, 3, 32, 32)
        X = datadict[b'data']

        Y = datadict[b'coarse_labels']+datadict[b'fine_labels']
        X = X.reshape(50000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


if __name__ == "__main__":
    #imgX, imgY = load_CIFAR_batch("./cifar-10-batches-py/data_batch_1")
    imgX, imgY = load_CIFAR_batch(r"D:\Code\data\cifar-100\cifar-100-python\train")
    print(imgX.shape,type(imgX))
    imgY = imgY[:50000]
    print(imgY.shape,type(imgY))
    print("正在保存图片:")
    # 创建文件夹
    for i in range(20):
        dirname = r"D:\Code\data\cifar-100" +"\\" +str(i)
        if  not os.path.exists(dirname):
            os.mkdir(dirname)

    for i in tqdm(range(imgX.shape[0])):
        imgs = imgX[i]
    #     if i < 100:#只循环100张图片,这句注释掉可以便利出所有的图片,图片较多,可能要一定的时间
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB",(i0,i1,i2))
        label = imgY[i]
        assert label<=19 and label>=0

        name = str(i)+".png"
        img.save("../../data/cifar-100/"+ str(label) + "/"+name,"png")#文件夹下是RGB融合后的图像
        # for j in range(imgs.shape[0]):
        #     img = imgs[j]
        #     name = "img" + str(i) + str(j) + ".jpg"
        #     print("正在保存图片" + name)
        #     plimg.imsave("./pic2/" + name, img)#文件夹下是RGB分离的图像
    print("保存完毕.")

