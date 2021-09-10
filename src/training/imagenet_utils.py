# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
from .training import ModelTrainer
from .training_types import FinalSummary, TQDMState, Summary
from .validating import ModelValidator

# fmt: off
try:
    import horovod.torch as hvd
    HAVE_HOROVOD = True
except ImportError:
    HAVE_HOROVOD = False
# fmt: on


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass        #不包含背景的类别
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵（空）
        self.iou = np.zeros((self.numClass), float)
        self.acc = 0.
        self.loss = np.array([0.])
        self.count = 0
        self.miou = 0.

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel, loss):
        imgPredict = imgPredict[1]      #只用在hrnet,它有两个输出

        # output = pred.cpu().numpy().transpose(0, 2, 3, 1)
        # seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)

        imgPredict = imgPredict.cpu().detach().numpy()
        preds = np.asarray(np.argmax(imgPredict,axis=1), np.int32)
        y = imgLabel.cpu().detach().numpy()

        assert preds.shape == y.shape
        # predicted = torch.sigmoid(imgPredict)
        # predicted = (predicted>0.5).float()
        # preds, y = predicted.cpu().numpy(), imgLabel.cpu().numpy()
        preds, y = preds.astype(np.int32), y.astype(np.int32)
        self.confusionMatrix += self.genConfusionMatrix(preds, y)  # 得到混淆矩阵
        iou = self.IntersectionOverUnion()
        acc = self.pixelAccuracy()

        self.iou += iou
        self.miou += np.nanmean(iou).item()
        self.count += 1
        self.loss += np.array(loss.item())
        self.acc += acc

        self._latest_state = {
            "loss": loss.item(),
            "acc": acc,
            "iou": iou,
            "miou": np.nanmean(iou[1:]).item()
        }
        return self.confusionMatrix

    def get_latest_state(self):
        return self._latest_state

    def get_average_state(self):
        return {
            "loss": (self.loss/self.count).item(),
            "acc": self.acc/self.count,
            "iou": self.iou/self.count,
            "miou": self.miou/self.count
        }

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
        self.iou = np.zeros((self.numClass), float)
        self.acc = 0.
        self.loss = np.array([0.])
        self.count = 0
        self.miou = 0.


class ImagenetAccumulator(object):
    """Horovod-aware accumulator that keeps track of accuracy so far"""

    def __init__(self):
        self._n_total = 0
        self._n_correct = 0
        self._total_loss = 0.
        self._count = 0

    def reset(self):
        self._n_total = 0
        self._n_correct = 0
        self._total_loss = 0.
        self._count = 0

    def accumulate(self, targets: torch.Tensor, outputs: torch.Tensor, loss: torch.Tensor):
        """Updates the number of correct predictions and average loss so far.

        Parameters
            targets: The expected classes of the inputs to the model
            outputs: The classes that the model predicted
            loss: The loss that the model incurred on making its predictions
        """

        targets = targets.detach()
        outputs = outputs.detach()
        loss = loss.detach().cpu()

        _, predicted = outputs.max(dim=1)
        n_total = torch.tensor(targets.size(0), dtype=torch.float)
        n_correct = predicted.eq(targets).cpu().float().sum()

        if HAVE_HOROVOD:
            n_total = hvd.allreduce(n_total, average=False, name="accum_n_total")
            n_correct = hvd.allreduce(n_correct, average=False, name="accum_n_correct")
            loss = hvd.allreduce(loss, average=True, name="accum_loss")

        n_total = round(n_total.item())
        n_correct = round(n_correct.item())

        self._n_total += n_total
        self._n_correct += n_correct
        self._total_loss += loss.item()
        self._count += 1

        self._latest_state = {
            "loss": loss.item(),
            "acc": n_correct / n_total * 100,
            "correct": n_correct,
            "total": n_total,
        }

    def get_latest_state(self):
        return self._latest_state

    def get_average_state(self):
        return {
            "loss": self._total_loss / self._count,
            "acc": self._n_correct / self._n_total * 100,
            "correct": self._n_correct,
            "total": self._n_total,
        }


class ImagenetValidator(ModelValidator):
    def __init__(self, model, criterion):
        super().__init__()
        # self._accumulator = ImagenetAccumulator()
        self._accumulator = SegmentationMetric(2)

        self._model = model
        self._criterion = criterion

    def reset(self):
        self._accumulator.reset()

    def update(self, data: Tuple[torch.Tensor, torch.Tensor], device: torch.device):
        """Computes loss between what the model produces and the ground truth, and updates the accumulator

        Parameters:
            data: 2-tuple with inputs and targets for the model
            device: CPU or GPU
        """

        # inputs = data[0].cuda(non_blocking=True)
        # targets = data[1].cuda(non_blocking=True)
        inputs = data[0].to(device,non_blocking=True)
        targets = data[1].long().to(device,non_blocking=True)

        outputs = self._model(inputs)
        loss = self._criterion(outputs, targets)

        self._accumulator.addBatch(outputs, targets,  loss)

    def get_tqdm_state(self):
        state = self._accumulator.get_average_state()
        return TQDMState({
            "loss": f'{state["loss"]:.2f}',
            "accuracy": f'{state["acc"]:.2f}',
            "iou(1,2)": f'({state["iou"].item(1):.2f},{state["iou"].item(2):.2f})',
            "miou": f'{state["miou"]:.2f}',
        })

    def get_final_summary(self):
        state = self._accumulator.get_average_state()
        return FinalSummary(Summary({"loss": state["loss"],
                                     "accuracy": state["acc"],
                                     "iou-1": state["iou"].item(1),
                                     "iou-2": state["iou"].item(2),
                                     "miou": state["miou"]}))

    def get_final_metric(self):
        state = self._accumulator.get_average_state()
        return state["miou"]


class ImagenetTrainer(ModelTrainer):
    def __init__(self, model, optimizer, lr_scheduler, criterion):
        super().__init__(model, optimizer, lr_scheduler)
        self.accumulator = SegmentationMetric(3)
        self.criterion = criterion

    def reset(self):
        self.accumulator.reset()

    def pass_to_model(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        return outputs, loss

    def update_state(self, targets: torch.Tensor, outputs: torch.Tensor, loss: torch.Tensor):
        self.accumulator.addBatch(outputs, targets, loss)
        state = self.accumulator.get_latest_state()
        # self.latest_state = {"loss": state["loss"], "accuracy": state["acc"], "miou": state["miou"]}
        self.latest_state = state

    def get_final_summary(self):
        state = self.accumulator.get_average_state()
        return FinalSummary(Summary({"loss": state["loss"],
                                     "accuracy": state["acc"],
                                     "iou-1": state["iou"].item(1),
                                     "iou-2": state["iou"].item(2),
                                     "miou": state["miou"]}))


# Dice损失函数
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "The size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        return score


def get_imagenet_criterion():
    """Gets the typical training loss for Imagenet classification"""
    # return torch.nn.CrossEntropyLoss()
    return DiceLoss()
