# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from abc import abstractmethod
from typing import Optional, Tuple
import logging

import torch
from tensorboardX import SummaryWriter

from ..utils.logging import log_to_summary_writer
from .AbstractDataHandler import AbstractDataHandler
from .AbstractDataLogger import AbstractDataLogger
from .lr_scheduler import LR_Scheduler
from .training_types import FinalSummary, IntermediateSummary, TQDMState, Summary


class ModelTrainer(AbstractDataHandler):
    latest_state = {}

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler: LR_Scheduler):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def update(self, data: Tuple[torch.Tensor, torch.Tensor], device: torch.device) -> None:
        """Does one training step on a batch of data

        Parameters:
            data: A tuple with inputs and expected outputs
            device: CPU or GPU
        """

        # inputs = data[0].cuda(non_blocking=True)
        # targets = data[1].cuda(non_blocking=True)
        inputs = data[0].to(device, non_blocking=True)
        targets = data[1].long().to(device, non_blocking=True)

        self.optimizer.zero_grad()
        outputs, loss = self.pass_to_model(inputs, targets)
        self.update_state(targets, outputs, loss)

        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler.step_batch():
            self.lr_scheduler.step()

    @abstractmethod
    def pass_to_model(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def update_state(self, targets: torch.Tensor, outputs, loss: torch.Tensor):
        pass

    def get_tqdm_state(self) -> TQDMState:
        state = self.latest_state
        return TQDMState({
            "loss": f'{state["loss"]:.2f}',
            "accuracy": f'{state["acc"]:.2f}',
            # "iou(0,1)": f'({state["iou"].item(0):.2f},{state["iou"].item(1):.2f})',
            # "miou": f'{state["miou"]:.2f}',
        })

    def get_intermediate_summary(self) -> IntermediateSummary:
        return IntermediateSummary(Summary({
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **self.latest_state
        }))

    def get_final_metric(self) -> float:
        return -math.inf  # unused value


class TrainingLogger(AbstractDataLogger):
    def __init__(self, summary_writer: Optional[SummaryWriter]):
        super().__init__("Training")
        self.summary_writer = summary_writer

    def log_intermediate_summary(self, idx: int, summary: IntermediateSummary):
        # summary["iou-0"] = summary["iou"].item(0)
        # summary["iou-1"] = summary["iou"].item(1)
        # del summary["iou"]
        log_to_summary_writer("Train", idx, summary, self.summary_writer)

    def log_final_summary(self, epoch: int, summary: FinalSummary):
        statement = ", ".join(f"{k}: {v:.4f}" for k, v in summary.items())
        logging.info(f'{self.get_desc("Epoch", epoch)}: {statement}')


@torch.enable_grad()
def train_one_epoch(
    epoch: int,
    train_data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    trainer: ModelTrainer,
    logger: TrainingLogger,
    verbose: bool,
    device: torch.device,
    train_sampler: Optional[torch.utils.data.DistributedSampler] = None,
) -> None:
    """Perform one epoch of training given a model, a trainer, and possibly writing to tensorboard

    Parameters:
        epoch: Current epoch count
        train_data_loader: PyTorch dataloader for the training set
        model: Network to train
        trainer: Model trainer
        logger: Training logger
        verbose: Whether to write to logs
        device: GPU or CPU
        train_sampler: distributed multi GPU training
    """
    model.train()
    trainer.reset()

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    trainer.handle_data(epoch, train_data_loader, logger, verbose, device)
