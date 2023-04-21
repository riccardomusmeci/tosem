from typing import List, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import JaccardIndex as IoU


class SegmentationModelModule(pl.LightningModule):
    """Segmentation Model Lightning Module

    Args:
        model (str): segmentation model name
        num_classes (int): num classes to predict
        loss (_Loss): loss
        optimizer (Optimizer): optimizer
        lr_scheduler (_LRScheduler, optional): Optional; Lesarning rate scheduler. Defaults to None.
        IoU_task (str): either binary, multiclass, or multilabel
        IoU_ignore_index (int): class index to ignore to compute IoU. Defaults to 0.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        loss: _Loss,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        IoU_task: str,
        IoU_ignore_index: int = 0,
    ) -> None:

        assert IoU_task in [
            "binary",
            "multiclass",
            "multilabel",
        ], "IoU_task must be one of binary, multiclass, multilabel"

        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        if num_classes == 1:
            print("[WARNING] num_classes set to 1, but must inlcude also background. Forcing to 2.")
            self.num_classes = 2
        else:
            self.num_classes = num_classes
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # 0 is background
        self.IoU = IoU(task=IoU_task, num_classes=self.num_classes, ignore_index=IoU_ignore_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output a tensor of shape (batch size, num classes, height, width)
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """Training step

        Args:
            batch (torch.Tensor): batch of images and masks
            batch_idx (int): batch index

        Returns:
            float: loss value
        """
        x, mask = batch
        # Unnormalized scores
        logits = self(x)
        # Loss
        loss = self.loss(logits, mask)
        self.log("loss/train", loss, sync_dist=True)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Validation step

        Args:
            batch (torch.Tensor): batch of images and masks
            batch_idx (int): batch index

        """
        x, mask = batch
        logits = self(x)
        preds = self.preds_from_logits(logits)

        # loss + IoU
        loss = self.loss(logits, mask)

        self.log("loss_val", loss, sync_dist=True)
        self.IoU.update(preds, mask)

    def preds_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 2:
            preds = torch.sigmoid(logits)
        else:
            preds = torch.softmax(logits, dim=1)
        preds = torch.argmax(preds, dim=1, keepdim=True)
        return preds

    def on_validation_epoch_end(self) -> None:
        iou = self.IoU.compute()
        self.IoU.reset()
        self.log("IoU_val", iou, prog_bar=True)

    def configure_optimizers(
        self,
    ) -> Union[List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]:
        if self.lr_scheduler is None:
            return [self.optimizer]
        else:
            return [self.optimizer], [self.lr_scheduler]
