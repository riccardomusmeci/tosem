from typing import List, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import JaccardIndex as IoU


class SegmentationModelModule(pl.LightningModule):
    """Segmentation Lightning Module.

    Args:
        model (str): segmentation model namypy --install-typesme
        num_classes (int): num classes to predict
        loss (_Loss): loss
        optimizer (Optimizer): optimizer
        lr_scheduler (_LRScheduler, optional): learning rate scheduler. Defaults to None.
        mode (str): either binary, multiclass, or multilabel
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        loss: _Loss,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        mode: str,
    ) -> None:
        assert mode in [
            "binary",
            "multiclass",
            "multilabel",
        ], "mode must be one of binary, multiclass, multilabel"

        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        if num_classes == 1:
            self.num_classes = num_classes + 1  # for the background
        else:
            self.num_classes = num_classes
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # 0 is background
        self.IoU = IoU(task=mode, num_classes=self.num_classes, ignore_index=0)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        # Output a tensor of shape (batch size, num classes, height, width)
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch (torch.Tensor): batch of images and masks
            batch_idx (int): batch index

        Returns:
            torch.Tensor: loss
        """
        x, mask = batch
        # Unnormalized scores
        logits = self(x)
        # try to prevent memory leak
        del batch
        # Loss
        loss = self.loss(y_pred=logits, y_true=mask)
        self.log("loss/train", loss, sync_dist=True)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Validation step.

        Args:
            batch (torch.Tensor): batch of images and masks
            batch_idx (int): batch index
        """
        x, mask = batch

        logits = self(x)
        preds = self.preds_from_logits(logits)

        # loss + IoU
        loss = self.loss(y_pred=logits, y_true=mask)

        self.log("loss_val", loss, sync_dist=True)
        iou = self.IoU(target=mask, preds=preds)  # type: ignore
        self.log("IoU_val", iou, prog_bar=True)

    def preds_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute predictions from logits.

        Args:
            logits (torch.Tensor): logits

        Returns:
            torch.Tensor: predictions
        """
        if self.num_classes == 2:
            preds = torch.sigmoid(logits)
        else:
            preds = torch.softmax(logits, dim=1)
            preds = torch.argmax(preds, dim=1, keepdim=True)
        return preds

    def configure_optimizers(
        self,
    ) -> Union[List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]:
        """Configure optimizer and lr scheduler.

        Returns:
            Union[List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]: optimizer and lr scheduler
        """
        if self.lr_scheduler is None:
            return [self.optimizer]  # type: ignore
        else:
            return [self.optimizer], [self.lr_scheduler]
