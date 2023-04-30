from typing import List, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import FBetaScore, JaccardIndex, MeanSquaredError


class SegmentationModelModule(pl.LightningModule):
    """Segmentation Model Lightning Module

    Args:
        model (str): segmentation model name
        num_classes (int): num classes to predict
        loss (_Loss): loss
        optimizer (Optimizer): optimizer
        lr_scheduler (_LRScheduler, optional): Optional; Lesarning rate scheduler. Defaults to None.
        ignore_index (int, optional): class index to ignore to compute IoU. Defaults to 0.
        beta (float, optional): beta factor of FBetaScore. Defaults to 0.5.
        squared (bool, optional): MSE squared param. Defaults to True.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        loss: _Loss,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        ignore_index: int = 0,
        beta: float = 0.5,
        squared: bool = True,
    ) -> None:

        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.num_classes = 1 if num_classes < 3 else num_classes
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.task = "binary" if self.num_classes < 3 else "multiclass"
        # 0 is background
        self.jaccard = JaccardIndex(task=self.task, num_classes=self.num_classes, ignore_index=ignore_index)
        self.f_beta = FBetaScore(task=self.task, beta=beta, num_classes=self.num_classes, ignore_index=ignore_index)
        self.mse = MeanSquaredError(squared=squared)

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

        # if self.task == "binary":
        #     mask = mask.long()

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
        self.jaccard.update(preds, mask)
        self.f_beta.update(preds, mask)
        self.mse.update(preds, mask)

    def preds_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.task == "binary":
            preds = torch.sigmoid(logits)
        else:
            preds = torch.softmax(logits, dim=1)
            preds = torch.argmax(preds, dim=1, keepdim=True)
        return preds

    def on_validation_epoch_end(self) -> None:
        iou = self.jaccard.compute()
        fbeta = self.f_beta.compute()
        mse = self.mse.compute()

        self.jaccard.reset()
        self.f_beta.reset()
        self.mse.reset()

        self.log("IoU_val", iou, prog_bar=True)
        self.log("fbeta_val", fbeta, prog_bar=True)
        self.log("mse_val", mse, prog_bar=True)

    def configure_optimizers(
        self,
    ) -> Union[List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]:
        if self.lr_scheduler is None:
            return [self.optimizer]
        else:
            return [self.optimizer], [self.lr_scheduler]
