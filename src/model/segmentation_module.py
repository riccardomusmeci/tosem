import torch
from torch import logit, nn
from torchmetrics import IoU
import pytorch_lightning as pl
from torch.optim import Optimizer
from typing import List, Tuple, Union
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler


class RoadSegmentationModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        loss: _Loss,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler = None,
    ) -> None:
        """Road Segmentation Lightning Module

        Args:
            model (str): segmentation model name
            num_classes (int): num classes to predict
            loss (_Loss): loss
            optimizer (Optimizer): optimizer
            lr_scheduler (_LRScheduler, optional): Optional; Learning rate scheduler. Defaults to None.
        """
        
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.num_classes = num_classes + 1 # for the background
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        # 0 is backgrounf
        self.IoU = IoU(num_classes=self.num_classes, ignore_index=0) 
        self.classwise_IoU = IoU(num_classes=self.num_classes, reduction="none")

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
        iou = self.IoU(preds, mask)
        
        self.log("loss/val", loss, sync_dist=True)
        self.log("IoU/all/val", iou, prog_bar=True)
        for class_idx, value in enumerate(self.classwise_IoU(preds, mask)):
            self.log(f"IoU/class-{class_idx}/val", value, prog_bar=True, sync_dist=True)

    def preds_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 2:
            preds = torch.sigmoid(logits)
        else:
            preds = torch.softmax(logits, dim=1)

        return preds

    def configure_optimizers(self,) -> Union[List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]:
        if self.lr_scheduler is None:
            return [self.optimizer]
        else:
            return [self.optimizer], [self.lr_scheduler]