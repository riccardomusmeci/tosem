
import torch
from torch import nn
from torchmetrics import IoU
import pytorch_lightning as pl
from torch.optim import Optimizer
from typing_extensions import Literal
from torch.nn.modules.loss import _Loss
from typing import List, Optional, Tuple, Union
from torch.optim.lr_scheduler import _LRScheduler


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        loss: _Loss,
        optimizer: Optimizer,
        lr_scheduler: Optional[Union[LRDict, _LRScheduler]] = None,
    ) -> None:
        """Lightning Segmentation Module.

        Args:
            model: Model.
            num_classes: Number of classes, including background (minimum 2).
            loss: Loss.
            optimizer: Optimizer.
            lr_scheduler: Optional; Learning rate scheduler or dictionary containing the
              scheduler and other parameters such as interval and frequency.
        """
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2, remember to consider the background "
                f"too (got {num_classes})."
            )

        self.model = model
        self.num_classes = num_classes
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.configure_metrics()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Output a tensor of shape (batch size, num classes, height, width)
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        x, y = batch
        # Unnormalized scores
        out = self(x)
        # Loss
        loss = self.loss(out, y)
        self.log("loss/train", loss, sync_dist=True)
        return loss

    def evaluate(self, batch: torch.Tensor, stage: Literal["val", "test"]) -> None:
        x, y = batch
        # Logits (unnormalized predictions)
        logits = self(x)
        # Predictions
        preds = self.preds_from_logits(logits)
        # Loss
        loss = self.loss(logits, y)
        self.log(f"loss/{stage}", loss, sync_dist=True)
        # IoU
        iou = getattr(self, f"{stage}_iou")
        iou(preds, y)
        self.log(f"iou/all/{stage}", iou, prog_bar=True)
        # Class-wise IoU
        classwise_iou = getattr(self, f"{stage}_classwise_iou")
        for class_idx, value in enumerate(classwise_iou(preds, y)):
            self.log(f"iou/{class_idx}/{stage}", value, prog_bar=True, sync_dist=True)

    def validation_step(  # type: ignore
        self, batch: torch.Tensor, batch_idx: int
    ) -> None:
        self.evaluate(batch, stage="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:  # type: ignore
        self.evaluate(batch, stage="test")

    def predict_step(  # type: ignore
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = None
    ) -> torch.Tensor:
        # Logits (unnormalized predictions)
        logits = self(batch)
        # Predictions
        preds = self.preds_from_logits(logits)
        return preds

    def preds_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 2:
            preds = torch.sigmoid(logits)
        else:
            preds = torch.softmax(logits, dim=1)

        return preds

    def configure_metrics(self) -> None:
        self.val_iou = IoU(num_classes=self.num_classes)
        self.test_iou = IoU(num_classes=self.num_classes)
        self.val_classwise_iou = IoU(num_classes=self.num_classes, reduction="none")
        self.test_classwise_iou = IoU(num_classes=self.num_classes, reduction="none")

    def configure_optimizers(
        self,
    ) -> Union[
        List[Optimizer], Tuple[List[Optimizer], List[Union[LRDict, _LRScheduler]]]
    ]:
        if self.lr_scheduler is None:
            return [self.optimizer]
        else:
            return [self.optimizer], [self.lr_scheduler]