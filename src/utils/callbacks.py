import os
from typing import List
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def get_callbacks(output_dir: str) -> List[pytorch_lightning.Callback]:
    """Returns list of pl.Callbacks

    Args:
        output_dir (str): output dir

    Returns:
        List[pytorch_lightning.Callback]: list of pl.Callbacks
    """
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoint"),
            filename="epoch={epoch}-step={step}-val_loss={loss/val:.3f}-val_iou={IoU/all/val:.3f}",
            monitor="loss/val",
            verbose=True,
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=False
        )
    )
    
    callbacks.append(
         LearningRateMonitor(
             logging_interval="epoch",
             log_momentum=True
        )
    )
    
    return callbacks