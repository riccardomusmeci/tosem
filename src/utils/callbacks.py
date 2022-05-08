from fileinput import filename
import pytorch_lightning
from typing import List
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def get_callbacks(
    output_dir: str,
    filename: str,
    monitor: str,
    verbose: bool,
    save_top_k: int,
    mode: str,
    auto_insert_metric_name: bool,
    log_momentum: bool
     
    ) -> List[pytorch_lightning.Callback]:
    
    model_checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename=filename,
        monitor=monitor,
        verbose=verbose,
        mode=mode,
        save_top_k=save_top_k,
        auto_insert_metric_name=auto_insert_metric_name
    )
    
    learning_rate_monitor = LearningRateMonitor(
        log_momentum=log_momentum
    )
    
    return [model_checkpoint, learning_rate_monitor]