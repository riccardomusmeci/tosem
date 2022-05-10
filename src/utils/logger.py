import os
from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase

def get_logger(output_dir: str) -> LightningLoggerBase:
    """returns a logger

    Args:
        output_dir (str): output dir
        name (str): name
        version (str): versione

    Returns:
        pytorch_lightning.loggers.LightningLoggerBase: logger
    """
    return TensorBoardLogger(
        save_dir=os.path.join(output_dir, "tensorboard")
    )