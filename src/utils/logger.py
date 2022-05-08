from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase

def get_logger(output_dir: str, name: str, version: str):
    """returns a logger

    Args:
        output_dir (str): output dir
        name (str): name
        version (str): versione

    Returns:
        pytorch_lightning.loggers.LightningLoggerBase: logger
    """
    return TensorBoardLogger(
        save_dir=output_dir,
        name=name,
        version=version
    )