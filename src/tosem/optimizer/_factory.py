from typing import Dict, Iterator, List

from torch.optim import SGD, Adam, AdamW, Optimizer

_FACTORY = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
}

__all__ = ["create_optimizer", "list_optimizers"]


def list_optimizers() -> List[str]:
    """List available optimizers.

    Returns:
        list: list of available optimizers
    """
    return list(_FACTORY.keys())


def create_optimizer(params: Iterator, optimizer: str, **kwargs) -> Optimizer:  # type: ignore
    """Create optimizer.

    Args:
        params (Dict): model parameters
        optimizer (str): optimizer name

    Returns:
        Optimizer: optimizer
    """

    if optimizer.lower() not in _FACTORY.keys():
        raise ValueError(f"Optimizer {optimizer} not supported. Choose one of {list(_FACTORY.keys())}")
    else:
        return _FACTORY[optimizer.lower()](params, **kwargs)
