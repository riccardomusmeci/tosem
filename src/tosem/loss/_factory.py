from typing import List

import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, JaccardLoss
from torch.nn.modules.loss import _Loss

_FACTORY = {"jaccard": JaccardLoss, "dice": DiceLoss, "focal": FocalLoss}

__all__ = ["list_criteria", "create_criterion"]


def list_criteria() -> List[str]:
    """List available criteria.

    Returns:
        list: list of available criteria
    """
    return list(_FACTORY.keys())


def create_criterion(loss: str, **kwargs) -> _Loss:  # type: ignore
    """Create criterion.

    Args:
        loss (str): loss name

    Raises:
        ValueError: if loss is not supported

    Returns:
        _Loss: criterion
    """
    if loss.lower() not in _FACTORY.keys():
        raise ValueError(f"Loss {loss} not supported. Choose one of {list(_FACTORY.keys())}")
    else:
        return _FACTORY[loss.lower()](**kwargs)
