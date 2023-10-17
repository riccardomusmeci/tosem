from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

_FACTORY = {
    "cosine_annealing_warm_restarts": CosineAnnealingWarmRestarts,
    "cosine_annealing": CosineAnnealingLR,
    "reduce_on_plateau": ReduceLROnPlateau,
    "one_cycle": OneCycleLR,
    "lambda_lr": LambdaLR,
    "step_lr": StepLR,
    "multi_step_lr": MultiStepLR,
    "exponential_lr": ExponentialLR,
    "cyclic_lr": CyclicLR,
}
__all__ = ["list_lr_schedulers", "create_lr_scheduler"]


def list_lr_schedulers() -> List[str]:
    """List available schedulers.

    Returns:
        list: list of available schedulers
    """
    return list(_FACTORY.keys())


def create_lr_scheduler(optimizer: Optimizer, lr_scheduler: str, **kwargs):  # type: ignore
    """Create lr scheduler.

    Args:
        optimizer (Optimizer): optimizer
        lr_scheduler (str): lr scheduler name

    Returns:
        lr_scheduler: lr scheduler
    """
    if lr_scheduler.lower() not in _FACTORY.keys():
        raise ValueError(f"LR scheduler {lr_scheduler} not supported. Choose one of {list(_FACTORY.keys())}")
    else:
        return _FACTORY[lr_scheduler.lower()](optimizer=optimizer, **kwargs)
