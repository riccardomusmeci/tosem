import torch.nn as nn
from torch.optim import SGD, AdamW, Optimizer

def get_optimizer(params: nn.Module, algo: str, **kwargs) -> Optimizer:
    """returns optimizer algorithm

    Args:
        params (nn.Module): model params
        algo (str): optimizer algorithm (sgd, adamw)

    Returns:
        Optimizer: optimizer
    """
    if algo=="sgd":
        return SGD(
            params=params,
            **kwargs
        )
    
    if algo=="adamw":
        return AdamW(
            params=params,
            **kwargs
        )
    
    print("Only SGD and AdamW are implemented. Quitting.")
    quit()
    