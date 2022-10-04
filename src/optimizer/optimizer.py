import torch.nn as nn
from torch.optim import SGD, AdamW, Optimizer

def optimizer(
    params: nn.Module, 
    algorithm: str, 
    **kwargs
) -> Optimizer:
    """returns optimizer algorithm

    Args:
        params (nn.Module): model params
        algorithm (str): optimizer algorithm (sgd, adamw)

    Returns:
        Optimizer: optimizer
    """
    if algorithm=="sgd":
        return SGD(
            params=params,
            **kwargs
        )
    
    if algorithm=="adamw":
        return AdamW(
            params=params,
            **kwargs
        )
    
    print("Only SGD and AdamW are implemented. Quitting.")
    quit()
    