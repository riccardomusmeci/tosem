import torch.nn as nn
from torch.optim import SGD, AdamW

def get_optimizer(params: nn.Module, algo: str, **kwargs):
    
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
    