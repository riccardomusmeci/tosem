from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, _LRScheduler

def lr_scheduler(
    optimizer: Optimizer, 
    algorithm: str, 
    t_0: int, 
    t_mult: int, 
    eta_min: float
) -> _LRScheduler:
    """returns learning rate scheduler

    Args:
        optimizer (Optimizer): optimizer
        algo (str): algorithm name (e.g. cosine_annealing_warm)
        t_0 (int): first epoch to restart learningrate
        t_mult (int): multiplicative factore between restart
        eta_min (float): min learning rate

    Returns:
        _LRScheduler: scheduler
    """
    if algorithm == "cosine_annealing_warm":
        return CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=t_0,
            T_mult=t_mult, 
            eta_min=eta_min,
        )
    
    print("Only CosineAnnealingWarmRestarts is implemented. Quitting.")
    quit()