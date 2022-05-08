from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.optim import Optimizer

def get_scheduler(optimizer: Optimizer, algo: str, t_0: int, t_mult: int, eta_min: float):
    
    if algo == "cosine_annealing_warm":
        return CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=t_0,
            T_mult=t_mult, 
            eta_min=eta_min,
        )
    print("Only CosineAnnealingWarmRestarts is implemented. Quitting.")