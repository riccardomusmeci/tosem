import torch
from torch import nn
from pytorch_toolbelt.losses import JaccardLoss

def loss_fn(criterion: str, **kwargs) -> nn.Module:
    """returns loss for segmentation model

    Args:
        fn (str): loss function

    Returns:
        nn.Module: loss module
    """
    if criterion == "jaccard":
        return JaccardLoss(
            **kwargs
        )
    else:
        print("Only Jaccard is implemented. Quitting.")
        quit()