from pathlib import Path
from typing import Dict, List, Optional, Union

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from ..utils import get_device

_FACTORY = {"unet": smp.Unet, "deeplabv3": smp.DeepLabV3, "deeplabv3+": smp.DeepLabV3Plus}


def list_models() -> List[str]:
    """List available models.

    Returns:
        List: list of available models
    """
    return list(_FACTORY.keys())


def create_model(
    model_name: str,
    encoder_name: str,
    num_classes: int = 1,
    in_channels: int = 3,
    weights: str = "imagenet",
    ckpt_path: Optional[Union[Path, str]] = None,
    verbose: bool = True,
) -> nn.Module:
    """Create segmentation model.

    Args:
        Args:
        model_name (str): segmentation model name
        encoder_name (str): model backbone from timm (e.g. resnet18)
        num_classes (int, optional): num of segmentation classes. Defaults to 1.
        in_channels (int, optional): input channels. Defaults to 3.
        weights (str, optional): segmentation model weights. Defaults to "imagenet".
        ckpt_path (Optional[Union[Path, str]], optional): custom checkopoint path to load. Defaults to None.
        verbose (bool, optional): verbose mode. Defaults to True.
    """

    assert model_name in _FACTORY.keys(), f"{model_name} is not supported. Available models: {list(_FACTORY.keys())}."

    num_classes = num_classes
    model = _FACTORY[model_name](
        encoder_name=encoder_name,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=num_classes,
    )

    if ckpt_path is not None:
        state_dict = load_state_dict(ckpt_path=ckpt_path, verbose=verbose)
        model.load_state_dict(state_dict)

    if verbose:
        print("##### Segmentation Model #####")
        print(f"> Architecture: {model_name.upper()}")
        print(f"> Backbone: {encoder_name}")
        print(f"> Weights from: {weights}")
        print(f"> Num classes: {num_classes}")
        print(f"> State Dict from: {ckpt_path}")
        print("###############################")

    return model


def load_state_dict(ckpt_path: Union[Path, str], verbose: bool = True) -> Dict:
    """Load a state dict from ckpt file.

    Args:
        ckpt_path (Union[Path, str]): ckpt path

    Returns:
        Dict: model state dict (weights)
    """
    device = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
    ckpt_state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
    state_dict = {}
    for key, weights in ckpt_state_dict.items():
        l = key.replace("model.", "")
        state_dict[l] = weights
    return state_dict
