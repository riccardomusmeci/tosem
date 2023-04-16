from pathlib import Path
from typing import Dict

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from tosem.utils import get_device

__all__ = ["create_model", "list_models"]

_FACTORY = {
    "unet": smp.Unet,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus,
    "unet++": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "pan": smp.PAN,
}


def create_model(
    model_name: str,
    encoder_name: str,
    num_classes: int = 1,
    in_channels: int = 3,
    weights: str = "imagenet",
    ckpt_path: str = None,
    verbose: bool = True,
) -> nn.Module:
    """Create segmentation model

    Args:
        Args:
        model_name (str): segmentation model name
        encoder_name (str): model backbone from timm (e.g. resnet18)
        num_classes (int, optional): num of segmentation classes. Defaults to 1.
        in_channels (int, optional): input channels. Defaults to 3.
        weights (str, optional): segmentation model weights. Defaults to "imagenet".
        ckpt_path (str, optional): custom checkopoint path to load. Defaults to None.
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
        print("> Model recap:")
        print(f"\t- Architecture: {model_name.upper()}")
        print(f"\t- Backbone: {encoder_name}")
        print(f"\t- Weights from: {weights}")
        print(f"\t- Num classes: {num_classes}")
        print(f"\t- State Dict from: {ckpt_path}")

    return model


def list_models():
    print("Available Models:")
    for k, v in _FACTORY.items():
        print(f"> {k} - {v}")


def load_state_dict(ckpt_path: Path, verbose: bool = True) -> Dict:
    """Load a state dict from ckpt file

    Args:
        ckpt_path (str): checkpoint path
        verbose (bool, optional): verbose mode. Defaults to True.

    Returns:
        Dict: model state dict (layer-weights)
    """
    ckpt_state_dict = torch.load(ckpt_path, map_location=get_device())["state_dict"]
    state_dict = {}
    for key, weights in ckpt_state_dict.items():
        model_key = key.replace("model.", "")
        try:
            state_dict[model_key] = weights
        except Exception as e:
            print(f"Error while copying weights from layer {model_key}. Error: {e}")
            quit()
    return state_dict
