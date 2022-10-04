import ssl
import torch
from typing import Dict
from src.model.factory import FACTORY

ssl._create_default_https_context = ssl._create_unverified_context

def create_model(
    model_name: str, 
    backbone: str, 
    num_classes: int = 1, 
    in_channels: int = 3, 
    weights: str = "imagenet",
) -> torch.nn.Module:
    """returns a segmentation model

    Args:
        model_name (str): segmentation model name (e.g. Unet, DeepLab)
        backbone (str): choose backbone for the model encoder, e.g. mobilenet_v2 or efficientnet-b3
        num_classes (int, optional): model output channels (number of classes in your dataset). Defaults to 1.
        in_channels (int, optional): model input channels (1 for gray-scale images, 3 for RGB, etc.). Defaults to 3.
        weights (str, optional): pre-trained weights for encoder initialization. Defaults to "imagenet".

    Returns:
        smp.SegmentationModel: segmentation model
    """
    assert model_name in FACTORY.keys(), f"{model_name} is not supported. Only {list(FACTORY.keys())} available."
    
    print(f"##### Segmentation Model #####")
    print(f"> Architecture: {model_name.upper()}")
    print(f"> Backbone: {backbone}")
    print(f"> Weights from: {weights}")
    print(f"- Num classes: {num_classes}")
    print("###############################")
    return FACTORY[model_name](
        encoder_name=backbone,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    
def load_state_dict(ckpt_path: str) -> Dict:
    """laods a state dict from ckpt file

    Args:
        ckpt_path (str): ckpt path

    Returns:
        Dict: model state dict (weights)
    """
    device = torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')
    ckpt_state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
    state_dict = {}
    for key, weights in ckpt_state_dict.items():
        l = key.replace("model.", "")
        state_dict[l] = weights
    return state_dict