import ssl
import torch
import segmentation_models_pytorch as smp

ssl._create_default_https_context = ssl._create_unverified_context

models = {
    "unet": smp.Unet,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus
}

def segmentation_model(
    arch: str, 
    backbone: str, 
    num_classes: int, 
    in_channels: int = 3, 
    weights: str = "imagenet",
    ) -> torch.nn.Module:
    """returns a segmentation model

    Args:
        arch (str): segmentation model name (e.g. Unet, DeepLab)
        backbone (str): choose backbone for the model encoder, e.g. mobilenet_v2 or efficientnet-b3
        num_classes (int): model output channels (number of classes in your dataset).
        in_channels (int, optional): model input channels (1 for gray-scale images, 3 for RGB, etc.). Defaults to 3.
        weights (str, optional): pre-trained weights for encoder initialization. Defaults to "imagenet".

    Returns:
        smp.SegmentationModel: segmentation model
    """
    print(f"> Creating segmentation model {arch} - backbone {backbone} - weights from {weights} - num_classes {num_classes}")
    _model = models[arch](
        encoder_name=backbone,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    
    return _model