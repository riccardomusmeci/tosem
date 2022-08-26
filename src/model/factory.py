
import segmentation_models_pytorch as smp

FACTORY = {
    "unet": smp.Unet,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus
}