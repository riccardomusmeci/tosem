import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="timm-efficientnet-b3",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=10,                      # model output channels (number of classes in your dataset)
)

import torch

x = torch.rand((1, 3, 512, 512))
out = model(x)
print(out.shape)
quit()

