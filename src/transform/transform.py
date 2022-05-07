import torch
import numpy as np
import albumentations as A
from typing import Callable

def train_transforms(
    hflip_p: float = .5,
    scale_limit: float = .5,
    rotate_limit: float = 0,
    shift_limit: float = .1,
    shif_scale_rotate_p: float = 1,
    border_mode: float = 0,
    height: int = 320,
    width: int = 320, 
    gauss_noise_p: float = .5,
    perspective_p: float = .5,
    one_of_p: float = .9,
    blur_limit: float = 3,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    
    augs = [
        
        A.HorizontalFlip(p=hflip_p),
        A.ShiftScaleRotate(
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            shift_limit=shift_limit,
            p=shif_scale_rotate_p,
            border_mode=border_mode,
        ),
        A.PadIfNeeded(
            min_height=height,
            min_width=width,
            always_apply=True,
            border_mode=border_mode
        ),
        A.RandomCrop(
            height=height,
            width=width,
            always_apply=True
        ),
        A.GaussNoise(p=gauss_noise_p),
        A.Perspective(p=perspective_p),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1)
            ],
            p=one_of_p
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=blur_limit, p=1),
                A.MotionBlur(blur_limit=blur_limit, p=1)
            ],
            p=one_of_p
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=one_of_p
        ),
        A.Resize(
            height=height,
            width=width
        ),
        A.Normalize(
            mean=mean,
            std=std,
            always_apply=True
        )
    ]
    
    return A.Compose(augs)

def val_transform(
    height: int = 320,
    width: int = 320,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    ) -> A.Compose:

    augs = [
        A.PadIfNeeded(
            min_height=height, 
            min_width=width
        ),
        A.Resize(
            height=height,
            width=width
        ),
        A.Normalize(
            mean=mean,
            std=std,
            always_apply=True
        ),
    ]
    return A.Compose(augs)

def transform(
    train=True,
    hflip_p: float = .5,
    scale_limit: float = .5,
    rotate_limit: float = 0,
    shift_limit: float = .1,
    shif_scale_rotate_p: float = 1,
    border_mode: float = 0,
    height: int = 320,
    width: int = 320, 
    gauss_noise_p: float = .5,
    perspective_p: float = .5,
    one_of_p: float = .9,
    blur_limit: float = 3,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    ) -> A.Compose:
    """returns set of data transformations with Albumentation

    Args:
        train (bool, optional): if True train transform is returned, else val. Defaults to True.

    Returns:
        A.Compose: set of augmentations
    """
    
    if train:
        return train_transforms(
            hflip_p=hflip_p,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            shift_limit=shift_limit,
            shif_scale_rotate_p=shif_scale_rotate_p,
            border_mode=border_mode,
            height=height,
            width=width, 
            gauss_noise_p=gauss_noise_p,
            perspective_p=perspective_p,
            one_of_p=one_of_p,
            blur_limit=blur_limit,
            mean=mean,
            std=std
        )
    else:
        return val_transform(
            height=height,
            width=width,
            mean=mean,
            std=std
        )
    
def to_tensor(x: np.array, **kwargs) -> np.array:
    """transforms np array into tensor shate (c, h, w)

    Args:
        x (np.array): input np array

    Returns:
        np.array: np array in tensor format (c, h, w)
    """
    return x.transpose(2, 0, 1).astype('float32')
