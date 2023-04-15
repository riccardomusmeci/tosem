from typing import Tuple, Union

import albumentations as A
import numpy as np
import PIL
from PIL import Image


class Transform:
    def __init__(
        self,
        train: bool,
        input_size: Union[int, list, tuple],
        interpolation: int = 3,
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225],
        hflip_p: float = 0.5,
        scale_limit: float = 0.5,
        rotate_limit: float = 0,
        shift_limit: float = 0.1,
        shift_scale_rotate_p: float = 1,
        border_mode: float = 1,
        random_crop_p: float = 0.5,
        gauss_noise_p: float = 0.5,
        perspective_p: float = 0.5,
        one_of_p: float = 0.9,
        blur_limit: float = 3,
    ) -> None:
        """Saliency based transformation

        Args:
            train (bool): train/val mode.
            input_size (Union[int, list, tuple]): image input size.
            interpolation (int, optional): resize interpolation. Defaults to 3.
            mean (list, optional): normalization mean. Defaults to [0.485, 0.456, 0.406].
            std (list, optional): normalization std. Defaults to [0.229, 0.224, 0.225].
            random_crop_p (float, optional): random crop probability. Defaults to 0.1.
            h_flip_p (float, optional): horizontal flip probability. Defaults to 0.1.
            v_flip_p (float, optional): vertical flip probability. Defaults to 0.1.
        """

        if isinstance(input_size, tuple) or isinstance(input_size, list):
            height = input_size[0]
            width = input_size[1]
        else:
            height = input_size
            width = input_size

        if train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=hflip_p),
                    A.ShiftScaleRotate(
                        scale_limit=scale_limit,
                        rotate_limit=rotate_limit,
                        shift_limit=shift_limit,
                        p=shift_scale_rotate_p,
                        border_mode=border_mode,
                    ),
                    A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=border_mode),
                    A.RandomCrop(
                        height=height,
                        width=width,
                        always_apply=False,
                        p=random_crop_p,
                    ),
                    A.GaussNoise(p=gauss_noise_p),
                    A.Perspective(p=perspective_p),
                    A.OneOf([A.CLAHE(p=1), A.RandomBrightnessContrast(p=1), A.RandomGamma(p=1)], p=one_of_p),
                    A.OneOf(
                        [A.Sharpen(p=1), A.Blur(blur_limit=blur_limit, p=1), A.MotionBlur(blur_limit=blur_limit, p=1)],
                        p=one_of_p,
                    ),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(p=1),
                            A.HueSaturationValue(p=1),
                        ],
                        p=one_of_p,
                    ),
                    A.Resize(height=height, width=width, always_apply=True),
                    A.Normalize(mean=mean, std=std, always_apply=True),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
                    A.Resize(height=height, width=width),
                    A.Normalize(mean=mean, std=std, always_apply=True),
                ]
            )

    def __call__(
        self, image: Union[np.array, PIL.Image.Image], mask: Union[np.array, PIL.Image.Image]
    ) -> Tuple[np.array, np.array]:
        """Apply augmentations

        Args:
            img (Union[np.array, PIL.Image.Image]): input image

        Returns:
            Tuple[np.array, np.array, np.array]: vanilla img (resize + normalize), view 1, view 2
        """

        if isinstance(image, Image.Image):
            image = np.array(image)

        if isinstance(mask, Image.Image):
            image = np.array(image)

        if mask is not None:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        else:
            image = self.transform(image=image)["image"]
            mask = None

        return image, mask
