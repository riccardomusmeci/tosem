from typing import Optional, Tuple, Union

import albumentations as A
import numpy as np
import PIL
from PIL import Image


class Transform:
    """Semantic Segmentation base transformation.

    Args:
        train (bool): train/val mode.
        input_size (Union[int, list, tuple]): image input size.
        interpolation (int, optional): resize interpolation. Defaults to 3.
        mean (Tuple[float, float, float], optional): normalization mean. Defaults to (0.485, 0.456, 0.406).
        std (Tuple[float, float, float], optional): normalization std. Defaults to (0.229, 0.224, 0.225).
        pad_border_mode (int, optional): padding value. Defaults to 1.
        random_crop_prob (float, optional): random crop probability. Defaults to 0.1.
        horizontal_flip_prob (float, optional): horizontal flip probability. Defaults to 0.1.
        vertical_flip_prob (float, optional): vertical flip probability. Defaults to 0.1.
        rotate_90_prob (float, optional): rotate 90 probability. Defaults to 0.1.
    """

    def __init__(
        self,
        train: bool,
        input_size: Union[int, list, tuple],
        interpolation: int = 3,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        pad_border_mode: int = 0,
        random_crop_prob: float = 0.1,
        horizontal_flip_prob: float = 0.1,
        vertical_flip_prob: float = 0.1,
        rotate_90_prob: float = 0.1,
    ) -> None:
        if isinstance(input_size, tuple) or isinstance(input_size, list):
            height = input_size[0]
            width = input_size[1]
        else:
            height = input_size
            width = input_size

        if train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=horizontal_flip_prob),
                    A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=pad_border_mode),
                    A.RandomCrop(height=height, width=width, p=random_crop_prob),
                    A.VerticalFlip(p=vertical_flip_prob),
                    A.RandomRotate90(p=rotate_90_prob),
                    A.Resize(height=height, width=width, interpolation=interpolation, always_apply=True),
                    A.Normalize(mean=mean, std=std, always_apply=True),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    # A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=pad_border_mode),
                    A.Resize(height=height, width=width),
                    A.Normalize(mean=mean, std=std, always_apply=True),
                ]
            )

    def __call__(
        self, image: Union[np.array, PIL.Image.Image], mask: Optional[Union[np.array, PIL.Image.Image]] = None  # type: ignore
    ) -> Tuple[np.array, np.array]:  # type: ignore
        """Apply augmentations.

        Args:
            img (Union[np.array, PIL.Image.Image]): input image
            mask (Optional[Union[np.array, PIL.Image.Image]], optional): input mask. Defaults to None.

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
