import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from ..io import read_binary_mask, read_rgb


class SegmentationDataset(Dataset):
    """Segmentation Dataset class to train a model.

    Args:
        data_dir (Union[Path, str]): data dir
        transform (Optional[Callable], optional): augmentation function. Defaults to None.
        mode: (str): segmentation mode (binary/multi). Defaults to "binary".
        engine (str, optional): image loading engine (pil/cv2). Defaults to "pil".
        verbose (bool, optional): verbose mode. Defaults to True.
    """

    EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".gif")

    def __init__(
        self,
        data_dir: Union[Path, str],
        transform: Optional[Callable] = None,
        mode: str = "binary",
        mask_channel: int = 0,
        engine: str = "pil",
        verbose: bool = True,
    ) -> None:
        assert mode in [
            "binary",
            "multiclass",
            "multilabel",
        ], f"Param mode {mode} is not supported. Available modes: binary, multiclass, multilabel."

        self.data_dir = data_dir
        self.mode = mode
        self.mask_channel = mask_channel
        self.verbose = verbose
        self.engine = engine

        # checking sanity of data_dir
        try:
            self._sanity_check()
        except Exception as e:
            print("[ERROR] Failed on sanity check")
            raise e

        # defining dirs
        self.images_dir = os.path.join(self.data_dir, "images")
        self.masks_dir = os.path.join(self.data_dir, "masks")
        # loading images and masks
        self.images, self.masks = self._load_samples()
        self.transform = transform

    def _sanity_check(self) -> None:
        """Sanity check on data_dir.

        Raises:
            FileNotFoundError: Dataset directory does not exist.
            FileNotFoundError: Images directory does not exist.
            FileNotFoundError: Masks directory does not exist.
        """

        if not (os.path.exists(self.data_dir)):
            raise FileNotFoundError(f"Data dir {self.data_dir} does not exist.")
        else:
            images_dir = os.path.join(self.data_dir, "images")
            if not (os.path.exists(images_dir)):
                raise FileNotFoundError(f"Images dir {images_dir} does not exist.")
            masks_dir = os.path.join(self.data_dir, "masks")
            if not (os.path.exists(masks_dir)):
                raise FileNotFoundError(f"Masks dir {masks_dir} does not exist.")

    def _load_samples(self) -> Tuple:
        """Load images and masks assuring data consistency.

        Raises:
            FileNotFoundError: if no images and related masks are found

        Returns:
            Tuple: images, masks paths
        """
        # loading all images and masks
        all_images = [f for f in os.listdir(self.images_dir) if os.path.splitext(f)[-1] in self.EXTENSIONS]
        base_to_mask_map = {
            os.path.splitext(f)[0]: f for f in os.listdir(self.masks_dir) if os.path.splitext(f)[-1] in self.EXTENSIONS
        }

        # images and masks to return
        images, masks = [], []
        not_found = 0
        for image in all_images:
            base_name = os.path.splitext(image)[0]
            if base_name in base_to_mask_map.keys():
                images.append(os.path.join(self.images_dir, image))
                masks.append(os.path.join(self.masks_dir, base_to_mask_map[base_name]))
            else:
                not_found += 1
        if self.verbose:
            print(f"> Found {len(images)} images with masks")
            if not_found > 0:
                print(f"> Not found {not_found} images with masks")

        if len(images) == 0:
            raise FileNotFoundError("No images and related masks found. Check your data.")

        return images, masks

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return an item composed of image and mask.

        Args:
            index (int): index of element

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: image, mask
        """
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = read_rgb(file_path=image_path, engine=self.engine)
        if self.mode == "binary":
            mask = read_binary_mask(file_path=mask_path, engine=self.engine)
        elif self.mode == "multiclass":
            mask = read_rgb(file_path=mask_path, engine=self.engine)
            mask = mask[:, :, self.mask_channel]  # type: ignore
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented.")
        if self.transform:
            try:
                image, mask = self.transform(image=image, mask=mask)
            except Exception as e:
                print(f"Transform on image {image_path} not working. Reason: {e}")
                quit()

        image = torch.from_numpy(image.transpose(2, 0, 1))  # type: ignore
        mask = torch.from_numpy(mask).long()
        mask = mask.unsqueeze(dim=0)

        return image, mask

    def __len__(self) -> int:
        """Return number of images in the dataset.

        Returns:
            int: number of images in the dataset
        """
        return len(self.images)
