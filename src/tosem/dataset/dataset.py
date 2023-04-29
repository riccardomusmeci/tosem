import os
from typing import Callable, Dict, Tuple

import torch
from torch.utils.data import Dataset

from ..io.image import read_mask, read_rgb


class SegmentationDataset(Dataset):
    """Segmentation Dataset class to train a model

    Args:
        data_dir (str): data dir
        train (bool): if True it looks for "train" folder, else for "val" folder.
        transform (Callable, optional): augmentation function. Defaults to None.
        class_map (Dict, optional): map to change some mask pixel value (e.g. from 3 to background, so 0). Defaults to None
        class_channel (int, optional): which channel the pixel masks are saved (0 -> R, 1 -> G, 2 -> B). Defaults to 0.
        verbose (bool, optional): verbose mode. Defaults to True.
    """

    EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    )

    def __init__(
        self,
        data_dir: str,
        train: bool,
        transform: Callable = None,
        class_map: Dict = None,
        class_channel: int = None,
        verbose: bool = True,
    ) -> None:

        self.train = train
        self.data_dir = data_dir
        self.verbose = verbose
        self.class_channel = class_channel

        # checking sanity of data_dir
        try:
            self._sanity_check()
        except Exception as e:
            print("[ERROR] Failed on sanity check")
            raise e

        # defining dirs
        self.data_dir = os.path.join(data_dir, "train" if train else "val")
        self.images_dir = os.path.join(self.data_dir, "images")
        self.masks_dir = os.path.join(self.data_dir, "masks")
        # TO-DO -> implement class map
        self.class_map = class_map
        # loading images and masks
        self.images, self.masks = self._load_samples()
        self.transform = transform

    def _sanity_check(self):
        """Check correct structure of data dir

        Returns:
            bool: if True structure is ok, else False
        """
        for split in ["train", "val"]:
            split_dir = os.path.join(self.data_dir, split)
            if not (os.path.exists(split_dir)):
                raise FileNotFoundError(f"Split data dir {split_dir} does not exist.")
            else:
                images_dir = os.path.join(split_dir, "images")
                if not (os.path.exists(images_dir)):
                    raise FileNotFoundError(f"Images dir {images_dir} does not exist.")
                masks_dir = os.path.join(split_dir, "masks")
                if not (os.path.exists(masks_dir)):
                    raise FileNotFoundError(f"Masks dir {masks_dir} does not exist.")
        if self.verbose:
            print(f"> {'Train' if self.train else 'Val'} dataset sanity check OK")

    def _load_samples(self) -> Tuple:
        """Load images and masks assuring data consistency

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
        """Return an item composed of image and mask

        Args:
            index (int): index of element

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: image, mask
        """
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = read_rgb(file_path=image_path)
        if self.class_channel is not None:
            mask = read_rgb(file_path=mask_path)
            mask = mask[:, :, self.class_channel]
            # if self.class_map is not None:
            #     for src_pixel, dst_pixel in self.class_map.items():
            #         mask[mask==src_pixel] = dst_pixel
        else:
            mask = read_mask(file_path=mask_path) / 255

        if self.transform:
            try:
                image, mask = self.transform(image=image, mask=mask)
            except Exception as e:
                print(f"Transform on image {image_path} not working. Reason: {e}")
                quit()
        return image, mask

    def __len__(self) -> int:
        """Return number of images in the dataset

        Returns:
            int: number of images in the dataset
        """
        return len(self.images)


class InferenceDataset(Dataset):

    EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    )

    def __init__(
        self,
        data_dir: str,
        transform: Callable = None,
    ) -> None:
        """Inference Dataset (a folder with images)

        Args:
            root_dir (str): root data dir (must be with train and val folders)
            transform (Callable, optional): set of data transformations. Defaults to None.

        Raises:
            FileNotFoundError: if something is found erroneous in the dataset
        """

        super().__init__()
        self.data_dir = data_dir
        self.images = [
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if os.path.splitext(f)[-1] in self.EXTENSIONS
        ]
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.images[index]
        img = read_rgb(img_path)

        if self.transform is not None:
            img, _ = self.transform(img)

        return img

    def __len__(self):
        return len(self.images)
