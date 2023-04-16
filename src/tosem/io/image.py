import os
from pathlib import Path

import cv2
import numpy as np


def read_rgb(file_path: Path) -> np.array:
    """Load an image from file_path as a numpy array

    Args:
        file_path (Path): path to image

    Raises:
        FileNotFoundError: if file is not found

    Returns:
        np.array: image
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path {file_path} does not exist")
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Unable to read {file_path}.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_mask(file_path: Path) -> np.array:
    """Load a segmentation mask from file_path as a numpy array

    Args:
        file_path (Path): path to image

    Raises:
        FileNotFoundError: if file is not found

    Returns:
        np.array: mask
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path {file_path} does not exist")
    img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read {file_path}.")

    return img
