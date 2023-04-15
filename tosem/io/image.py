import os
from pathlib import Path

import numpy as np
from PIL import Image


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
    image = Image.open(file_path).convert("RGB")
    image = np.array(image)
    return image


def read_binary(file_path: Path) -> np.array:
    """Load a binary mask from file_path as a numpy array

    Args:
        file_path (Path): path to image

    Raises:
        FileNotFoundError: if file is not found

    Returns:
        np.array: mask
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path {file_path} does not exist")
    image = Image.open(file_path).convert("L")
    image = np.array(image, dtype=np.float32) / 255
    return image
