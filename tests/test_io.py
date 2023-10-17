import os

import numpy as np

from tosem.io import TosemConfiguration, read_binary_mask, read_rgb

from .path import CONFIG_DIR, TRAIN_DIR


def test_merge_config() -> None:
    config = TosemConfiguration.load_experiment(config_dir=CONFIG_DIR, experiment="experiment")
    assert config["model"]["model_name"] == "unet"
    assert config["dataset"]["mode"] == "binary"


def test_read_rgb_pil() -> None:
    """Test read_rgb."""
    image = read_rgb(os.path.join(TRAIN_DIR, "images", "1.jpg"), engine="pil")
    assert image.shape == (1536, 2048, 3)
    assert image.dtype == np.uint8


def test_read_rgb_cv2() -> None:
    """Test read_rgb."""
    image = read_rgb(os.path.join(TRAIN_DIR, "images", "1.jpg"), engine="cv2")
    assert image.shape == (1536, 2048, 3)
    assert image.dtype == np.uint8


def test_read_binary_mask_pil() -> None:
    """Test read_rgb."""
    mask = read_binary_mask(os.path.join(TRAIN_DIR, "masks", "1.png"), engine="pil")
    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == (1536, 2048)
    assert mask.dtype == np.uint8


def test_read_binary_mask_cv2() -> None:
    """Test read_rgb."""
    mask = read_binary_mask(os.path.join(TRAIN_DIR, "masks", "1.png"), engine="cv2")
    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == (1536, 2048)
    assert mask.dtype == np.uint8
