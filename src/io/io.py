

import cv2
import torch
import numpy as np
from typing import Dict

def load_state_dict_from_ckpt(ckpt_path: str) -> Dict:
    """laods a state dict from ckpt file

    Args:
        ckpt_path (str): ckpt path

    Returns:
        Dict: model state dict (weights)
    """
    device = torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')
    ckpt_state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
    state_dict = {}
    for key, weights in ckpt_state_dict.items():
        l = key.replace("model.", "")
        state_dict[l] = weights
    return state_dict

def read_rgb(img_path: str) -> np.array:
    """reads an image in RGB mode

    Args:
        img_path (str): path to image

    Returns:
        np.array: rgb image
    """
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image