import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset import InferenceDataset
from ..io import TosemConfiguration
from ..model import create_model
from ..plot import save_legend, save_masks
from ..transform import Transform
from ..utils import get_device


def predict(
    ckpt_path: Union[str, Path],
    config_path: Union[str, Path],
    images_dir: Union[str, Path],
    output_dir: Union[str, Path],
    mask_threshold: float = 0.5,
    apply_mask: bool = True,
    alpha_mask: float = 0.7,
    exclude_classes: Optional[List[int]] = None,
    class_map: Optional[Dict] = None,
) -> None:
    """Predict from segmentation model.

    Args:
        ckpt_path (Union[str, Path]): path to checkpoint
        config_path (Union[str, Path]): path to configuration file
        images_dir (Union[str, Path]): path to images directory
        output_dir (Union[str, Path]): path to output directory
        mask_threshold (float, optional): mask threshold. Defaults to .5.
        apply_mask (bool, optional): apply mask to image. Defaults to True.
        alpha_mask (float, optional): relative weight of the mask, 1.0 means no transparency, 0.0 means the mask is completely transparent. Defaults to 0.7.
        exclude_classes (Optional[List[int]], optional): list of classes to exclude when applying masks. Defaults to None.
        class_map (Optional[Dict], optional): class map with class_id as key and value set as a list of [class_name, color]. Defaults to None.
    """

    config = TosemConfiguration.load(config_path=config_path)

    dataset = InferenceDataset(
        data_dir=images_dir,
        transform=Transform(train=False, **config["transform"]),
        engine=config["datamodule"]["engine"],
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config["datamodule"]["batch_size"],
        num_workers=config["datamodule"]["num_workers"],
        shuffle=False,
        drop_last=False,
    )

    device = get_device()
    model = create_model(ckpt_path=ckpt_path, **config["model"])
    model.to(device)

    model.eval()

    if class_map is not None:
        save_legend(
            class_map=class_map,
            output_path=os.path.join(output_dir, "legend.png"),
            exclude_classes=exclude_classes,
        )
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            images, file_paths = batch
            images = images.to(device)
            logits = model(images)
            if config["model"]["num_classes"] <= 2:
                masks = torch.sigmoid(logits)
            else:
                masks = torch.softmax(logits, dim=1)
                masks = torch.argmax(masks, dim=1, keepdim=True)

            masks = masks.cpu().numpy()
            save_masks(
                file_paths=file_paths,
                masks=masks,
                num_classes=config["model"]["num_classes"],
                mask_threshold=mask_threshold if config["model"]["num_classes"] <= 2 else None,
                apply_mask=apply_mask,
                output_dir=output_dir,
                engine=config["datamodule"]["engine"],
                alpha_mask=alpha_mask,
                exclude_classes=exclude_classes,
                class_map=class_map,
            )
