import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from ..io import read_rgb, save_image


def prepare_mask(mask: np.array, width: int, height: int, threshold: Optional[float] = 0.5) -> np.array:  # type: ignore
    """Prepare mask to save.

    Args:
        mask (np.array): mask to save
        width (int): resize width
        height (int): resize height
        threshold (float, optional): mask threshold. Defaults to .5.

    Returns:
       np.array: PIL Image
    """
    mask = mask.squeeze().astype("float32")  # type: ignore
    mask = cv2.resize(mask, (width, height))
    if threshold is not None:
        mask = (mask > threshold).astype("uint8") * 255  # type: ignore
    else:
        mask = mask.astype("uint8")  # type: ignore
    return mask


def apply_mask_to_image(
    image: np.array,  # type: ignore
    mask: np.array,  # type: ignore
    alpha: float = 0.7,
    exclude_classes: Optional[List[int]] = None,
    class_map: Optional[Dict] = None,
) -> np.array:  # type: ignore
    """Apply segmentation masks to an image.

    Args:
        image (np.array): image to be blended with the mask.
        mask (np.array): mask to be blended with the image.
        alpha (float): relative weight of the mask, 1.0 means no transparency, 0.0 means the mask is completely transparent.
            Defaults to 0.7.
        exclude_classes (Optional[List[int]], optional): list of classes to exclude when applying masks. Defaults to None.
        class_map (Optional[Dict], optional): class map with class_id as key and value set as a list of [class_name, color]. Defaults to None.

    Returns:
        np.array: image with masks applied
    """
    out = image.copy()  # type: ignore
    cmap = plt.cm.tab10  # type: ignore
    for cat_id in np.unique(mask)[1:]:
        if exclude_classes is not None and cat_id in exclude_classes:
            continue
        bool_mask = mask == cat_id
        if class_map is not None:
            color = class_map[cat_id][1]
        else:
            color = np.array(cmap(cat_id - 1)[:3]) * 255
        cat_mask = (np.expand_dims(mask, axis=2) * color)[bool_mask].astype(np.uint8)
        out[bool_mask] = cv2.addWeighted(out[bool_mask], 1.0 - alpha, cat_mask, alpha, 0.0)

    return out


def save_legend(
    class_map: Dict,
    output_path: Union[Path, str],
    exclude_classes: Optional[List[int]] = None,
) -> None:
    """Save legend image.

    Args:
        class_map (Dict): class map with class_id as key and value set as a list of [class_name, color]
        output_path (Union[Path, str]): output path
        exclude_classes (Optional[List[int]], optional): list of classes to exclude when applying masks. Defaults to None.
    """
    legend = np.zeros(((len(class_map) * 25) + 25, 300, 3), dtype="uint8")
    for class_id, (class_name, color) in class_map.items():
        if exclude_classes is not None and class_id in exclude_classes:
            continue
        legend = cv2.putText(
            img=legend,
            text=class_name,
            org=(5, (class_id * 25) + 17),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(200, 200, 200),
            thickness=2,
        )
        cv2.rectangle(legend, (100, (class_id * 25)), (300, (class_id * 25) + 25), tuple(color), -1)

    save_image(image=legend, output_path=output_path)


def save_masks(
    file_paths: List[str],
    masks: np.array,  # type: ignore
    num_classes: int,
    output_dir: Union[str, Path],
    mask_threshold: Optional[float] = 0.5,
    apply_mask: bool = True,
    engine: str = "pil",
    alpha_mask: float = 0.7,
    exclude_classes: Optional[List[int]] = None,
    class_map: Optional[Dict] = None,
) -> None:
    """Save masks to output directory and apply mask to image if apply_mask is
    True, saving also images with applied masks.

    Args:
        file_paths (List[str]): list of paths to images
        masks (np.array): array of masks
        num_classes (int): number of classes
        output_dir (Union[str, Path]): path to output directory
        mask_threshold (Optional[float], optional): mask threshold. Defaults to .5.
        apply_mask (bool, optional): if True, apply mask to image. Defaults to True.
        engine (str, optional): image loading engine (pil/cv2). Defaults to "pil".
        alpha_mask (float, optional): relative weight of the mask, 1.0 means no transparency, 0.0 means the mask is completely transparent. Defaults to 0.7.
        exclude_classes (Optional[List[int]], optional): list of classes to exclude when applying masks. Defaults to None.
        class_map (Optional[Dict], optional): class map with class_id as key and value set as a list of [class_name, color]. Defaults to None.
    """
    for file_path, mask in zip(file_paths, masks):  # type: ignore
        image = read_rgb(file_path, engine=engine)
        height, width = image.shape[:2]  # type: ignore
        pred_mask = prepare_mask(
            mask=mask,
            width=width,
            height=height,
            threshold=mask_threshold if num_classes <= 2 else None,
        )
        save_image(image=pred_mask, output_path=os.path.join(output_dir, "masks", os.path.basename(file_path)))
        if apply_mask:
            image_with_mask = apply_mask_to_image(
                image=image,
                mask=pred_mask,
                alpha=alpha_mask,
                exclude_classes=exclude_classes,
                class_map=class_map,
            )
            save_image(
                image=image_with_mask,
                output_path=os.path.join(output_dir, "applied_masks", os.path.basename(file_path)),
            )
