import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from tosem import create_model
from tosem.dataset import InferenceDataset
from tosem.io import load_config, read_rgb, save_image
from tosem.transform import Transform
from tosem.utils import apply_mask, get_device, prepare_mask


def easy_inference(args: argparse.Namespace):

    device = get_device()
    config = load_config(args.config)
    model = create_model(**config["model"], ckpt_path=args.ckpt).to(device)
    model.eval()

    dataset = InferenceDataset(data_dir=args.data_dir, transform=Transform(train=False, **config["transform"]))

    output_dir = os.path.join(args.output_dir, "masks")
    print(f"> Masks will be saved at {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    if args.apply_mask:
        apply_mask_dir = os.path.join(args.output_dir, "applied_masks")
        os.makedirs(apply_mask_dir, exist_ok=True)
        print(f"> Original images with masks applied will be saved at {output_dir}")

    for index, x in tqdm(enumerate(dataset), total=len(dataset)):
        img_path = dataset.images[index]
        with torch.no_grad():
            x = x.unsqueeze(0).to(device)
            logits = model(x)
            if config["model"]["num_classes"] <= 2:
                preds = torch.sigmoid(logits)
            else:
                preds = torch.softmax(logits, dim=1)
                preds = torch.argmax(preds, dim=1, keepdim=True)
            image = read_rgb(img_path)
            height, width = image.shape[:2]
            pred_mask = prepare_mask(
                mask=preds,
                width=width,
                height=height,
                threshold=args.threshold if config["model"]["num_classes"] <= 2 else None,
            )
            save_image(image=pred_mask, output_path=os.path.join(output_dir, os.path.basename(img_path)))
            if args.apply_mask:
                image_with_mask = apply_mask(image=image, mask=pred_mask, alpha=0.7)
                save_image(image=image_with_mask, output_path=os.path.join(apply_mask_dir, os.path.basename(img_path)))
