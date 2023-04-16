import cv2
import matplotlib.pyplot as plt
import numpy as np


def compare_gt_pred(image: np.array, gt_mask: np.array, mask: np.array, rot_90: bool = False):
    """Plot ground truth and predicted mask on original image in one row.

    Args:
        image (np.array): original image
        gt_mask (np.array): ground truth image (can be None)
        mask (np.array): predicted mask
        rot_90 (bool, optional): if images to retate 90 degrees. Defaults to False.
    """

    plt.figure(figsize=(32, 9))
    plt.subplot(1, 3, 1)
    plt.title("Ground Truth Image + Mask")
    color = np.array([0, 0, 255], dtype=np.uint8)
    if gt_mask is not None:
        masked_img = np.where(gt_mask[..., None], color, image)
        gt_image = cv2.addWeighted(image, 0.5, masked_img, 0.5, 0)
    else:
        gt_image = image
    if rot_90:
        gt_image = cv2.rotate(gt_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    plt.imshow(gt_image)

    plt.subplot(1, 3, 2)
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # masked_img = np.where(mask[...,None], color, img)
    color = np.array([0, 0, 255], dtype=np.uint8)
    masked_img = np.where(mask[..., None], color, image)
    pred_image = cv2.addWeighted(image, 0.5, masked_img, 0.5, 0)
    if rot_90:
        pred_image = cv2.rotate(pred_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    plt.imshow(pred_image)
    plt.show()
