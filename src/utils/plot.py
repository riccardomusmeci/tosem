import cv2
import numpy as np
import matplotlib.pyplot as plt

# def visualize(image: np.array, masks: np.array, classes: list, colors: list):
#     """plots images in one row
#     """
    
#     plt.figure(figsize=(32, 9))
#     plt.subplot(1, 5, 1)
#     plt.title(f"Original Image with augmentations")
#     plt.imshow(image)
    
#     for i, c in enumerate(classes):
#         mask = np.expand_dims(masks[..., i], axis=2)
#         masked_img = np.where(mask, colors[i], image)
#         # use `addWeighted` to blend the two images
#         # the object will be tinted toward `color`
#         image = cv2.addWeighted(image, 0.5, masked_img, 0.5, 0)
#         plt.subplot(1, 5, i+3)
#         plt.title(f"Mask {c}")
#         plt.imshow(mask)
    
#     plt.subplot(1, 5, 2)
#     plt.title(f"Image with masks")
#     plt.imshow(image)
#     plt.show()
    
def visualize(
    image: np.array, 
    gt_mask: np.array, 
    mask: np.array, 
    category: list, 
):
    """plots images in one row
    """
    
    plt.figure(figsize=(32, 9))
    plt.subplot(1, 3, 1)
    plt.title(f"Grount Truth Image + Mask")
    color = np.array([0, 0, 255], dtype=np.uint8)
    if gt_mask is not None:
        masked_img = np.where(gt_mask[..., None], color, image)
        gt_image = cv2.addWeighted(image, 0.5, masked_img, 0.5, 0)
    else:
        gt_image=image
    
    # gt_image = cv2.rotate(gt_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    plt.imshow(gt_image)
    
    plt.subplot(1, 3, 2)
    plt.title(f"Pred Mask for {category}")
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # masked_img = np.where(mask[...,None], color, img)
    color = np.array([0, 0, 255], dtype=np.uint8)
    masked_img = np.where(mask[..., None], color, image)
    pred_image = cv2.addWeighted(image, 0.5, masked_img, 0.5, 0)
    # pred_image = cv2.rotate(pred_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    plt.imshow(pred_image)
    plt.show()