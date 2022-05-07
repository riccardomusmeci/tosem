import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize(image: np.array, masks: np.array, classes: list, colors: list):
    """plots images in one row
    """
    
    plt.figure(figsize=(32, 9))
    plt.subplot(1, 5, 1)
    plt.title(f"Original Image with augmentations")
    plt.imshow(image)
    
    for i, c in enumerate(classes):
        mask = np.expand_dims(masks[..., i], axis=2)
        masked_img = np.where(mask, colors[i], image)
        # use `addWeighted` to blend the two images
        # the object will be tinted toward `color`
        image = cv2.addWeighted(image, 0.5, masked_img, 0.5, 0)
        plt.subplot(1, 5, i+3)
        plt.title(f"Mask {c}")
        plt.imshow(mask)
    
    plt.subplot(1, 5, 2)
    plt.title(f"Image with masks")
    plt.imshow(image)
    plt.show()