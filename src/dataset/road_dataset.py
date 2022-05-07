import os
from typing import Tuple
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class RoadDataset(Dataset):
    
    CLASSES = ["crack", "lane", "pothole"]
    
    def __init__(self, data_dir, train=True, transform=None, preprocessing=None):
        
        data_dir = os.path.join(data_dir, "train" if train else "val")
        self.images = [os.path.join(data_dir, img, img) for img in os.listdir(data_dir) if img != ".DS_Store"]    
        # convert str names to class values on masks        
        self.transform = transform
        self.preprocessing = preprocessing
        
    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # getting image
        image_name = self.images[i]
        image = cv2.imread(f"{self.images[i]}_RAW.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # creating masks
        masks = [cv2.imread(f"{self.images[i]}_{c.upper()}.png") for c in self.CLASSES]
        for i, mask in enumerate(masks):
            mask = mask == 255
            masks[i] = mask[..., 0]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask = torch.from_numpy(mask.transpose(2, 0, 1))
        
        return image, mask
        
    def __len__(self):
        return len(self.images)
        