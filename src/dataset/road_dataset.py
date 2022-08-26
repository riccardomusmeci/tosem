import os
import cv2
import torch
import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset
from src.io.io import read_rgb, read_gray

class RoadAnomalyDataset(Dataset):

    def __init__(self, data_dir, classes: List[str], train=True, transform=None):
        
        data_dir = os.path.join(data_dir, "train" if train else "val")
        self.images = [os.path.join(data_dir, img, img) for img in os.listdir(data_dir) if img != ".DS_Store"]
        
        if classes is None:
            print("Plase specify at least one class among crack, lane and pothole.}")
            quit()
            
        self.classes = classes
        # convert str names to class values on masks        
        self.transform = transform
        
    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # getting image
        image = read_rgb(f"{self.images[i]}_RAW.jpg")
    
        # creating masks
        masks = []
        for c in self.classes:
            if c == "background":
                continue
            mask = read_gray(f"{self.images[i]}_{c.upper()}.png")
            mask[mask==255] = 1
            masks.append(mask)    
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # casting to Torch.Tensor
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).long()
        
        return image, mask
        
    def __len__(self):
        return len(self.images)
    

class RoadDataset(Dataset):
    
    def __init__(self, data_dir, classes: List[str], train=True, transform=None):
        
        data_dir = os.path.join(data_dir, "train" if train else "val")
        self.images = [os.path.join(data_dir, img, img) for img in os.listdir(data_dir) if img != ".DS_Store"]
        
        if classes is None:
            print("Plase specify at least one class among crack, lane and pothole.}")
            quit()
            
        self.classes = classes
        # convert str names to class values on masks        
        self.transform = transform
        
    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # getting image
        image = read_rgb(f"{self.images[i]}_RAW.jpg")
    
        # creating masks
        masks = []
        for c in self.classes:
            if c == "background":
                continue
            mask = read_gray(f"{self.images[i]}_{c.upper()}.png")
            mask[mask==255] = 1
            masks.append(mask)    
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # casting to Torch.Tensor
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).long()
        
        return image, mask
        
    def __len__(self):
        return len(self.images)
    
class RoadInferenceDataset(Dataset):
    
    def __init__(self, data_dir, transform=None) -> None:
        super().__init__()
        self.images = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img != ".DS_Store"]
        self.transform = transform
        
    def __getitem__(self, index):
        
        image_path = self.images[index]
        image = read_rgb(image_path)
        if self.transform:
            image = self.transform(image=image)['image']
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
        image = torch.from_numpy(image.transpose(2, 0, 1))
        return image
    
    def __len__(self):
        return len(self.images)
        
        