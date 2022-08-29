import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Callable, List, Tuple
from src.io.io import read_rgb, read_binary, read_gray

class RoadDataset(Dataset):
    
    def __init__(
        self,
        data_dir: str,
        train: bool,
        transform: Callable = None
    ) -> None:
        
        self.data_dir = os.path.join(data_dir, "train" if train else "val")
        self.images_dir = os.path.join(self.data_dir, "images")
        self.masks_dir = os.path.join(self.data_dir, "masks")
        self.images = [f for f in os.listdir(self.images_dir) if f != ".DS_Store"]
        self.transform = transform
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        
        image_name = self.images[index]
        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, image_name)
        
        image = read_rgb(file_path=image_path)
        mask = read_binary(file_path=mask_path)
        
        if self.transform:
            try:
                sample = self.transform(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            except Exception as e:
                print(f"Transform on image {image_path} not working. Reason: {e}")
                quit()

        #print(image.shape, mask.shape)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).long()
        mask = mask.unsqueeze(dim=0)
        
        return image, mask
    
    def __len__(self):
        return len(self.images)
    
    
class RoadInferenceDataset(Dataset):
    
    def __init__(
        self,
        data_dir: str,
        transform: Callable = None
    ) -> None:
        
        self.data_dir = data_dir
        self.image_paths = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f != ".DS_Store"]
        self.transform = transform
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        
        image_path = self.image_paths[index]
        
        image = read_rgb(file_path=image_path)
        
        if self.transform:
            try:
                image = self.transform(image=image)["image"]
            except Exception as e:
                print(f"Transform on image {image_path} failed. Reason: {e}")
                quit()

        #print(image.shape, mask.shape)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        
        return image
    
    def __len__(self):
        return len(self.image_paths)
    
'''
class RoadDatasetMulti(Dataset):
    
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
'''     