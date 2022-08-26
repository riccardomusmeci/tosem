
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.dataset.road_dataset import RoadDataset
from typing import Callable, Dict, List, Optional, Union

class RoadDataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        data_dir: str, 
        classes: List[str],
        batch_size: int, 
        train_transform: Callable, 
        val_transform: Callable,
        shuffle: bool = True,
        num_workers: int = 1,
        pin_memory: bool = False,
        drop_last: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.classes = classes
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last            
        
    def prepare_data(self) -> None:  # type: ignore
        pass
    
    def setup(self, stage: Optional[str] = None) -> None:
        """loads the data

        Args:
            stage (Optional[str], optional): pipeline stage (fit, validate, test, predict). Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = RoadDataset(
                data_dir=self.data_dir,
                classes=self.classes,
                train=True,
                transform=self.train_transform,
            )
            
            self.val_dataset = RoadDataset(
                data_dir=self.data_dir,
                classes=self.classes,
                train=False,
                transform=self.val_transform,
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = RoadDataset(
                data_dir=self.data_dir,
                classes=self.classes,
                train=False,
                transform=self.val_transform
            )
            
    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )
        
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
         
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )        

