from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from tosem.dataset import SegmentationDataset

from ..utils import get_num_workers


class SegmentationDataModule(pl.LightningDataModule):
    """Segmentation Data Module from PyTorch Lightning.

    Args:
        train_data_dir (str): path to train dataset
        val_data_dir (str): path to val dataset
        train_transform (Callable): train transform function
        val_transform (Callable): validation transform function
        mode (str): segmentation mode (binary/multiclass/multilabel).
        mask_channel (int, optional): mask channel if mode is "multiclass". Defaults to 0.
        engine (str, optional): image loading engine (pil/cv2). Defaults to "pil".
        batch_size (int, optional): DataLoader batch size. Defaults to 8.
        shuffle (bool, optional): whether to shuffle the dataset in training. Defaults to True.
        num_workers (Union[str, int], optional): number of workers for DataLoader. If full/half the number is found based num cpus. Defaults to "full".
        persistent_workers (bool, optional): persistent workers for DataLoader. Defaults to True.
        pin_memory (bool, optional): DataLoader pin memory. Defaults to False.
        drop_last (bool, optional): whether to drop the last batch in training. Defaults to False.
        verbose (bool, optional): verbose mode. Defaults to True.
    """

    def __init__(
        self,
        train_data_dir: Union[Path, str],
        val_data_dir: Union[Path, str],
        train_transform: Callable,
        val_transform: Callable,
        mode: str,
        mask_channel: int = 0,
        engine: str = "pil",
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: Union[str, int] = "half",
        persistent_workers: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.mode = mode
        self.mask_channel = mask_channel
        self.engine = engine
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = get_num_workers(num_workers) if isinstance(num_workers, str) else num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self.verbose = verbose

    def prepare_data(self) -> None:
        """Prepare data."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load the data.

        Args:
            stage (Optional[str], optional): pipeline stage (fit, validate, test, predict). Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = SegmentationDataset(
                data_dir=self.train_data_dir,
                transform=self.train_transform,
                mode=self.mode,
                mask_channel=self.mask_channel,
                engine=self.engine,
            )

            self.val_dataset = SegmentationDataset(
                data_dir=self.val_data_dir,
                transform=self.val_transform,
                mode=self.mode,
                mask_channel=self.mask_channel,
                engine=self.engine,
            )

    def train_dataloader(self) -> DataLoader:
        """Train dataloader.

        Returns:
            DataLoader: train data loader
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader.

        Returns:
            DataLoader: validation data loader
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
