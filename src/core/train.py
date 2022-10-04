import os
import argparse
from shutil import copy
from src.utils import now
from src.io import load_config
import pytorch_lightning as pl
from src.loss import Criterion
from src.trainer import Callbacks
from src.optimizer import Optimizer
from src.transform import Transform
from src.scheduler import LRScheduler
from src.datamodule import RoadDataModule
from src.model import create_model, SegmentationModule

def train(args: argparse.Namespace):

    pl.seed_everything(seed=args.seed, workers=True)
    config = load_config(path=args.config)
    output_dir = os.path.join(args.output_dir, now())
    
    # Copying config
    os.makedirs(output_dir)
    copy(args.config, os.path.join(output_dir, "config.yml"))
        
    # data module
    datamodule = RoadDataModule(
        data_dir=args.data_dir,
        train_transform=Transform(train=True, **config["transform"]),
        val_transform=Transform(train=False, **config["transform"]),
        **config["datamodule"],
    )
    
    # creating segmentation model + loss + optimizer + lr_scheduler
    seg_model = create_model(**config["model"])
    loss = Criterion(**config["loss"])
    optimizer = Optimizer(params=seg_model.parameters(), **config["optimizer"])
    lr_scheduler = LRScheduler(optimizer=optimizer, **config["scheduler"])
    
    # segmentation pl.LightningModule
    # TODO: verifica la dimensione della maschera
    model = SegmentationModule(
        model=seg_model,
        num_classes=config["model"]["num_classes"], 
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler       
    )
    
    # lightning callbacks
    callbacks = Callbacks(
        output_dir=output_dir,
        **config["callbacks"]
    )
     
    # trainer
    trainer = pl.Trainer(
        logger=False,
        callbacks=callbacks,
        **config["trainer"]
    )
    
    # fit
    print(f"Launching training..")
    trainer.fit(model=model, datamodule=datamodule)
