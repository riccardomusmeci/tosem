import os
import torch
import argparse
from shutil import copy
import pytorch_lightning as pl
from src.utils.time import now
from src.loss.loss import loss_fn
from src.io.io import load_config
from src.utils.logger import get_logger
from src.transform.transform import transform
from src.utils.callbacks import get_callbacks
from src.model.utils import segmentation_model
from src.optimizer.optimizer import get_optimizer
from src.scheduler.scheduler import get_scheduler

from src.dataset.road_data_module import RoadDataModule
from src.model.segmentation_module import RoadSegmentationModule

def train(args: argparse.Namespace):
    
    ### Setting reproducibility seed
    pl.seed_everything(seed=args.seed, workers=True)
    
    ### Loading configs
    config = load_config(args.config)
        
    ### Detting output dir
    output_dir = os.path.join(args.output_dir, config['project_name'], now())
    print(f"> Output from training will be saved at {output_dir}")
    os.makedirs(output_dir)

    
    copy(args.config, output_dir)
    # copytree(args.config, os.path.join(output_dir, "config"))
    
    ### Creating data_module
    data_module = RoadDataModule(
        data_dir=args.data_dir,
        train_transform=transform(train=True, **config["transform"]),
        val_transform=transform(train=False,  **config["transform"]),
        **config["dataset"]
    )
    
    ### Creating model
    model = segmentation_model(
        num_classes=len(config["dataset"]["classes"]),
        **config["model"]
    )
    
    ### Creating loss, optimizer and scheduler ###
    loss = loss_fn(**config["loss"])
    optimizer = get_optimizer(params=model.parameters(), **config["optimizer"])
    scheduler = get_scheduler(optimizer=optimizer, **config["scheduler"])
    
    
    ### pl Module ###
    seg_module = RoadSegmentationModule(
        model=model,
        num_classes=len(data_module.classes),
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=scheduler
    )
    
    ### Callbacks and Logger for Trainer ###
    callbacks = get_callbacks(output_dir=output_dir)
    logger = get_logger(output_dir=output_dir)
    
    ### Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=output_dir,
        **config["trainer"]
    )
    
    print(f"\n**** Starting Training for classes: {data_module.classes} ****\n")
    trainer.fit(seg_module, datamodule=data_module)
    
    