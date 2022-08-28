import os
import argparse
from src.utils.time import now
import pytorch_lightning as pl
from src.io.io import load_config
from src.transform import Transform
from src.loss.loss import criterion
from src.datamodule import RoadDataModule
from src.optimizer.optimizer import get_optimizer
from src.scheduler.scheduler import get_scheduler
from src.model import create_model, SegmentationModule
from src.utils.callbacks import get_callbacks, get_logger


def train(args: argparse.Namespace):

    pl.seed_everything(seed=args.seed, workers=True)
    config = load_config(path=args.config)
    output_dir = os.path.join(args.output_dir, now())
    print(f"Output dir is {output_dir}")
    
    # data module
    datamodule = RoadDataModule(
        data_dir=args.data_dir,
        train_transform=Transform(train=True, **config["transform"]),
        val_transform=Transform(train=False, **config["transform"]),
        **config["datamodule"],
    )
    
    # creating segmentation model + loss + optimizer + lr_scheduler
    seg_model = create_model(**config["model"])
    loss = criterion(**config["loss"])
    optimizer = get_optimizer(params=seg_model.parameters(), **config["optimizer"])
    lr_scheduler = get_scheduler(optimizer=optimizer, **config["scheduler"])
    
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
    callbacks = get_callbacks(output_dir=output_dir)
    #logger = get_logger(output_dir=output_dir)
     
    # trainer
    trainer = pl.Trainer(
        logger=False,
        callbacks=callbacks,
        **config["trainer"]
    )
    
    # fit
    print(f"Launching training..")
    trainer.fit(model=model, datamodule=datamodule)
