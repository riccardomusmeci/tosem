import os
import argparse
from shutil import copytree
import pytorch_lightning as pl
from src.utils.time import now
from src.utils.loss import loss_fn
from src.io.config import load_config
from src.utils.logger import get_logger
from src.utils.optimizer import get_optimizer
from src.utils.scheduler import get_scheduler
from src.transform.transform import transform
from src.utils.callbacks import get_callbacks
from src.model.utils import segmentation_model
from src.dataset.road_data_module import RoadDataModule
from src.model.segmentation_module import RoadSegmentationModule

def train(args: argparse.Namespace):
    
    ### Setting reproducibility seed
    pl.seed_everything(seed=args.seed, workers=True)
    
    ### Detting output dir
    output_dir = os.path.join(args.output_dir, now())
    print(f"> Output from training will be saved at {output_dir}")
    os.makedirs(output_dir)
    
    ### Loading configs
    transform_config, model_config, train_config, dataset_config = load_config(args.config)
    copytree(args.config, os.path.join(output_dir, "config"))
    
    ### Creating dataset
    data_module = RoadDataModule(
        data_dir=args.data_dir,
        train_transform=transform(train=True, **transform_config["train"]),
        val_transform=transform(train=False, **transform_config["val"]),
        **dataset_config
    )
    
    ### Creating model
    _model = segmentation_model(
        model=model_config["model"],
        backbone=model_config["backbone"],
        num_classes=len(data_module.classes),
        in_channels=3,
        weights=model_config["weights"],
    )

    ### Creating loss, optimizer and scheduler ###
    _loss = loss_fn(**train_config["loss"])
    print(f"> Loss - {train_config['loss']['fn']}")
    _optimizer = get_optimizer(
        params=_model.parameters(),
        **train_config["optimizer"]
    )
    print(f"> Optimizer - {train_config['optimizer']['algo']}")
    _scheduler = get_scheduler(
        optimizer=_optimizer,
        **train_config["scheduler"]
    )
    print(f"> Scheduler - {train_config['scheduler']['algo']}")
    
    ### pl Module ###
    model = RoadSegmentationModule(
        model=_model,
        num_classes=len(data_module.classes),
        loss=_loss,
        optimizer=_optimizer,
        lr_scheduler=_scheduler
    )
    
    ### Callbacks and Logger for Trainer ###
    callbacks = get_callbacks(
        output_dir=os.path.join(output_dir, "checkpoints"),
        **train_config["callbacks"]
    )
    
    logger = get_logger(
        output_dir=os.path.join(output_dir, "tensorboard"),
        **train_config["logger"]
    )
    
    ### Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=output_dir,
        **train_config["trainer"]
    )
    
    print(f"\n**** Starting Training for classes: {data_module.classes} ****\n")
    trainer.fit(model, datamodule=data_module)
    
    