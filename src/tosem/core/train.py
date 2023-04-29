import argparse
import os
import warnings
from shutil import copy

import pytorch_lightning as pl

from tosem import create_model
from tosem.io import load_config
from tosem.pl import Callbacks, SegmentationDataModule, SegmentationModelModule
from tosem.transform import Transform
from tosem.utils import now

warnings.filterwarnings("ignore", category=UserWarning)
from pytorch_toolbelt.losses import JaccardLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def easy_train(args: argparse.Namespace):

    pl.seed_everything(seed=args.seed, workers=True)
    config = load_config(config_path=args.config)
    output_dir = os.path.join(args.output_dir, now())

    # Copying config
    os.makedirs(output_dir)
    copy(args.config, os.path.join(output_dir, "config.yml"))

    # data module
    pl_datamodule = SegmentationDataModule(
        data_dir=args.data_dir,
        train_transform=Transform(train=True, **config["transform"]),
        val_transform=Transform(train=False, **config["transform"]),
        **config["dataset"],
        **config["datamodule"],
    )

    # creating segmentation model + loss + optimizer + lr_scheduler
    model = create_model(**config["model"])
    loss = JaccardLoss(**config["loss"])
    optimizer = SGD(model.parameters(), **config["optimizer"])
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, **config["lr_scheduler"])

    # segmentation pl.LightningModule
    pl_model = SegmentationModelModule(
        model=model,
        num_classes=config["model"]["num_classes"],
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ignore_index=0,
        beta=0.5,
    )

    # lightning callbacks
    callbacks = Callbacks(output_dir=output_dir, **config["callbacks"])

    # trainer
    trainer = pl.Trainer(logger=False, callbacks=callbacks, **config["trainer"])

    # fit
    print("Launching training..")
    trainer.fit(model=pl_model, datamodule=pl_datamodule)
