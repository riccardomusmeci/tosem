# **tosem**
PyTorch Semantic Segmentation library with support to PyTorch Lightning and easy access to experiment with your own dataset.

## **How to install 🔨**
```
git clone https://github.com/riccardomusmeci/tosem
cd tosem
pip install .
```


## **Concepts 💡**
tosem tries to avoid writing again, again, and again (and again) the same code to train, test and make predictions with a semantic segmentation model.

tosem works in three different ways:
* fully automated with configuration files 🚀
* semi-automated with full support to PyTorch Lightning ⚡️
* I-want-to-write-my-own-code-but-also-using-tosem 🧑‍💻

### **TosemConfiguration 📄**
With TosemConfiguration file you don't need to write any code for training an inference.

A configuration file is like the on in config/config.yaml.

## **Train**

### **Dataset Structure**
tosem dataset must have the following structure:
```
dataset
      |__train
      |       |__images
      |       |        |__img_1.jpg
      |       |        |__img_2.jpg
      |       |        |__ ...
      |       |___masks
      |                |__img_1.png
      |                |__img_2.png
      |____val
              |__images
              |        |__img_1.jpg
              |        |__img_2.jpg
              |        |__ ...
              |___masks
                       |__img_1.png
                       |__img_2.png

```

In `binary` mode, each mask has shape (W, H).

In `multiclass` mode, each mask has shape (W, H, 3) and it must be specified the mask channel. For instance, if the mask values are in the second channel, then `mask_channel=1`.

### **Fully Automated 🚀**
Once configuration experiment file is ready, just use tosem like this:

```python
from tosem.core import train

train(
    config_path="PATH/TO/CONFIG.YAML",
    train_data_dir="PATH/TO/TRAIN/DATA/DIR",
    val_data_dir="PATH/TO/VAL/DATA/DIR",
    output_dir="PATH/TO/OUTPUT/DIR",
    resume_from="PATH/TO/CKPT/TO/RESUME/FROM", # this is when you want to start retraining from a Lightning ckpt
)
```

### **Semi-Automated ⚡️**
tosem delivers some pre-built modules based on PyTorch-Lightning to speed up experiments.

```python
from tosem import create_model
from tosem.transform import Transform
from tosem.loss import create_criterion
from tosem.optimizer import create_optimizer
from tosem.lr_scheduler import create_lr_scheduler
from tosem.pl import create_callbacks
from pytorch_lightning import Trainer
from tosem.pl import SegmentationDataModule, SegmentationModelModule

# Setting up datamodule, model, callbacks, logger, and trainer
datamodule = SegmentationDataModule(
    train_data_dir=...,
    val_data_dir=...,
    train_transform=Transform(train=True, ...),
    val_transform=Transform(train=False, ...,
    mode="binary" # multiclass
    ...,
)
model = create_model("unet", encoder_name="resnet18", weights="ssl")
criterion = create_criterion("jaccard", ...)
optimizer = create_optimizer(params=model.parameters(), optimizer="sgd", lr=.001, ...)
lr_scheduler = create_lr_scheduler(optimizer=optimizer, ...)
pl_model = SegmentationModelModule(
    model=model,
    num_classes=1,
    loss=criterion,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    mode="binary",
)
callbacks = create_callbacks(output_dir=..., ...)
trainer = Trainer(callbacks=callbacks, ...)

# Training
trainer.fit(model=pl_model, datamodule=datamodule)
```

### **I want to write my own code 🧑‍💻**
Use tosem `SegmentationDataset`, `Transform`, and `create_stuff` functions to write your own training loop.

```python
from tosem.transform import Transform
from tosem.dataset import SegmentationDataset
from tosem import create_model
from tosem.loss import create_loss
from tosem.optimizer import create_optimizer
from torch.utils.data import DataLoader
import torch

train_dataset = SegmentationDataset(
    data_dir=data_dir,
    train=True,
    transform=Transform(train=True, input_size=224),
    class_channel=0
)
train_dl = DataLoader(dataset=train_dataset, batch_size=16)

model = create_model(
    model_name="unet",
    encoder_name="timm-efficientnet-b0",
    num_classes=10,
    weights="noisy-student",
)
criterion = create_loss(loss="dice", mode="multiclass")
optimizer = create_optimizer(params=model.parameters(), optimizer="sgd", lr=0.0005)

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in train_dl:
        optimizer.zero_grad()
        x, mask = batch
        logits = model(x)
        loss = criterion(preds, mask)
        loss.backward()
        optimizer.step()
```

## **Inference 🧐**
Also in inference mode, you can pick between "fully automated", "semi-automated", "write my own code" mode.


### **Fully Automated 🚀**
Once the train is over, you'll find a *config.yaml* file merging all the setups from different sections.

```python
from tosem.core import predict

predict(
    ckpt_path="PATH/TO/OUTPUT/DIR/checkpoints/model.ckpt",
    config_path="PATH/TO/OUTPUT/DIR/config.yaml",
    images_dir="PATH/TO/IMAGES",
    output_dir="PATH/TO/OUTPUT/DIR/predictions", # you can choose your own path
    mask_threshold=0.5, # if "binary" pick your own val, None if mode=="multiclass"
    apply_mask=True, # it will apply masks to original images
    alpha_mask=0.6, # blending images and masks alpha value
    exclude_classes=[0, 2, 3], # if your want to exclude some classes from applying masks to original images
    class_map={  # color map for classes to keep
            1: ["building", (70, 70, 70)],
            4: ["pedestrian", (220, 20, 60)],
            5: ["pole", (153, 153, 153)],
        },
    )
```
