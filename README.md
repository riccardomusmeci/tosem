# **tosem**
Semantic Segmentation library based on PyTorch.

----

## **Before starting**
**tosem** is built on torch
- timm
- torch
- pytorch-lightning (easy multi-GPU support)
- torchmetrics

## **Train**

### **Easy mode**
Training a semantic segmentation can be easily accessed with:
```
python train.py \
    --data-dir PATH/TO/YOUR/TRAIN/DATASET \
    --config config/config.yaml \
    --output-dir PATH/TO/YOUR/OUTPUT/DIR
    --seed 42
```
The YAML configuration file contains a lot of params to configure. Go check the default one in *config/config.yaml*.

### **PyTorch-Lightning Code Mode**
**tosem** has some pre-built modules based on PyTorch-Lightning to speed up experiments.

```python
import pytorch_lightning as pl
from tosem import create_model
from tosem.io import load_config
from tosem.pl import Callbacks, SegmentationDataModule, SegmentationModelModule
from tosem.transform import Transform
from pytorch_toolbelt.losses import JaccardLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# data module
pl_datamodule = SegmentationDataModule(
    data_dir=args.data_dir,
    train_transform=Transform(
        train=True,
        input_size=512,
        ... # check the config.yaml file to look for params ('transform' section)
    ),
    val_transform=Transform(
        train=False,
        input_size=512,
        ... # check the config.yaml file to look for params ('transform' section)
    ),
    ...
)

# creating segmentation model + loss + optimizer + lr_scheduler
model = create_model(
    model_name="unet",
    encoder_name="resnet18",
    num_classes=10,
    ckpt_path=... # you can directly load your trained model
)
loss = JaccardLoss(...)
optimizer = SGD(model.parameters(), ...)
lr_scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, ...)

# segmentation pl.LightningModule
pl_model = SegmentationModelModule(
    model=model,
    num_classes=10,
    loss=loss,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    ignore_index=0, # not considering background in IoU computation
    beta=0.5 # fbeta score beta param
)

# lightning callbacks
callbacks = Callbacks(output_dir=output_dir, ...)

# trainer
trainer = pl.Trainer(logger=False, callbacks=callbacks, ...)

# fit
print("Launching training..")
trainer.fit(model=pl_model, datamodule=pl_datamodule)
```

### **PyTorch Code Mode**
If you don't like PyTorch-Lightning you can always use the class PyTorch mode to train your model. You can always use the modules from tosem to speed-up experiments.

```python
from pytorch_toolbelt.losses import JaccardLoss
from torch.optim import SGD
from tosem.transform import Transform
from tosem.dataset import SegmentationDataset
from tosem import create_model
from pytorch_toolbelt.losses import JaccardLoss
from torch.optim import SGD
from torchmetrics import JaccardIndex as IoU
from torch.utils.data import DataLoader
from tosem.utils import get_device
import torch

device = get_device()
train_dataset = SegmentationDataset(
    data_dir=data_dir,
    train=True,
    transform=Transform(train=True, input_size=224),
    class_channel=0
)
val_dataset = SegmentationDataset(
    data_dir=data_dir,
    train=False,
    transform=Transform(train=False, input_size=224),
    class_channel=0
)

train_dl = DataLoader(dataset=train_dataset, batch_size=16)
val_dl = DataLoader(dataset=val_dataset, batch_size=16)

model = create_model(
    model_name="unet",
    encoder_name="timm-efficientnet-b0",
    num_classes=10,
    weights="noisy-student",
)

criterion = JaccardLoss(mode="multiclass" if num_classes>2 else "binary")
optimizer = SGD(model.parameters(), lr=0.0005)
iou = IoU(task="multiclass", num_classes=num_classes, ignore_index=0).to(device)

model.to(device)
for epoch in range(100):
    model.train()
    for batch in train_dl:
        optimizer.zero_grad()
        x, mask = (el.to(device) for el in batch)
        logits = model(x)
        loss = criterion(preds, mask)
        loss.backward()
        optimizer.step()
    model.eval()
    for batch in val_dl:
        x, mask = (el.to(device) for el in batch)
        logits = model(x)
        if num_classes == 2:
            preds = torch.sigmoid(logits)
        else:
            preds = torch.softmax(logits, dim=1)
        preds = torch.argmax(preds, dim=1, keepdim=True)
        iou.update(preds, mask)
    print(f"val IoU: {iou.compute()}")
    iou.reset()
```

## **Inference**
Please have a look at *notebooks/inference_visualization.ipynb* notebook to have a look at some predictions based on your model.


## **TO-DOs**
[ ] inference API
