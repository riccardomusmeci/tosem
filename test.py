#from src.transform.transform import transform
from src.loss.loss import criterion
from src.dataset.road_dataset import RoadDataset
from src.model import create_model
from src.transform import Transform
from torch.utils.data import DataLoader
from src.datamodule import RoadDataModule
from torchmetrics import JaccardIndex as IoU

#from src.dataset.road_dataset import RoadDatasetSingle


train_dataset = RoadDataset(
    data_dir="/Users/riccardomusmeci/Developer/data/smart-arrotino/pothole/dataset/split",
    train=False,
    transform=Transform(
        train=True,
        input_size=224
    )
)

train_dl = DataLoader(
    dataset=train_dataset,
    batch_size=2
)

model = create_model(
    model_name="unet",
    backbone="resnet18",
    num_classes=2
)

model = model.to("mps")
loss = criterion(
    criterion="jaccard",
    mode="binary"
)

for batch in train_dl:
    
    x, mask = batch
    x = x.to("mps")
    mask = mask.to("mps")
    
    logits = model(x)
    print(mask.shape)
    print(logits.shape)
    quit()
    


# #b81e4517-126.jpg
# for i in range(len(dataset)):
#     image, mask = dataset[i]
#     print(dataset.images[i], image.shape, mask.shape)


data_dir="/Users/riccardomusmeci/Developer/data/smart-arrotino/pothole/dataset/split"


datamodule = RoadDataModule(
    data_dir=data_dir,
    batch_size=2, 
    train_transform=Transform(train=True,input_size=512),
    val_transform=Transform(train=False, input_size=512)
)
datamodule.setup()
dl = DataLoader(dataset=datamodule.train_dataset, batch_size=16)


for batch in dl:
    
    x, masks = batch[0], batch[1]
    print(x.shape, masks.shape)
    
    quit()