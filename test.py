from src.datamodule import RoadDataModule
#from src.transform.transform import transform
from src.transform import Transform
from torch.utils.data import DataLoader
#from src.dataset.road_dataset import RoadDatasetSingle


# dataset = RoadDatasetSingle(
#     data_dir="/Users/riccardomusmeci/Developer/data/smart-arrotino/pothole/dataset/split",
#     train=True,
#     transform=transform(
#         train=True,
#         input_size=512
#     )
# )

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