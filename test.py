import torch
from src.metrics.jaccard_index import JaccardIndex as IoU
from torchmetrics import JaccardIndex

preds = torch.zeros((2, 1, 10, 10))
preds[:, :, 1:5, 2:8] = 1
target = torch.zeros((2, 1, 10, 10)).long()
target[:, :, 2:8, 2:8] = 1

my_iou = IoU(
    num_classes=1,
    threshold=0.5
)

pl_iou = JaccardIndex(
    num_classes=2,
    ignore_index=0
)

print(f"My IoU {my_iou(preds, target)}")
print(f"PL IoU {pl_iou(preds, target)}")
