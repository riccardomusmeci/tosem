import os

import torch
from torch.nn.functional import sigmoid

from tests.path import TRAIN_DIR
from tosem import create_model
from tosem.dataset import InferenceDataset
from tosem.loss import create_criterion
from tosem.transform import Transform

torch.manual_seed(42)


def test_model() -> None:
    model = create_model(model_name="unet", encoder_name="resnet18", num_classes=1)
    inference_dataset = InferenceDataset(
        data_dir=os.path.join(TRAIN_DIR, "images"), transform=Transform(train=False, input_size=224)
    )
    x, _ = inference_dataset[0]
    x = x.unsqueeze(dim=0)
    out = sigmoid(model(x).squeeze(dim=0))
    assert out.shape == (1, 224, 224)
    assert out.max() <= 1
    assert out.min() >= 0


def test_loss() -> None:
    loss = create_criterion("jaccard", mode="multiclass")
    model = create_model(model_name="unet", encoder_name="resnet18", num_classes=5)
    x = torch.rand((1, 3, 224, 224))
    out = model(x)
    target = torch.randint(0, 5, (1, 1, 224, 224))
    loss_value = loss(out, target)
    assert loss_value >= 0
