import os

from tests.path import TRAIN_DIR
from tosem.dataset import InferenceDataset, SegmentationDataset
from tosem.transform import Transform


def test_train_dataset() -> None:
    dataset = SegmentationDataset(
        data_dir=TRAIN_DIR, transform=Transform(train=True, input_size=224), mode="binary", verbose=False
    )

    img, mask = dataset[0]
    assert img.shape == (3, 224, 224)
    assert mask.shape == (1, 224, 224)
    assert len(dataset) == 3


def test_inference_dataset() -> None:
    dataset = InferenceDataset(
        data_dir=os.path.join(TRAIN_DIR, "images"),
        transform=Transform(train=False, input_size=224),
        verbose=False,
    )

    img, _ = dataset[0]
    assert img.shape == (3, 224, 224)
    assert len(dataset) == 3
