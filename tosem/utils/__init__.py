from .device import get_device
from .plot import compare_gt_pred
from .time import now

__all__ = ["my_device", "now", "compare_gt_pred"]

my_device = get_device()
