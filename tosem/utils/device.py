import torch


def get_device() -> str:
    """Return device type of current machine

    Returns:
        str: device type (mps, cuda, cpu)
    """
    if torch.has_mps:
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
