import torch


def select_device(device):
    if device is not None:
        return device
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"
