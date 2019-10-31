import torch


def select_device(device: str) -> str:
    """
    Chooses the torch device to run in.
    :return: the passed-as-parameter device if any, otherwise
    cuda if available. Last option is cpu.
    """
    if device is not None:
        return device
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"
