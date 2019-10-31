from typing import Optional

import torch


def select_device(device: Optional[str]) -> str:
    """
    Chooses the torch device to run in.

     Parameters
       ------------
     device: Union[torch.device, str]
         A device or a string representing a device, such as 'cpu'

     Returns
     ---------
     str
         the passed-as-parameter device if any, otherwise
         cuda if available. Last option is cpu.
    """

    if device is not None:
        return device
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"
