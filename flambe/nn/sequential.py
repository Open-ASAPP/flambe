# type: ignore[override]

from typing import Union, Dict

import torch

from flambe.nn import Module


class Sequential(Module):
    """Implement a Sequential module.

    This class can be used in the same way as torch's nn.Sequential,
    with the difference that it accepts kwargs arguments.

    """
    def __init__(self, **kwargs: Dict[str, Union[Module, torch.nn.Module]]) -> None:
        """Initialize the Sequential module.

        Parameters
        ----------
        kwargs: Dict[str, Union[Module, torch.nn.Module]]
            The list of modules.

        """
        super().__init__()

        modules = []
        for name, module in kwargs.items():
            setattr(self, name, module)
            modules.append(module)

        self.seq = torch.nn.Sequential(modules)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data: torch.Tensor
            input to the model

        Returns
        -------
        output: torch.Tensor
            output of the model

        """
        return self.seq(data)
