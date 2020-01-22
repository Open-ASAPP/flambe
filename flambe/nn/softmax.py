# type: ignore[override]

from typing import Optional

import torch
from torch import nn

from flambe.nn.mlp import MLPEncoder
from flambe.nn.module import Module


class SoftmaxLayer(Module):
    """Implement an SoftmaxLayer module.

    Can be used to form a classifier out of any encoder.
    Note: by default takes the log_softmax so that it can be fed to
    the NLLLoss module. You can disable this behavior through the
    `take_log` argument.

    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 mlp_layers: int = 1,
                 mlp_dropout: float = 0.,
                 mlp_hidden_activation: Optional[nn.Module] = None,
                 take_log: bool = True) -> None:
        """Initialize the SoftmaxLayer.

        Parameters
        ----------
        input_size : int
            Input size of the decoder, usually the hidden size of
            some encoder.
        output_size : int
            The output dimension, usually the number of target labels
        mlp_layers : int
            The number of layers in the MLP
        mlp_dropout: float, optional
            Dropout to be used before each MLP layer
        mlp_hidden_activation: nn.Module, optional
            Any PyTorch activation layer, defaults to None
        take_log: bool, optional
            If ``True``, compute the LogSoftmax to be fed in NLLLoss.
            Defaults to ``False``.

        """
        super().__init__()

        softmax = nn.LogSoftmax(dim=-1) if take_log else nn.Softmax()
        self.mlp = MLPEncoder(input_size=input_size, output_size=output_size,
                              n_layers=mlp_layers, dropout=mlp_dropout,
                              hidden_activation=mlp_hidden_activation,
                              output_activation=softmax)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data: torch.Tensor
            input to the model of shape (*, input_size)

        Returns
        -------
        output: torch.Tensor
            output of the model of shape (*, output_size)

        """
        return self.mlp(data)
