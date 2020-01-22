# type: ignore[override]

import torch
import torch.nn as nn
from torch import Tensor

from flambe.nn.mlp import MLPEncoder
from flambe.nn.module import Module


class MixtureOfSoftmax(Module):
    """Implement the MixtureOfSoftmax output layer.

    Attributes
    ----------
    pi: FullyConnected
        softmax layer over the different softmax
    layers: [FullyConnected]
        list of the k softmax layers

    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 k: int = 1,
                 take_log: bool = True) -> None:
        """Initialize the MOS layer.

        Parameters
        ----------
        input_size: int
            input dimension
        output_size: int
            output dimension
        k: int (Default: 1)
            number of softmax in the mixture

        """
        super().__init__()

        self.pi_w = MLPEncoder(input_size, k)
        self.softmax = nn.Softmax()

        self.layers = [MLPEncoder(input_size, output_size) for _ in range(k)]
        self.tanh = nn.Tanh()

        self.activation = nn.LogSoftmax() if take_log else nn.Softmax()

    def forward(self, data: Tensor) -> Tensor:
        """Implement mixture of softmax for language modeling.

        Parameters
        ----------
        data: torch.Tensor
            seq_len x batch_size x hidden_size

        Return
        -------
        out: Variable
            output matrix of shape seq_len x batch_size x out_size

        """
        w = self.softmax(self.pi_w(data))
        # Compute k softmax, and combine using above weights
        out = [w[:, :, i] * self.tanh(W(data)) for i, W in enumerate(self.layers)]
        out = torch.cat(out, dim=0).sum(dim=0)

        return self.activation(out)
