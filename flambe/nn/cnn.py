# type: ignore[override]

from typing import Optional, Tuple, List, Union

from torch import nn
from torch import Tensor

from flambe.nn.module import Module


def conv_block(conv_mod: nn.Module,
               activation: nn.Module,
               pooling: nn.Module,
               dropout: float,
               batch_norm: Optional[nn.Module] = None) -> nn.Module:
    """Return a convolutional block.

    """

    mods = [conv_mod]

    if pooling:
        mods.append(pooling)

    if batch_norm is None:
        mods.append(batch_norm)

    mods.append(activation)
    mods.append(nn.Dropout(dropout))

    return nn.Sequential(*mods)


class CNNEncoder(Module):
    """Implements a multi-layer n-dimensional CNN.

    This module can be used to create multi-layer CNN models.

    Attributes
    ----------
    cnn: nn.Module
        The cnn submodule

    """
    def __init__(self,
                 input_channels: int,
                 channels: List[int],
                 conv_dim: int = 2,  # Support only for 1, 2 or 3
                 kernel_size: Union[int, List[Union[Tuple[int, ...], int]]] = 3,
                 activation: nn.Module = None,
                 pooling: nn.Module = None,
                 dropout: float = 0,
                 batch_norm: bool = True,
                 stride: int = 1,
                 padding: int = 0) -> None:
        """Initializes the CNNEncoder object.

        Parameters
        ----------
        input_channels: int
            The input's channels. For example, 3 for RGB images.
        channels: List[int]
            A list to specify the channels of the convolutional layers.
            The length of this list will be the amount of convolutions
            in the encoder.
        conv_dim: int, optional
            The dimension of the convolutions. Can be 1, 2 or 3.
            Defaults to 2.
        kernel_size: Union[int, List[Union[Tuple[int], int]]], optional
            The kernel size for the convolutions. This could be an int
            (the same kernel size for all convolutions and dimensions),
            or a List where for each convolution you can specify an int
            or a tuple (for different sizes per dimension, in which case
            the length of the tuple must match the dimension of the
            convolution). Defaults to 3.
        activation: nn.Module, optional
           The activation function to use in all layers.
           Defaults to nn.ReLU
        pooling: nn.Module, optional
            The pooling function to use after all layers.
            Defaults to None
        dropout: float, optional
            Amount of dropout to use between CNN layers, defaults to 0
        batch_norm: bool, optional
            Wether to user Batch Normalization or not. Defaults to True
        stride: int, optional
            The stride to use when doing convolutions. Defaults to 1
        padding: int, optional
            The padding to use when doing convolutions. Defaults to 0

        Raises
        ------
        ValueError
            The conv_dim should be 1, 2, 3.

        """
        super().__init__()

        dim2mod = {
            1: (nn.Conv1d, nn.BatchNorm1d, nn.MaxPool1d),
            2: (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d),
            3: (nn.Conv3d, nn.BatchNorm3d, nn.MaxPool3d),
        }

        if conv_dim not in dim2mod:
            raise ValueError(f"Invalid conv_dim value {conv_dim}. Values 1, 2, 3 supported")

        if isinstance(kernel_size, List) and len(kernel_size) != len(channels):
            raise ValueError("Kernel size list should have same length as channels list")

        conv, bn, pool = dim2mod[conv_dim]
        activation = activation or nn.ReLU()

        layers = []

        prev_c = input_channels
        for i, c in enumerate(channels):
            k: Union[int, Tuple]
            if isinstance(kernel_size, int):
                k = kernel_size
            else:
                k = kernel_size[i]
                if not isinstance(k, int) and len(k) != conv_dim:
                    raise ValueError("Kernel size tuple should have same length as conv_dim")

            layer = conv_block(
                conv(prev_c, c, k, stride, padding),
                activation,
                pooling,
                dropout,
                bn(c)
            )
            layers.append(layer)
            prev_c = c

        self.cnn = nn.Sequential(*layers)

    def forward(self, data: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor

        Returns
        -------
        Union[Tensor, Tuple[Tensor, ...]]
            The encoded output, as a float tensor

        """
        return self.cnn(data)
