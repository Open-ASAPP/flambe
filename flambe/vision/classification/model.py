import torch
from torch import Tensor

from typing import Optional, Tuple, Union
from flambe.nn import Module  # type: ignore[attr-defined]


class ImageClassifier(Module):
    """Implements a simple image classifier.

    This classifier consists of an encocder module, followed by
    a fully connected output layer that outputs a probability
    distribution.

    Attributes
    ----------
    encoder: Moodule
        The encoder layer
    output_layer: Module
        The output layer, yields a probability distribution over targets
    """
    def __init__(self,
                 encoder: Module,
                 output_layer: Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.output_layer = output_layer

    def forward(self,
                data: Tensor,
                target: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Run a forward pass through the network.

        Parameters
        ----------
        data: Tensor
            The input data
        target: Tensor, optional
            The input targets, optional

        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor]
            The output predictions, and optionally the targets

        """
        encoded = self.encoder(data)
        pred = self.output_layer(torch.flatten(encoded, 1))
        return (pred, target) if target is not None else pred
