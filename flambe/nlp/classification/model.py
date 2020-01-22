# type: ignore[override]

from typing import Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from flambe.nn import Embedder, Module


class TextClassifier(Module):
    """Implements a standard classifier.

    The classifier is composed of an encoder module, followed by
    a fully connected output layer, with a dropout layer in between.

    Attributes
    ----------
    embedder: Embedder
        The embedder layer
    output_layer : Module
        The output layer, yields a probability distribution over targets
    drop: nn.Dropout
        the dropout layer
    loss: Metric
        the loss function to optimize the model with
    metric: Metric
        the dev metric to evaluate the model on

    """

    def __init__(self,
                 embedder: Embedder,
                 output_layer: Module,
                 dropout: float = 0) -> None:
        """Initialize the TextClassifier model.

        Parameters
        ----------
        embedder: Embedder
            The embedder layer
        output_layer : Module
            The output layer, yields a probability distribution
        dropout : float, optional
            Amount of dropout to include between layers (defaults to 0)

        """
        super().__init__()

        self.embedder = embedder
        self.output_layer = output_layer

        self.drop = nn.Dropout(dropout)

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
        outputs = self.embedder(data)
        if isinstance(outputs, tuple):
            encoding = outputs[0]
        else:
            encoding = outputs

        pred = self.output_layer(self.drop(encoding))
        return (pred, target) if target is not None else pred
