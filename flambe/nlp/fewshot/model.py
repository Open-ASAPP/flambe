# type: ignore[override]

from typing import Tuple, Dict, Any, Union, Optional

import torch
from torch import Tensor

from flambe.nn import Embedder, Module
from flambe.nn.distance import get_distance_module, get_mean_module


class PrototypicalTextClassifier(Module):
    """Implements a standard classifier.

    The classifier is composed of an encoder module, followed by
    a fully connected output layer, with a dropout layer in between.

    Attributes
    ----------
    encoder: Module
        the encoder object
    decoder: Decoder
        the decoder layer
    drop: nn.Dropout
        the dropout layer
    loss: Metric
        the loss function to optimize the model with
    metric: Metric
        the dev metric to evaluate the model on

    """

    def __init__(self,
                 embedder: Embedder,
                 distance: str = 'euclidean',
                 detach_mean: bool = False) -> None:
        """Initialize the TextClassifier model.

        Parameters
        ----------
        embedder: Embedder
            The embedder layer

        """
        super().__init__()

        self.embedder = embedder

        self.distance_module = get_distance_module(distance)
        self.mean_module = get_mean_module(distance)
        self.detach_mean = detach_mean

    def compute_prototypes(self, support: Tensor, label: Tensor) -> Tensor:
        """Set the current prototypes used for classification.

        Parameters
        ----------
        data : torch.Tensor
            Input encodings
        label : torch.Tensor
            Corresponding labels

        """
        means_dict: Dict[int, Any] = {}
        for i in range(support.size(0)):
            means_dict.setdefault(int(label[i]), []).append(support[i])

        means = []
        n_means = len(means_dict)

        for i in range(n_means):
            # Ensure that all contiguous indices are in the means dict
            supports = torch.stack(means_dict[i], dim=0)
            if supports.size(0) > 1:
                mean = self.mean_module(supports).squeeze(0)
            else:
                mean = supports.squeeze(0)
            means.append(mean)

        prototypes = torch.stack(means, dim=0)
        return prototypes

    def forward(self,  # type: ignore
                query: Tensor,
                query_label: Optional[Tensor] = None,
                support: Optional[Tensor] = None,
                support_label: Optional[Tensor] = None,
                prototypes: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        """Run a forward pass through the network.

        Parameters
        ----------
        data: Tensor
            The input data

        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor]]
            The output predictions

        """
        query_encoding = self.embedder(query)
        if isinstance(query_encoding, tuple):  # RNN
            query_encoding = query_encoding[0]

        if prototypes is not None:
            prototypes = prototypes
        elif support is not None and support_label is not None:
            if self.detach_mean:
                support = support.detach()
                support_label = support_label.detach()  # type: ignore

            support_encoding = self.embedder(support)
            if isinstance(support_encoding, tuple):  # RNN
                support_encoding = support_encoding[0]

            # Compute prototypes
            prototypes = self.compute_prototypes(support_encoding, support_label)
        else:
            raise ValueError("No prototypes set or provided")

        dist = self.distance_module(query_encoding, prototypes)
        if query_label is not None:
            return - dist, query_label
        else:
            return - dist
