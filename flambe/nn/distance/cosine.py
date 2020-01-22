# type: ignore[override]

import torch
from torch import Tensor
from flambe.nn.distance import DistanceModule, MeanModule


class CosineDistance(DistanceModule):
    """Implement a CosineDistance object.

    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize the CosineDistance module.

        Parameters
        ----------
        eps : float, optional
            Used for numerical stability

        """
        super().__init__()
        self.eps = eps

    def forward(self, mat_1: Tensor, mat_2: Tensor) -> Tensor:
        """Returns the cosine distance between each
        element in mat_1 and each element in mat_2.

        Parameters
        ----------
        mat_1: torch.Tensor
            matrix of shape (n_1, n_features)
        mat_2: torch.Tensor
            matrix of shape (n_2, n_features)

        Returns
        -------
        dist: torch.Tensor
            distance matrix of shape (n_1, n_2)

        """
        w1 = mat_1.norm(p=2, dim=1, keepdim=True)
        w2 = mat_2.norm(p=2, dim=1, keepdim=True)
        return 1 - torch.mm(mat_1, mat_2.t()) / (w1 * w2.t()).clamp(min=self.eps)


class CosineMean(MeanModule):
    """Implement a CosineMean object.

    """
    def forward(self, data: Tensor) -> Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor

        """
        data = data / (data.norm(dim=1, keepdim=True))
        return data.mean(0)
