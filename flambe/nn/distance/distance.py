# type: ignore[override]

from torch import Tensor

from flambe.nn.module import Module


class DistanceModule(Module):
    """Implement a DistanceModule object.

    """

    def forward(self, mat_1: Tensor, mat_2: Tensor) -> Tensor:
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
        raise NotImplementedError


class MeanModule(Module):
    """Implement a MeanModule object.

    """
    def __init__(self, detach_mean: bool = False) -> None:
        """Initilaize the MeanModule.

        Parameters
        ----------
        detach_mean : bool, optional
            Set to detach the mean computation, this is useful when the
            mean computation does not admit a closed form.

        """
        super().__init__()
        self.detach_mean = detach_mean

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
        raise NotImplementedError
