# type: ignore[override]

from typing import Optional

import torch

from flambe.nn import Module


class FirstPooling(Module):
    """Get the last hidden state of a sequence."""

    def forward(self,
                data: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        return data[:, 0, :]


class LastPooling(Module):
    """Get the last hidden state of a sequence."""

    def forward(self,
                data: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        # Compute lengths
        if padding_mask is None:
            lengths = torch.tensor([data.size(1)] * data.size(0)).long()
        else:
            lengths = padding_mask.long().sum(dim=1)

        return data[torch.arange(data.size(0)).long(), lengths - 1, :]


class SumPooling(Module):
    """Get the sum of the hidden state of a sequence."""

    def forward(self,
                data: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        padding_mask = padding_mask or _default_padding_mask(data)

        return _sum_with_padding_mask(data, padding_mask)


class AvgPooling(Module):
    """Get the average of the hidden state of a sequence."""

    def forward(self,
                data: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        padding_mask = padding_mask or _default_padding_mask(data)
        value_count = padding_mask.sum(dim=1).unsqueeze(1)
        data = _sum_with_padding_mask(data, padding_mask)
        return data / value_count


def _default_padding_mask(data: torch.Tensor) -> torch.Tensor:
    """
    Builds a 1s padding mask taking into account initial 2 dimensions
    of input data.

    Parameters
    ----------
    data : torch.Tensor
        The input data, as a tensor of shape [B x S x H]

    Returns
    ----------
    torch.Tensor
        A padding mask , as a tensor of shape [B x S]
    """
    return torch.ones((data.size(0), data.size(1))).to(data)


def _sum_with_padding_mask(data: torch.Tensor,
                           padding_mask: torch.Tensor) -> torch.Tensor:
    """
    Applies padding_mask and performs summation over the data

    Parameters
    ----------
    data : torch.Tensor
        The input data, as a tensor of shape [B x S x H]
    padding_mask: torch.Tensor
        The input mask, as a tensor of shape [B X S]
    Returns
    ----------
    torch.Tensor
        The result of the summation, as a tensor of shape [B x H]

    """
    return (data * padding_mask.unsqueeze(2)).sum(dim=1)
