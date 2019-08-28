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
            The input data, as a tensor of shape [S x B x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [S X B]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        return data[0, :, :]


class LastPooling(Module):
    """Get the last hidden state of a sequence."""

    def forward(self,
                data: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [S x B x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [S X B]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        # Compute lengths
        if padding_mask is None:
            lengths = torch.tensor([data.size(0)] * data.size(1)).long()
        else:
            lengths = padding_mask.long().sum(dim=0)

        return data[lengths - 1, torch.arange(data.size(1)).long(), :]


class SumPooling(Module):
    """Get the sum of the hidden state of a sequence."""

    def forward(self,
                data: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [S x B x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [S X B]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        # Apply pooling
        if padding_mask is None:
            padding_mask = torch.ones((data.size(0), data.size(1))).to(data)

        return (data * padding_mask.unsqueeze(2)).sum(dim=0)


class AvgPooling(Module):
    """Get the average of the hidden state of a sequence."""

    def forward(self,
                data: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [S x B x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [S X B]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        # Apply pooling
        if padding_mask is None:
            padding_mask = torch.ones((data.size(0), data.size(1))).to(data)

        output = (output * padding_mask.unsqueeze(2)).sum(dim=0)
        return output / padding_mask.sum(dim=0)
