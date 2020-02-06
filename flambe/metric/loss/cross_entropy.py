from typing import Optional

import torch
import torch.nn.functional as F

from flambe.metric.metric import Metric


class MultiLabelCrossEntropy(Metric):

    def __init__(self,
                 weight: Optional[torch.Tensor] = None,
                 ignore_index: Optional[int] = None,
                 reduction: str = 'mean') -> None:
        """Initialize the MultiLabelCrossEntropy.

        Parameters
        ----------
        weight : Optional[torch.Tensor]
            A manual rescaling weight given to each class.
            If given, has to be a Tensor of size N, where N is the
            number of classes.
        ignore_index : Optional[int], optional
            Specifies a target value that is ignored and does not
            contribute to the input gradient. When size_average is
            True, the loss is averaged over non-ignored targets.
        reduction : str, optional
            Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'.
            'none': no reduction will be applied,
            'mean': the output will be averaged
            'sum': the output will be summed.
        """
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __str__(self) -> str:
        """Return the name of the Metric (for use in logging)."""
        return 'MultiLabelCrossEntropy' if self.weight is None \
            else 'WeightedMultiLabelCrossEntropy'

    def compute(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        """Computes the multilabel cross entropy loss.

        Parameters
        ----------
        pred: torch.Tensor
            input logits of shape (B x N)
        target: torch.LontTensor
            target tensor of shape (B x N)

        Returns
        -------
        loss: torch.Tensor
            Multi label cross-entropy loss, of shape (B)

        """
        if self.ignore_index is not None:
            target[:, self.ignore_index] = 0

        if self.weight is None:
            self.weight = torch.ones(pred.size(1)).to(pred)

        norm_target = F.normalize(target.float(), p=1, dim=1)
        loss = - (self.weight * norm_target *
                  F.log_softmax(pred, dim=1)).sum(dim=1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction is not None:
            raise ValueError("Unknown reduction: {self.reduction}")

        return loss
