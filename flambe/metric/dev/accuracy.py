import torch

from flambe.metric.metric import Metric


class Accuracy(Metric):

    def compute(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        """Computes the loss.

        Parameters
        ----------
        pred: Tensor
            input logits of shape (B x N)
        target: LontTensor
            target tensor of shape (B) or (B x N)

        Returns
        -------
        accuracy: torch.Tensor
            single label accuracy, of shape (B)

        """
        # If 2-dimensional, select the highest score in each row
        if len(target.size()) == 2:
            target = target.argmax(dim=1)

        acc = (pred.argmax(dim=1) == target)
        return acc.float().mean()
