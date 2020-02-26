import torch

from flambe.metric.metric import Metric


class Recall(Metric):

    def __init__(self, top_k: int = 1) -> None:
        """Initialize the Recall metric.

        Parameters
        ---------
        top_k: int
            used to compute recall@k. For k = 1, this becomes
            accuracy
        """
        self.top_k = top_k

    def __str__(self) -> str:
        """Return the name of the Metric (for use in logging)."""
        return f'{self.__class__.__name__}@{self.top_k}'

    def compute(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        """Computes the recall @ k.

        Parameters
        ----------
        pred: Tensor
            input logits of shape (B x N)
        target: LongTensor
            target tensor of shape (B) or (B x N)

        Returns
        -------
        recall: torch.Tensor
            single label recall, of shape (B)

        """
        # If 2-dimensional, select the highest score in each row
        if len(target.size()) == 2:
            target = target.argmax(dim=1)

        ranked_scores = torch.argsort(pred, dim=1)[:, -self.top_k:]
        recalled = torch.sum((target.unsqueeze(1) == ranked_scores).float(), dim=1)
        return recalled.mean()
