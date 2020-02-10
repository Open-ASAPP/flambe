from typing import Dict

import torch
import sklearn.metrics
import numpy as np

from flambe.metric.metric import Metric


class AUC(Metric):

    def __init__(self, max_fpr=1.0):
        """Initialize the AUC metric.

        Parameters
        ----------
        max_fpr : float, optional
            Maximum false positive rate to compute the area under
        """
        self.max_fpr = max_fpr

    def __str__(self) -> str:
        """Return the name of the Metric (for use in logging)."""
        return f'AUC@{self.max_fpr}'

    @staticmethod
    def aggregate(state: dict, *args, **kwargs) -> Dict:
        """Aggregates by simply storing preds and targets

        Parameters
        ----------
        state: dict
            the metric state
        args: the pred, target tuple

        Returns
        -------
        dict
            the state dict
        """
        pred, target = args
        if not state:
            state['pred'] = []
            state['target'] = []
        state['pred'].append(pred.cpu().detach())
        state['target'].append(target.cpu().detach())
        return state

    def finalize(self, state: Dict) -> float:
        """Finalizes the metric computation

        Parameters
        ----------
        state: dict
            the metric state

        Returns
        -------
        float
            The final score.
        """
        if not state:
            # call on empty state
            return np.NaN
        pred = torch.cat(state['pred'], dim=0)
        target = torch.cat(state['target'], dim=0)
        state['accumulated_score'] = self.compute(pred, target)
        return state['accumulated_score']

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute AUC at the given max false positive rate.

        Parameters
        ----------
        pred : torch.Tensor
            The model predictions
        target : torch.Tensor
            The binary targets

        Returns
        -------
        torch.Tensor
            The computed AUC

        """
        scores = np.array(pred)
        targets = np.array(target)

        # Case when number of elements added are 0
        if not scores.size or not targets.size:
            return torch.tensor(0.5)

        fpr, tpr, _ = sklearn.metrics.roc_curve(targets, scores, sample_weight=None)

        # Compute the area under the curve using trapezoidal rule
        max_index = np.searchsorted(fpr, [self.max_fpr], side='right').item()

        # Ensure we integrate up to max_fpr
        fpr, tpr = fpr.tolist(), tpr.tolist()
        fpr, tpr = fpr[:max_index], tpr[:max_index]
        fpr.append(self.max_fpr)
        tpr.append(max(tpr))

        area = np.trapz(tpr, fpr)

        return torch.tensor(area / self.max_fpr).float()
