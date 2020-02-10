from abc import abstractmethod

import torch

from flambe.metric.metric import Metric


class BinaryMetric(Metric):

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize the Binary metric.

        Parameters
        ---------
        threshold: float
            Given a probability p of belonging to Positive class,
            p < threshold will be considered tagged as Negative by
            the classifier when computing the metric.
            Defaults to 0.5
        """
        self.threshold = threshold

    def __str__(self) -> str:
        """Return the name of the Metric (for use in logging)."""
        return f'{self.__class__.__name__}@{self.threshold}'

    def compute(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        """Compute the metric given predictions and targets

        Parameters
        ----------
        pred : Tensor
            The model predictions
        target : Tensor
            The binary targets

        Returns
        -------
        float
            The computed binary metric

        """
        pred = pred.squeeze()
        target = target.squeeze().bool()
        pred = (pred > self.threshold)

        return self.compute_binary(pred, target)

    @abstractmethod
    def compute_binary(self,
                       pred: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
        """Compute a binary-input metric.

        Parameters
        ---------
        pred: torch.Tensor
            Predictions made by the model. It should be a probability
            0 <= p <= 1 for each sample, 1 being the positive class.
        target: torch.Tensor
            Ground truth. Each label should be either 0 or 1.

        Returns
        ------
        torch.float
            The computed binary metric

        """
        pass


class BinaryAccuracy(BinaryMetric):
    """Compute binary accuracy.

    ```
    |True positives + True negatives| / N
    ```

    """
    def compute_binary(self,
                       pred: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
        """Compute binary accuracy.

        Parameters
        ---------
        pred: torch.Tensor
            Predictions made by the model. It should be a probability
            0 <= p <= 1 for each sample, 1 being the positive class.
        target: torch.Tensor
            Ground truth. Each label should be either 0 or 1.

        Returns
        ------
        torch.float
            The computed binary metric

        """
        acc = pred == target
        N = target.size()[0] if target.dim() > 0 else 1

        if N == 0:
            return torch.tensor(0)

        return acc.sum().float() / N


class BinaryPrecision(BinaryMetric):
    """Compute Binary Precision.

    An example is considered negative when its score is below the
    specified threshold. Binary precition is computed as follows:

    ```
    |True positives| / |True Positives| + |False Positives|
    ```

    """

    def __init__(self, threshold: float = 0.5, positive_label: int = 1) -> None:
        """Initialize the Binary metric.

        Parameters
        ---------
        threshold: float
            Given a probability p of belonging to Positive class,
            p < threshold will be considered tagged as Negative by
            the classifier when computing the metric.
            Defaults to 0.5
        positive_label: int
            Specify if the positive class should be 1 or 0.
            Defaults to 1.

        """
        if positive_label not in [0, 1]:
            raise ValueError("positive_label should be either 0 or 1")

        super().__init__(threshold)
        self.positive_label = positive_label

    def compute_binary(self,
                       pred: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
        """Compute binary precision.

        Parameters
        ---------
        pred: torch.Tensor
            Predictions made by the model. It should be a probability
            0 <= p <= 1 for each sample, 1 being the positive class.
        target: torch.Tensor
            Ground truth. Each label should be either 0 or 1.

        Returns
        ------
        torch.float
            The computed binary metric

        """
        if self.positive_label == 0:
            pred = ~pred
            target = ~target

        acc = pred == target
        true_p = acc & target

        if pred.sum() == 0:
            metric = torch.tensor(0)
        else:
            # Again, weird typing from pytorch
            # check periodically for a fix
            metric = (true_p.sum().float() / pred.sum().float())

        return metric

    def __str__(self) -> str:
        """Return the name of the Metric (for use in logging)."""
        invert_label = "Negative" if self.positive_label == 0 else "Positive"
        return f"{invert_label}{self.__class__.__name__}"


class BinaryRecall(BinaryMetric):
    """Compute binary recall.

    An example is considered negative when its score is below the
    specified threshold. Binary precition is computed as follows:

    ```
    |True positives| / |True Positives| + |False Negatives|
    ```

    """
    def __init__(self, threshold: float = 0.5, positive_label: int = 1) -> None:
        """Initialize the Binary metric.

        Parameters
        ---------
        threshold: float
            Given a probability p of belonging to Positive class,
            p < threshold will be considered tagged as Negative by
            the classifier when computing the metric.
            Defaults to 0.5
        positive_label: int
            Specify if the positive class should be 1 or 0.
            Defaults to 1.

        """
        if positive_label not in [0, 1]:
            raise ValueError("positive_label should be either 0 or 1")

        super().__init__(threshold)
        self.positive_label = positive_label

    def compute_binary(self,
                       pred: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
        """Compute binary recall.

        Parameters
        ---------
        pred: torch.Tensor
            Predictions made by the model. It should be a probability
            0 <= p <= 1 for each sample, 1 being the positive class.
        target: torch.Tensor
            Ground truth. Each label should be either 0 or 1.

        Returns
        ------
        torch.float
            The computed binary metric

        """
        if self.positive_label == 0:
            pred = ~pred
            target = ~target

        acc = pred == target
        true_p = acc & target

        if target.sum() == 0:
            metric = torch.tensor(0)
        else:
            metric = true_p.sum().float() / target.sum().float()

        return metric

    def __str__(self) -> str:
        """Return the name of the Metric (for use in logging)."""
        invert_label = "Negative" if self.positive_label == 0 else "Positive"
        return f"{invert_label}{self.__class__.__name__}"


class F1(BinaryMetric):

    def __init__(self,
                 threshold: float = 0.5,
                 positive_label: int = 1,
                 eps: float = 1e-8) -> None:
        """
        Parameters
        ---------
        threshold: float
            Given a probability p of belonging to Positive class,
            p < threshold will be considered tagged as Negative by
            the classifier when computing the metric.
            Defaults to 0.5
        positive_label: int
            Specify if the positive class should be 1 or 0.
            Defaults to 1.
        eps: float
            Float to sum to the denominator, so that we avoid division
            by zero. Defaults to 1e-8.

        """

        super().__init__(threshold)
        self.recall = BinaryRecall(threshold, positive_label)
        self.precision = BinaryPrecision(threshold, positive_label)
        self.eps = eps

    def compute_binary(self,
                       pred: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
        """Compute F1. Score, the harmonic mean between precision and
        recall.

        Parameters
        ---------
        pred: torch.Tensor
            Predictions made by the model. It should be a probability
            0 <= p <= 1 for each sample, 1 being the positive class.
        target: torch.Tensor
            Ground truth. Each label should be either 0 or 1.

        Returns
        ------
        torch.float
            The computed binary metric

        """
        recall = self.recall.compute_binary(pred, target)
        precision = self.precision.compute_binary(pred, target)

        return 2 * precision * recall / (precision + recall + self.eps)
