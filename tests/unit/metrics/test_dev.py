import torch
import sklearn.metrics
import numpy as np

from flambe.metric import Accuracy, AUC, Perplexity, BPC
from flambe.metric import BinaryRecall, BinaryPrecision, BinaryAccuracy
from pytest import approx

NUMERIC_PRECISION = 1e-2


def metric_test_case(predicted, true, metric, expected):
    assert metric(predicted, true).item() == approx(expected, NUMERIC_PRECISION)


def test_auc_full():
    """Test random score list."""
    y_pred = np.random.randint(0, 10000, 100)
    y_true = np.random.randint(0, 2, 100)

    metric_test_case(y_pred, y_true, AUC(), sklearn.metrics.roc_auc_score(y_true, y_pred))


def test_auc_threshold():
    """To compute rates:
         -> fpr: number of false positives / total number of negatives
         -> tpr: number of true positives / total number of positives
    In this example, consider the first threshold of 0.1, and consider everything
    higher than 0.1 as positives, and everything lower as negatives:
        fpr = 1 / 2 = 0.5
        tpr = 1
    Thus, if the max_fpr rate is 0.1, the tpr is 0, and if the max_fpr is 0.5,
    then the tpr is 1. So we get an auc of 0 for the 0.1 threshold, and 0.5
    for the 0.5 threshold.
    For more details: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    """
    y_pred = torch.tensor([0.1, 0.2, 0.3, 0.4])
    y_true = torch.tensor([0, 1, 0, 1])

    metric_test_case(y_pred, y_true, AUC(max_fpr=1.), sklearn.metrics.roc_auc_score(y_true, y_pred))
    metric_test_case(y_pred, y_true, AUC(max_fpr=0.500001), 0.5)
    metric_test_case(y_pred, y_true, AUC(max_fpr=0.1), 0.5)


def test_auc_empty():
    """Should be completely random on an empty list"""
    y_pred = torch.tensor([])
    y_true = torch.tensor([])

    auc = AUC()
    assert auc(y_pred, y_true).item() == approx(0.5, NUMERIC_PRECISION)


def test_accuracy():
    """Test random score list."""
    metric_test_case(torch.tensor([[0.1, 0.2], [0.9, 0.1]]), torch.tensor([1, 1]), Accuracy(), 0.5)
    metric_test_case(torch.tensor([[1.0, 0.0], [0.6, 0.4]]), torch.tensor([1, 1]), Accuracy(), 0)


def test_perplexity():
    """Test perplexity"""
    metric_test_case(torch.tensor([[0.2, 0.8], [0.9, 0.1]]), torch.tensor([0, 0]), Perplexity(), 2.022418975830078)


def test_bpc():
    """Test BPC"""
    metric_test_case(torch.tensor([[0.2, 0.8], [0.9, 0.1]]), torch.tensor([0, 0]), BPC(), 1.0161)


def test_binary_precision():
    """
        precision = tp/tp+fp
    """
    y_pred = torch.tensor([0.8, 0.1, 0.7, 0.1])

    metric_test_case(y_pred, torch.tensor([1, 1, 1, 1]), BinaryPrecision(), 1)
    metric_test_case(y_pred, torch.tensor([1, 0, 1, 0]), BinaryPrecision(), 1)
    metric_test_case(y_pred, torch.tensor([0, 0, 0, 0]), BinaryPrecision(), 0)
    metric_test_case(y_pred, torch.tensor([0, 1, 0, 1]), BinaryPrecision(), 0)

    # fp fn tp fn
    metric_test_case(y_pred, torch.tensor([0, 1, 1, 1]), BinaryPrecision(), 0.5)

    # tp tp
    metric_test_case(y_pred, torch.tensor([0, 1, 1, 1]), BinaryPrecision(0.65), 0.5)

    # tp tp
    metric_test_case(y_pred, torch.tensor([0, 1, 1, 1]), BinaryPrecision(0.75), 0)

    # tp fn tp tn
    metric_test_case(y_pred, torch.tensor([1, 1, 1, 0]), BinaryPrecision(), 1)


def test_inverted_binary_precision():
    """
        precision = tp/tp+fp
    """
    y_pred = torch.tensor([0.8, 0.1, 0.7, 0.1])

    metric_test_case(y_pred, torch.tensor([1, 1, 1, 1]), BinaryPrecision(positive_label=0), 0)
    metric_test_case(y_pred, torch.tensor([1, 0, 1, 0]), BinaryPrecision(positive_label=0), 1)
    metric_test_case(y_pred, torch.tensor([0, 0, 0, 0]), BinaryPrecision(positive_label=0), 1)
    metric_test_case(y_pred, torch.tensor([0, 1, 0, 1]), BinaryPrecision(positive_label=0), 0)
    metric_test_case(y_pred, torch.tensor([0, 0, 0, 1]), BinaryPrecision(positive_label=0), 0.5)
    metric_test_case(y_pred, torch.tensor([0, 1, 1, 1]), BinaryPrecision(positive_label=0), 0)
    metric_test_case(y_pred, torch.tensor([0, 0, 1, 1]), BinaryPrecision(0.65, positive_label=0), 0.5)
    metric_test_case(y_pred, torch.tensor([0, 0, 0, 1]), BinaryPrecision(0.75, positive_label=0), 2 / 3)


def test_binary_recall():
    """
    recall = tp/tp+fn
    """
    y_pred = torch.tensor([0.75, 0.3, 0.6, 0.1])

    # fp fn fp fn
    metric_test_case(y_pred, torch.tensor([0, 1, 0, 1]), BinaryRecall(), 0)

    # tp fn tp fn
    metric_test_case(y_pred, torch.tensor([1, 1, 1, 1]), BinaryRecall(), 0.5)

    # tp fn fn fn
    metric_test_case(y_pred, torch.tensor([1, 1, 1, 1]), BinaryRecall(0.7), 0.25)

    # tp tp tp fn
    metric_test_case(y_pred, torch.tensor([1, 1, 1, 1]), BinaryRecall(0.25), 0.75)

    # all tps
    metric_test_case(y_pred, torch.tensor([1, 1, 1, 1]), BinaryRecall(0.05), 1)

    # fp tn fp tn
    metric_test_case(y_pred, torch.tensor([0, 0, 0, 0]), BinaryRecall(), 0)

    # tp tn tp tn
    metric_test_case(y_pred, torch.tensor([1, 0, 1, 0]), BinaryRecall(), 1)

    # tp tn fn tn
    metric_test_case(y_pred, torch.tensor([1, 0, 1, 0]), BinaryRecall(0.7), 0.5)

    # fn tn fn tn
    metric_test_case(y_pred, torch.tensor([1, 0, 1, 0]), BinaryRecall(0.8), 0)


def test_inverted_binary_recall():
    """
        precision = tp/tp+fn
    """
    y_pred = torch.tensor([0.8, 0.1, 0.7, 0.1])

    metric_test_case(y_pred, torch.tensor([1, 1, 1, 1]), BinaryRecall(positive_label=0), 0)
    metric_test_case(y_pred, torch.tensor([1, 0, 1, 0]), BinaryRecall(positive_label=0), 1)
    metric_test_case(y_pred, torch.tensor([0, 0, 0, 0]), BinaryRecall(positive_label=0), 0.5)
    metric_test_case(y_pred, torch.tensor([0, 1, 0, 1]), BinaryRecall(positive_label=0), 0)
    metric_test_case(y_pred, torch.tensor([0, 0, 0, 1]), BinaryRecall(positive_label=0), 1 / 3)
    metric_test_case(y_pred, torch.tensor([0, 1, 1, 1]), BinaryRecall(positive_label=0), 0)
    metric_test_case(y_pred, torch.tensor([0, 0, 1, 1]), BinaryRecall(0.65, positive_label=0), 0.5)
    metric_test_case(y_pred, torch.tensor([0, 0, 0, 1]), BinaryRecall(0.75, positive_label=0), 2 / 3)


def test_binary_accuracy():
    """
        precision = tp/tp+fn
    """
    y_pred = torch.tensor([0.8, 0.1, 0.7, 0.1])

    metric_test_case(y_pred, torch.tensor([1, 0, 1, 0]), BinaryAccuracy(), 1)
    metric_test_case(y_pred, torch.tensor([1, 1, 1, 0]), BinaryAccuracy(), 3 / 4)
    metric_test_case(y_pred, torch.tensor([1, 1, 1, 1]), BinaryAccuracy(), 0.5)
    metric_test_case(y_pred, torch.tensor([0, 0, 0, 0]), BinaryAccuracy(), 0.5)
    metric_test_case(y_pred, torch.tensor([0, 1, 0, 1]), BinaryAccuracy(), 0)
    metric_test_case(y_pred, torch.tensor([1, 0, 0, 0]), BinaryAccuracy(0.75), 1)


def test_multiple_dims():
    """
        precision = tp/tp+fn
    """
    y_pred = torch.tensor([[0.8], [0.1], [0.7], [0.1]])
    metric_test_case(y_pred, torch.tensor([1, 0, 1, 0]), BinaryAccuracy(), 1)

    y_pred = torch.tensor([[[0.8]], [[0.1]], [[0.7]], [[0.1]]])
    metric_test_case(y_pred, torch.tensor([1, 0, 1, 0]), BinaryAccuracy(), 1)

    y_pred = torch.tensor([[[0.8]], [[0.1]], [[0.7]], [[0.1]]])
    metric_test_case(y_pred, torch.tensor([[1], [0], [1], [0]]), BinaryAccuracy(), 1)
