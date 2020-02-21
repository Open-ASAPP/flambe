import pytest

from torch.nn import NLLLoss
from torch.optim import Adam

from flambe.learn import train

from flambe.dataset import Dataset
from flambe.compile import Schema, State, Component, Link
from flambe.learn.utils import select_device
from flambe.nn import Module  # type: ignore[attr-defined]
from flambe.sampler import BaseSampler
from flambe.metric import Metric
from flambe.logging import log


class DummyDataset(Dataset):
    @property
    def train(self):
        return [['hello']]

    @property
    def val(self):
        pass

    @property
    def test(self):
        pass


class DummyModel(Module):
    pass


@pytest.fixture()
def trainer():
    return train.Trainer(
        dataset=DummyDataset(),
        train_sampler=BaseSampler(),
        val_sampler=BaseSampler(),
        model=DummyModel(),
        loss_fn=NLLLoss(),
        metric_fn=NLLLoss(),
        optimizer=Adam,
        extra_validation_metrics=[NLLLoss()] * 3
    )


def test_validation_metrics_property(trainer):
    assert trainer.validation_metrics == trainer.extra_validation_metrics
    assert len(trainer.validation_metrics) == 3
