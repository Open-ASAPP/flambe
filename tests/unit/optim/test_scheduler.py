import math

from torch.optim import Adam

from flambe.nn import MLPEncoder
from flambe.optim import NoamScheduler, WarmupLinearScheduler


def test_warmup_linear():
    model = MLPEncoder(10, 10)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = WarmupLinearScheduler(optimizer, warmup=100, n_steps=200)

    assert scheduler.get_lr()[0] == 0
    scheduler.step()
    assert scheduler.get_lr()[0] == 1e-5

    for _ in range(99):
        scheduler.step()
    assert scheduler.get_lr()[0] == 1e-3
    for _ in range(100):
        scheduler.step()
    assert scheduler.get_lr()[0] == 0


def test_noam():
    model = MLPEncoder(10, 10)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = NoamScheduler(optimizer, warmup=100, d_model=512)

    assert scheduler.get_lr()[0] == 0
    scheduler.step()
    assert math.isclose(scheduler.get_lr()[0], 0.001 / (512 ** 0.5) / (100 ** 1.5))
