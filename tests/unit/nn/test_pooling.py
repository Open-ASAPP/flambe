import torch

from flambe.nn import AvgPooling, SumPooling
from torch import allclose


def test_avg_pooling():
    layer = AvgPooling()
    average = layer.forward(build_tensor())
    assert allclose(average[0], torch.tensor([10 / 4, 14 / 4, 18 / 4], dtype=torch.float32))
    assert allclose(average[1], torch.tensor([0, 0, 0], dtype=torch.float32))


def test_sum_pooling():
    layer = SumPooling()
    summation = layer.forward(build_tensor())
    assert allclose(summation[0], torch.tensor([10, 14, 18], dtype=torch.float32))
    assert allclose(summation[1], torch.tensor([0, 0, 0], dtype=torch.float32))


def build_tensor():
    batch_size = 2
    items = 4
    embedding_size = 3
    data = torch.zeros([batch_size, items, embedding_size], dtype=torch.float32)
    data[0][0] = torch.tensor([1, 2, 3])
    data[0][1] = torch.tensor([4, 5, 6])
    data[0][2] = torch.tensor([1, 2, 3])
    data[0][3] = torch.tensor([4, 5, 6])
    return data
