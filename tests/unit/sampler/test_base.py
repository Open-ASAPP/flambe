import pytest
import torch

from flambe.sampler import BaseSampler


def test_compose_padded_batches_from_nested_seq():
    """
    In the first two observations:
        * The max number of first-level elements is 4.
        * The max size of a first-level element is 8.
    In the second two observations:
        * The max number of first-level elements is 2.
        * The max size of a first-level element is 5.

    > The first batch should be of size (bs, 4, 8)
    > The second batch should be of size (bs, 2, 5)
    """
    bs = 2
    data = (
        (
            torch.tensor([7]),  # Not nested
            [torch.tensor([1, 2]), torch.tensor([3, 4, 5, 6, 7])],  # Nested
            [torch.tensor([1, 2]), torch.tensor([1, 2]), torch.tensor([3, 4, 5])]  # Nested
        ),
        (
            torch.tensor([7, 8]),  # Not nested
            [torch.tensor([7, 8]), torch.tensor([7, 8]), torch.tensor([7, 8]), torch.tensor([7, 8])],  # Nested
            [torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])]  # Nested
        ),
        (
            torch.tensor([7, 8, 9]),  # Not nested
            [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],  # Nested
            [torch.tensor([1, 2])]  # Nested
        ),
        (
            torch.tensor([7, 8, 9, 10]),  # Not nested
            [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8, 9])],  # Nested
            [torch.tensor([1, 2, 4, 5, 6, 7, 8])]  # Nested
        ),
    )
    sampler = BaseSampler(
        batch_size=bs,
        shuffle=False,
        pad_index=0
    )
    sampler = sampler.sample(data)

    batch = next(sampler)
    a, b, c = batch
    assert a.size() == (bs, 2)  # In first batch, first col: largest element has length 2
    assert b.size() == (bs, 4, 5)  # In first batch, second col: largest child has length 5; largest seq. of children has length 4
    assert c.size() == (bs, 3, 8)  # In first batch, third col: largest child has length 8; largest seq. of children has length 3

    batch = next(sampler)
    a, b, c = batch
    assert a.size() == (bs, 4) # In second batch, first col: largest element has length 4
    assert b.size() == (bs, 2, 6)  # In first batch, second col: largest child has length 6; largest seq. of children has length 2
    assert c.size() == (bs, 1, 7)  # In first batch, third col: largest child has length 7; largest seq. of children has length 1


def test_column_specific_pad_indexes():
    """
    The first column should be padded to 10 elements; (10 - 1) + (10 - 4) + (10 - 10) of these should be -2.

    The second column should be padded to 10 elements; (10 - 1) + (10 - 5) + (10 - 10) of these should be -1.

    The third column should be padded to 10 elements; (10 - 1) + (10 - 6) + (10 - 10) of these should be 0.
    """
    bs = 3
    data = (
        (
            torch.tensor(1 * [1]),
            torch.tensor(1 * [2]),
            torch.tensor(1 * [3])
        ),
        (
            torch.tensor(4 * [1]),
            torch.tensor(5 * [2]),
            torch.tensor(6 * [3])
        ),
        (
            torch.tensor(10 * [1]),
            torch.tensor(10 * [2]),
            torch.tensor(10 * [3])
        )
    )
    sampler = BaseSampler(
        batch_size=bs,
        shuffle=False,
        pad_index=(-2, -1, 0)
    )
    sampler = sampler.sample(data)
    batch = next(sampler)
    a, b, c = batch

    assert (a == -2).sum() == (10 - 1) + (10 - 4) + (10 - 10)
    assert (b == -1).sum() == (10 - 1) + (10 - 5) + (10 - 10)
    assert (c == 0).sum() == (10 - 1) + (10 - 6) + (10 - 10)


def test_incorrect_num_column_specific_pad_indexes_raises_error():
    """
    The first column should be padded to 10 elements; (10 - 1) + (10 - 4) + (10 - 10) of these should be -2.

    The second column should be padded to 10 elements; (10 - 1) + (10 - 5) + (10 - 10) of these should be -1.

    The third column should be padded to 10 elements; (10 - 1) + (10 - 6) + (10 - 10) of these should be 0.
    """
    bs = 3
    data = (
        (
            torch.tensor(1 * [1]),
            torch.tensor(1 * [2]),
            torch.tensor(1 * [3])
        ),
        (
            torch.tensor(4 * [1]),
            torch.tensor(5 * [2]),
            torch.tensor(6 * [3])
        ),
        (
            torch.tensor(10 * [1]),
            torch.tensor(10 * [2]),
            torch.tensor(10 * [3])
        )
    )
    num_cols = len(data)

    sampler = BaseSampler(
        batch_size=bs,
        shuffle=False,
        pad_index=(num_cols + 99) * (0,)
    )
    sampler = sampler.sample(data)
    with pytest.raises(Exception):
        batch = next(sampler)
