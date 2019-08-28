import pytest
from flambe.nn import CNNEncoder
import torch
import random


def test_invalid_conv_dim():
    with pytest.raises(ValueError):
        cnn_enc = CNNEncoder(
            input_channels=3,
            channels=[8,8,8],
            conv_dim=4
        )


def test_invalid_conv_dim2():
    with pytest.raises(ValueError):
        cnn_enc = CNNEncoder(
            input_channels=3,
            channels=[8,8,8],
            conv_dim=0
        )


def test_invalid_channels_kernels():
    with pytest.raises(ValueError):
        cnn_enc = CNNEncoder(
            input_channels=3,
            channels=[8,8,8],
            kernel_size=[3,3],
            conv_dim=0
        )


def test_cnn_blocks_size():
    for i in range(5):
        cnn_enc = CNNEncoder(
            input_channels=3,
            channels=list(range(8, 8+i))
        )

        assert len(cnn_enc.cnn) == i


def test_cnn_blocks_filters():
    aux = [random.randint(5, 32) for _ in range(10)]

    cnn_enc = CNNEncoder(
        input_channels=aux[0],
        channels=aux[1:]
    )

    for i, l in enumerate(cnn_enc.cnn):
        conv = l[0]
        assert conv.in_channels == aux[i]
        assert conv.out_channels == aux[i+1]


def test_kernel_sizes():
    channels = [random.randint(5, 32) for _ in range(10)]
    kernels = [random.randint(5, 32) for _ in range(10)]

    cnn_enc = CNNEncoder(
        input_channels=3,
        channels=channels,
        kernel_size=5,
        conv_dim=3
    )
    for i in cnn_enc.cnn:
        conv = i[0]
        assert conv.kernel_size == (5,5,5)


def test_kernel_sizes2():
    channels = [random.randint(5, 32) for _ in range(10)]
    kernels = [random.randint(5, 32) for _ in range(10)]

    cnn_enc = CNNEncoder(
        input_channels=3,
        channels=channels,
        kernel_size=kernels,
        conv_dim=3
    )
    for i, b in enumerate(cnn_enc.cnn):
        conv = b[0]
        assert conv.kernel_size == (kernels[i], ) * 3


def test_kernel_sizes3():
    channels = [random.randint(5, 32) for _ in range(10)]
    kernels = [random.randint(5, 32) for _ in range(8)]

    with pytest.raises(ValueError):
        cnn_enc = CNNEncoder(
            input_channels=3,
            channels=channels,
            kernel_size=kernels,
            conv_dim=3
        )


def test_kernel_sizes4():
    channels = [10]
    kernels = [(10, 10)]

    with pytest.raises(ValueError):
        cnn_enc = CNNEncoder(
            input_channels=3,
            channels=channels,
            kernel_size=kernels,
            conv_dim=3
        )

    kernels = [(10, 10, 10, 10)]

    with pytest.raises(ValueError):
        cnn_enc = CNNEncoder(
            input_channels=3,
            channels=channels,
            kernel_size=kernels,
            conv_dim=3
        )


def test_kernel_sizes5():
    channels = [random.randint(5, 32) for _ in range(10)]
    kernels = [(random.randint(5, 32), random.randint(5, 32), random.randint(5, 32)) for _ in range(10)]

    cnn_enc = CNNEncoder(
        input_channels=3,
        channels=channels,
        kernel_size=kernels,
        conv_dim=3
    )

    for i, b in enumerate(cnn_enc.cnn):
        conv = b[0]
        assert conv.kernel_size == kernels[i]


def test_forward_passes():
    channels = [8, 16, 32]
    batch_size = 32

    cnn_enc = CNNEncoder(
        input_channels=3,
        channels=channels,
        kernel_size=3,
        conv_dim=2,
        pooling=torch.nn.MaxPool2d(2),
        stride=1,
        padding=1,
    )

    _in = torch.rand((batch_size, 3, 32, 32))
    out = cnn_enc(_in)

    assert out.shape == torch.Size([batch_size, channels[-1], 4, 4])

    cnn_enc = CNNEncoder(
        input_channels=3,
        channels=channels,
        kernel_size=3,
        conv_dim=2,
        pooling=None,
        stride=1,
        padding=1,
    )

    _in = torch.rand((batch_size, 3, 32, 32))
    out = cnn_enc(_in)

    assert out.shape == torch.Size([batch_size, channels[-1], 32, 32])
