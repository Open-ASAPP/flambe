from flambe.nn import MLPEncoder
import torch


def test_forward_pass_1_layer():
    input_size = 128
    output_size = 64
    batch_size = 32

    # no activation
    mlp_enc = MLPEncoder(input_size, output_size)
    _in = torch.rand((batch_size, input_size))
    out = mlp_enc(_in)

    assert out.shape == torch.Size((batch_size, output_size))

    # with activation
    mlp_enc = MLPEncoder(
        input_size,
        output_size,
        output_activation=torch.nn.ReLU(),
    )
    _in = torch.rand((batch_size, input_size))
    out = mlp_enc(_in)

    assert out.shape == torch.Size((batch_size, output_size))


def test_forward_pass_multi_layers():
    input_size = 256
    hidden_size = 128
    output_size = 64
    batch_size = 32

    # no activation
    mlp_enc = MLPEncoder(
        input_size,
        output_size,
        n_layers=3,
        hidden_size=hidden_size,
    )
    _in = torch.rand((batch_size, input_size))
    out = mlp_enc(_in)

    assert out.shape == torch.Size((batch_size, output_size))

    # with activation
    mlp_enc = MLPEncoder(
        input_size,
        output_size,
        n_layers=3,
        output_activation=torch.nn.ReLU(),
        hidden_size=hidden_size,
        hidden_activation=torch.nn.ReLU(),
    )
    _in = torch.rand((batch_size, input_size))
    out = mlp_enc(_in)

    assert out.shape == torch.Size((batch_size, output_size))
