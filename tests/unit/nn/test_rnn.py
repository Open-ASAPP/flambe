import pytest
from flambe.nn import RNNEncoder
import torch
import mock


@pytest.mark.parametrize("rnn_type", ['LSTM', 'random', ''])
def test_invalid_type(rnn_type):
    with pytest.raises(ValueError):
        RNNEncoder(
            input_size=10,
            hidden_size=20,
            rnn_type=rnn_type
        )


def test_sru_kwargs():
    rnn = RNNEncoder(
        input_size=10,
        hidden_size=20,
        rnn_type='sru',
        use_tanh=True
    )

    for i in rnn.rnn.rnn_lst:
        assert i.activation == 'tanh'


def test_invalid_sru_kwargs():
    with pytest.raises(ValueError):
        _ = RNNEncoder(
            input_size=10,
            hidden_size=20,
            rnn_type='sru',
            use_tanh=True,
            some_invalid_param=123
        )


@pytest.mark.parametrize("rnn_type", ['lstm', 'gru', 'sru'])
def test_forward_pass(rnn_type):
    input_size = 300
    output_size = 10
    seq_len = 20
    batch_len = 32
    rnn = RNNEncoder(
        input_size=input_size,
        hidden_size=output_size,
        n_layers=4,
        rnn_type=rnn_type)

    input_t = torch.rand(batch_len, seq_len, input_size)

    output, state = rnn(input_t)
    assert output.shape == torch.Size((batch_len, seq_len, output_size))


@pytest.mark.parametrize("rnn_type", ['lstm', 'gru', 'sru'])
def test_transpose_on_forward_pass(rnn_type):
    input_size = 300
    output_size = 10
    rnn = RNNEncoder(
        input_size=input_size,
        hidden_size=output_size,
        n_layers=4,
        rnn_type=rnn_type)

    input_t = torch.rand(10, 10, input_size)

    input_t.transpose = mock.Mock(side_effect=input_t.transpose)
    output, state = rnn(input_t)

    input_t.transpose.assert_called()
    input_t.transpose.assert_called_with(0, 1)
