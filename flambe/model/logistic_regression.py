# type: ignore[override]

from typing import Optional, Tuple, Union

from torch import Tensor
from torch.nn import Sigmoid
from flambe.nn.module import Module  # type: ignore[attr-define]
from flambe.nn import MLPEncoder


class LogisticRegression(Module):
    """
    Logistic regression model given an input vector v
    the forward calculation is sigmoid(Wv+b), where
    W is a weight vector and b a bias term. The result
    is then passed to a sigmoid function, which maps it
    as a real number in [0,1]. This is typically interpreted
    in classification settings as the probability of belonging
    to a given class.

    Attributes
    ----------
    input_size : int
        Dimension (number of features) of the input vector.
    """

    def __init__(self, input_size: int) -> None:
        """
        Initialize the Logistic Regression Model.
        Parameters
        ----------
        input_size: int
            The dimension of the input vector
        """
        super().__init__()
        self.encoder = MLPEncoder(input_size, output_size=1,
                                  n_layers=1, output_activation=Sigmoid())

    def forward(self,
                data: Tensor,
                target: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass that encodes data
        Parameters
        ----------
        data : Tensor
            input data to encode
        target: Optional[Tensor]
            target value, will be casted to a float tensor.
        """
        encoding = self.encoder(data)
        return (encoding, target.float()) if target is not None else encoding
