from abc import abstractmethod
from typing import Any, Union, Tuple

import torch
import numpy as np

from flambe import Component


class Field(Component):
    """Base Field interface.

    A field processes raw examples and produces Tensors.

    """
    def setup(self, *data: np.ndarray) -> None:
        """Setup the field.

        This method will be called with all the data in the dataset and
        it can be used to compute aggregated information (for example,
        vocabulary in Fields that process text).

        ATTENTION: this method could be called multiple times in case
        the same field is used in different datasets. Take this into
        account and build a stateful implementation.

        Parameters
        ----------
        *data: np.ndarray
            Multiple 2d arrays (ex: train_data, dev_data, test_data).
            First dimension is for the examples, second dimension for
            the columns specified for this specific field.

        """
        pass

    @abstractmethod
    def process(self, *example: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Process an example into a Tensor or tuple of Tensor.

        This method allows N to M mappings from example columns (N)
        to tensors (M).

        Parameters
        ----------
        *example: Any
            Column values of the example

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            The processed example, as a tensor or tuple of tensors
        """
        pass
