from typing import Optional, Sequence, Tuple, Iterator
import math

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from flambe.sampler.sampler import Sampler


class CorpusSampler(Sampler):
    """Implement a CorpusSampler object.

    This object is useful for iteration over a large corpus of
    text in an ordered way. It takes as input a dataset with
    a single example containing the sequence of tokens and will yield
    batches that contain both source sequences of tensors corresponding
    to the Corpus's text, and these same sequences shifted by one as
    the target.

    """

    def __init__(self,
                 batch_size: int = 128,
                 unroll_size: int = 128,
                 n_workers: int = 0,
                 pin_memory: bool = False,
                 downsample: Optional[float] = None,
                 drop_last: bool = True) -> None:
        """Initialize the CorpusSampler object.

        Parameters
        ----------
        batch_size : int, optional
            The batch size to use. Default ``128``.
        unroll_size: int, optional
            Make every sequence this length. Default ``128``.
        n_workers : int, optional
            Number of workers to pass to the DataLoader
            (the default is 0, which means the main process)
        pin_memory : bool, optional
            Pin the memory when using cuda (the default is False)
        downsample: float, optional
            Percentage of the data to downsample to
        drop_last: bool, optional
            Set to True to drop the last incomplete batch if the dataset
            size is not divisible by the batch size.
            (the default is False)

        """
        self.unroll_size = unroll_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.downsample = downsample

    @staticmethod
    def collate_fn(data: Sequence[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Create a batch from data.

        Parameters
        ----------
        data : Sequence[Tuple[Tensor, Tensor]]
            List of (source, target) tuples.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Source and target Tensors.

        """
        x, y = zip(*data)
        return torch.stack(x).t(), torch.stack(y).t()

    def sample(self,
               data: Sequence[Sequence[Tensor]],
               n_epochs: int = 1) -> Iterator[Tuple[Tensor, ...]]:
        """Sample from the list of features and yields batches.

        Parameters
        ----------
        data: Sequence[Sequence[Tensor, ...]]
            The input data to sample from
        n_epochs: int, optional
            The number of epochs to run in the output iterator.
            Use -1 to run infinitely.

        Yields
        ------
        Iterator[Tuple[Tensor]]
            A batch of data, as a tuple of Tensors

        """
        if len(data) == 0:
            raise ValueError("No examples provided")
        elif len(data) > 1:
            raise ValueError("Expected a single input example")

        tensor = data[0][0]  # First example, first column

        if self.downsample:
            if not (0 < self.downsample <= 1):
                raise ValueError("Downsample value should be in the range (0, 1]")
            tensor = tensor[:int(self.downsample * tensor.size(0))]

        # Organize batch-wise
        final_length = (tensor.size(0) - 1) // self.batch_size * self.batch_size
        x = torch.reshape(tensor[:final_length], (self.batch_size, -1)).t()
        y = torch.reshape(tensor[1:final_length + 1], (self.batch_size, -1)).t()

        loader = DataLoader(dataset=torch.utils.data.TensorDataset(x, y),
                            collate_fn=self.collate_fn,
                            shuffle=False,
                            batch_size=self.unroll_size,
                            num_workers=self.n_workers,
                            pin_memory=self.pin_memory,
                            drop_last=self.drop_last)

        if n_epochs == -1:
            while True:
                yield from loader
        else:
            for _ in range(n_epochs):
                yield from loader

    def length(self, data: Sequence[Sequence[torch.Tensor]]) -> int:
        """Return the number of batches in the sampler.

        Parameters
        ----------
        data: Sequence[Sequence[torch.Tensor, ...]]
            The input data to sample from

        Returns
        -------
        int
            The number of batches that would be created per epoch

        """
        tensor = data[0][0]
        if self.drop_last:
            return ((tensor.size(0) - 1) // self.batch_size) // self.unroll_size
        else:
            return math.ceil(((tensor.size(0) - 1) // self.batch_size) / self.unroll_size)
