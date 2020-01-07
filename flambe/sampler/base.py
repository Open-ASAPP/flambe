import math
from collections import defaultdict, OrderedDict as odict
from itertools import chain
from functools import partial
from typing import Iterator, Tuple, Union, Sequence, List, Dict, Set, Optional

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np

from flambe.sampler.sampler import Sampler


def _bfs(obs: List, obs_idx: int) -> Tuple[Dict[int, List], Set[Tuple[int, ...]]]:
    """
    Given a single `obs`, itself a nested list, run BFS.

    This function enumerates:

        1. The lengths of each of the intermediary lists, by depth
        2. All paths to the child nodes

    Parameters
    ----------
    obs : List
        A nested list of lists of arbitrary depth, with the child nodes,
        i.e. deepest list elements, as `torch.Tensor`s

    obs_idx : int
        The index of `obs` in the batch.

    Returns
    -------
    Set[Tuple[int]]
        A set of all distinct paths to all children

    Dict[int, List[int]]
        A map containing the lengths of all intermediary lists, by depth

    """
    path, level, root = [obs_idx], 0, tuple(obs)
    queue = [(path, level, root)]
    paths = set()
    lens: Dict[int, List] = defaultdict(list)
    while queue:
        path, level, item = queue.pop(0)
        lens[level].append(len(item))
        for i, c in enumerate(item):
            if c.dim() == 0:  # We're iterating through child tensor itself
                paths.add(tuple(path))
            else:
                queue.append((path + [i], level + 1, c))
    return lens, paths


def _batch_from_nested_col(col: Tuple, pad: int) -> torch.Tensor:
    """Compose a batch padded to the max-size along each dimension.

    Parameters
    ----------
    col : List
        A nested list of lists of arbitrary depth, with the child nodes,
        i.e. deepest list elements, as `torch.Tensor`s

        For example, a `col` might be:

        [
            [torch.Tensor([1, 2]), torch.Tensor([3, 4, 5])],
            [torch.Tensor([5, 6, 7]), torch.Tensor([4, 5]),
             torch.Tensor([5, 6, 7, 8])]
        ]

        Level 1 sizes: [2, 3]
        Level 2 sizes: [2, 3]; [3, 2, 4]

        The max-sizes along each dimension are:

            * Dim 1: 3
            * Dim 2: 4

        As such, since this column contains 2 elements, with max-sizes
        3 and 4 along the nested dimensions, our resulting batch would
        have size (4, 3, 2), and the padded `Tensor`s would be inserted
        at their respective locations.

    Returns
    -------
    torch.Tensor
        A (n+1)-dimensional torch.Tensor, where n is the nesting
        depth, padded to the max-size along each dimension

    """
    bs = len(col)

    # Compute lengths of child nodes, and the path to reach them
    lens, paths = zip(*[_bfs(obs, obs_idx=i) for i, obs in enumerate(col)])

    # Compute the max length for each level
    lvl_to_lens: Dict[int, List] = defaultdict(list)
    for l in lens:
        for lvl, lns in l.items():
            lvl_to_lens[lvl].extend(lns)
    max_lens = odict([(lvl, max(lvl_to_lens[lvl])) for lvl in sorted(lvl_to_lens.keys())])

    # Instantiate the empty batch
    batch = torch.zeros(bs, *max_lens.values()).long() + pad

    # Populate the batch with each child node
    for p in chain.from_iterable(paths):
        el = col
        for i in p:
            el = el[i]
        diff = batch.size(-1) - len(el)
        pad_tens = torch.zeros(diff).long() + pad
        # TODO fix two typing errors below; likely because of typing
        # on el which is reused multiple times
        el = torch.cat((el, pad_tens))  # type: ignore
        batch.index_put_(indices=[torch.tensor([i]) for i in p], values=el)  # type: ignore

    return batch


def collate_fn(data: List[Tuple[torch.Tensor, ...]], pad: int) -> Tuple[torch.Tensor, ...]:
    """Turn a list of examples into a mini-batch.

    Handles padding on the fly on simple sequences, as well as
    nested sequences.

    Parameters
    ----------
    data : List[Tuple[torch.Tensor, ...]]
        The list of sampled examples.
        Each example is a tuple, each dimension representing a
        column from the original dataset
    pad: int
        The padding index

    Returns
    -------
    Tuple[torch.Tensor, ...]
        The output batch of tensors

    """
    columns = list(zip(*data))

    # Establish col-specific pad tokens
    if isinstance(pad, (tuple, list)):
        pad_tkns = pad
        if len(pad_tkns) != len(columns):
            raise Exception(f"The number of column-specific pad tokens \
                                ({len(pad_tkns)}) does not equal the number \
                                of columns in the batch ({len(columns)})")
    else:
        pad_tkns = tuple([pad] * len(columns))

    batch = []
    for pad, column in zip(pad_tkns, columns):
        # Prepare the tensors
        is_nested = any([isinstance(example, (list, tuple)) for example in column])
        if is_nested:
            # Column contains nested observations
            nested_tensors = _batch_from_nested_col(column, pad)
            batch.append(nested_tensors)
        else:
            tensors = [torch.tensor(example) for example in column]
            sizes = [tensor.size() for tensor in tensors]

            if all(s == sizes[0] for s in sizes):
                stacked_tensors = torch.stack(tensors)
                batch.append(stacked_tensors)
            else:
                # Variable length sequences
                padded_tensors = pad_sequence(tensors,
                                              batch_first=True,
                                              padding_value=pad)
                batch.append(padded_tensors)

    return tuple(batch)


class BaseSampler(Sampler):
    """Implements a BaseSampler object.

    This is the most basic implementation of a sampler.
    It uses Pytorch's DataLoader object internally, and
    offers the possiblity to override the sampling of the
    examples and how to from a batch from them.

    """

    def __init__(self,
                 batch_size: int = 64,
                 shuffle: bool = True,
                 pad_index: Union[int, Sequence[int]] = 0,
                 n_workers: int = 0,
                 pin_memory: bool = False,
                 seed: Optional[int] = None,
                 downsample: Optional[float] = None,
                 downsample_seed: Optional[int] = None,
                 drop_last: bool = False) -> None:
        """Initialize the BaseSampler object.

        Parameters
        ----------
        batch_size : int
            The batch size to use
        shuffle : bool, optional
            Whether the data should be shuffled every epoch
            (the default is True)
        pad_index : int, optional
            The index used for padding (the default is 0). Can be a
            single pad_index applied to all columns, or a list or tuple
            of pad_index's that apply to each column respectively.
            (In this case, this list or tuple must have length equal
            to the number of columns in the batch.)
        n_workers : int, optional
            Number of workers to pass to the DataLoader
            (the default is 0, which means the main process)
        pin_memory : bool, optional
            Pin the memory when using cuda (the default is False)
        seed: int, optional
            Optional seed for the sampler
        downsample: float, optional
            Percentage of the data to downsample to
        downsample_seed: int, optional
            The seed to use in downsampling
        drop_last: bool, optional
            Set to True to drop the last incomplete batch if the dataset
            size is not divisible by the batch size.
            (the default is False)

        """
        self.pad = pad_index
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.downsample = downsample
        self.downsample_seed = downsample_seed
        self.random_generator = np.random if seed is None else np.random.RandomState(seed)

    def sample(self,
               data: Sequence[Sequence[torch.Tensor]],
               n_epochs: int = 1) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Sample from the list of features and yields batches.

        Parameters
        ----------
        data: Sequence[Sequence[torch.Tensor, ...]]
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

        if self.downsample:
            if not (0 < self.downsample <= 1):
                raise ValueError("Downsample value should be in the range (0, 1]")
            if self.downsample_seed:
                downsample_generator = np.random.RandomState(self.downsample_seed)
            else:
                downsample_generator = np.random
            random_indices = downsample_generator.permutation(len(data))
            data = [data[i] for i in random_indices[:int(self.downsample * len(data))]]

        collate_fn_p = partial(collate_fn, pad=self.pad)
        # TODO investigate dataset typing in PyTorch; sequence should
        # be fine
        loader = DataLoader(dataset=data,  # type: ignore
                            shuffle=self.shuffle,
                            batch_size=self.batch_size,
                            collate_fn=collate_fn_p,
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
        downsample = self.downsample or 1
        return math.ceil(downsample * len(data) / self.batch_size)
