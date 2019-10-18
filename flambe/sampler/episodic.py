from typing import Tuple, Iterator, Dict, Any, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from flambe.sampler import Sampler


class EpisodicSampler(Sampler):
    """Implement an EpisodicSample object.

    Currently only supports sequence inputs.

    """

    def __init__(self,
                 n_support: int,
                 n_query: int,
                 n_episodes: int,
                 n_classes: int = None,
                 pad_index: int = 0,
                 balance_query: bool = False) -> None:
        """Initialize the EpisodicSampler.

        Parameters
        ----------
        n_support : int
            The number of support points per class
        n_query : int
            If balance_query is True, this should be the number
            of query points per class, otherwise, this is the total
            number of query points for the episode
        n_episodes : int
            Number of episodes to run in one "epoch"
        n_classes : int, optional
            The number of classes to sample per episode, defaults to all
        pad_index : int, optional
            The padding index used on sequences.
        balance_query : bool, optional
            If True, the same number of query points are sampled per
            class, otherwise query points are sampled uniformly
            from the input data.

        """
        self.pad = pad_index

        self.n_support = n_support
        self.n_query = n_query
        self.n_classes = n_classes
        self.n_episodes = n_episodes

        self.balance_query = balance_query

    def sample(self,
               data: Sequence[Sequence[torch.Tensor]],
               n_epochs: int = 1) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Sample from the list of features and yields batches.

        Parameters
        ----------
        data: Sequence[Sequence[torch.Tensor, torch.Tensor]]
            The input data as a list of (source, target) pairs
        n_epochs: int, optional
            The number of epochs to run in the output iterator.
            For this object, the total number of batches will be
            (n_episodes * n_epochs)

        Yields
        ------
        Iterator[Tuple[Tensor, Tensor, Tensor, Tensor]]
            In order: the query_source, the query_target
            the support_source, and the support_target tensors.
            For sequences, the batch is used as first dimension.

        """
        if len(data) == 0:
            raise ValueError("No examples provided")

        # Split dataset by target
        target_to_examples: Dict[int, Any] = dict()
        for source, target in data:
            target_to_examples.setdefault(int(target), []).append((source, target))

        all_classes = list(target_to_examples.keys())

        for epoch in range(n_epochs):
            for _ in range(self.n_episodes):
                # Sample n_classes to run a training episode over
                classes = all_classes
                if self.n_classes is not None:
                    classes = list(np.random.permutation(all_classes))[:self.n_classes]

                # Sample n_support and n_query points per class
                supports, queries = [], []
                for i, target_class in enumerate(classes):
                    examples = target_to_examples[target_class]
                    indices = np.random.permutation(len(examples))
                    supports.extend([(examples[j][0], i) for j in indices[:self.n_support]])

                    if self.balance_query:
                        query_indices = indices[self.n_support:self.n_support + self.n_query]
                        queries.extend([(examples[j][0], i) for j in query_indices])
                    else:
                        queries.extend([(examples[j][0], i) for j in indices[self.n_support:]])

                if not self.balance_query:
                    indices = np.random.permutation(len(queries))
                    queries = [queries[i] for i in indices[:self.n_query]]

                query_source, query_target = list(zip(*queries))
                support_source, support_target = list(zip(*supports))

                query_source = pad_sequence(query_source,
                                            batch_first=True,
                                            padding_value=self.pad)
                query_target = torch.tensor(query_target)

                support_source = pad_sequence(support_source,
                                              batch_first=True,
                                              padding_value=self.pad)
                support_target = torch.tensor(support_target)

                if len(query_target.size()) == 2:
                    query_target = query_target.squeeze()
                if len(support_target.size()) == 2:
                    support_target = support_target.squeeze()

                yield (query_source.long(),
                       query_target.long(),
                       support_source.long(),
                       support_target.long())

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
        return self.n_episodes
