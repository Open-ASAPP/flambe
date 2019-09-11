from typing import Optional, Sequence
from collections import OrderedDict as odict

import torch
import numpy as np

from flambe.field.field import Field
from flambe.tokenizer import LabelTokenizer


class LabelField(Field):
    """Featurizes input labels.

    The class also handles multilabel inputs and one hot encoding.

    """
    def __init__(self,
                 one_hot: bool = False,
                 multilabel_sep: Optional[str] = None,
                 labels: Optional[Sequence[str]] = None) -> None:
        """Initializes the LabelFetaurizer.

        Parameters
        ----------
        one_hot : bool, optional
            Set for one-hot encoded outputs, defaults to False
        multilabel_sep : str, optional
            If given, splits the input label into multiple labels
            using the given separator, defaults to None.
        labels: Sequence[str], optional
            If given, sets the labels and the ordering is used to map
            the labels to indices. That means the first item in this
            list will have label id 0, the next one id 1, etc..
            When not provided, indices are assigned as labels are
            encountered during preprocessing.

        """
        self.one_hot = one_hot
        self.multilabel_sep = multilabel_sep
        self.tokenizer = LabelTokenizer(multilabel_sep=self.multilabel_sep)

        if labels is not None:
            self.label_given = True
            self.vocab = odict((label, i) for i, label in enumerate(labels))
            self.label_count_dict = {label: 0 for label in self.vocab}
        else:
            self.label_given = False
            self.vocab = odict()
            self.label_count_dict = dict()

        self.register_attrs('vocab')
        self.register_attrs('label_count_dict')

    def setup(self, *data: np.ndarray) -> None:
        """Build the vocabulary.

        Parameters
        ----------
        data : Iterable[str]
            List of input strings.

        """
        # Iterate over all examples
        examples = (e for dataset in data for e in dataset if dataset is not None)

        for example in examples:
            # Tokenize and add to vocabulary
            for token in self.tokenizer(example):
                if self.label_given:
                    if token not in self.vocab:
                        raise ValueError(f"Found label {token} not provided in label list.")
                    else:
                        self.label_count_dict[token] += 1
                else:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
                        self.label_count_dict[token] = 1
                    else:
                        self.label_count_dict[token] += 1

    def process(self, example):
        """Featurize a single example.

        Parameters
        ----------
        example: str
            The input label

        Returns
        -------
        torch.Tensor
            A list of integer tokens

        """
        tokens = self.tokenizer(example)

        # Numericalize
        numericals = []
        for token in tokens:
            if token not in self.vocab:
                raise ValueError("Encounterd out-of-vocabulary label {token}")

            numerical = self.vocab[token]  # type: ignore
            numericals.append(numerical)

        out = torch.tensor(numericals).long()

        if self.one_hot:
            out = [int(i in out) for i in range(len(self.vocab))]
            out = torch.tensor(out).long()  # Back to Tensor

        return out.squeeze()

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary length.

        Returns
        -------
        int
            The length of the vocabulary

        """
        return len(self.vocab)

    @property
    def label_count(self) -> torch.Tensor:
        """Get the label count.

        Returns
        -------
        torch.Tensor
            Tensor containing the count for each label, indexed
            by the id of the label in the vocabulary.
        """
        counts = [self.label_count_dict[label] for label in self.vocab]
        return torch.tensor(counts).float()

    @property
    def label_freq(self) -> torch.Tensor:
        """Get the frequency of each label.

        Returns
        -------
        torch.Tensor
            Tensor containing the frequency of each label, indexed
            by the id of the label in the vocabulary.

        """
        counts = [self.label_count_dict[label] for label in self.vocab]
        return torch.tensor(counts).float() / sum(counts)

    @property
    def label_inv_freq(self) -> torch.Tensor:
        """Get the inverse frequency for each label.

        Returns
        -------
        torch.Tensor
            Tensor containing the inverse frequency of each label,
            indexed by the id of the label in the vocabulary.

        """
        return 1. / self.label_freq  # type: ignore
