from typing import Dict, Optional
from collections import OrderedDict as odict

import torch

from flambe.field import Field
from flambe.tokenizer import Tokenizer, NGramsTokenizer


class BoWField(Field):
    """Featurize raw text inputs using bag of words (BoW)

    This class performs tokenization and numericalization.

    The pad, unk, when given, are assigned the first indices in the
    vocabulary, in that order. This means, that whenever a pad token
    is specified, it will always use the 0 index.

    Examples
    --------

    >>> f = BoWField(min_freq=2, normalize=True)
    >>> f.setup(['thank you', 'thank you very much', 'thanks a lot'])
    >>> f._vocab.keys()
    ['thank', you']

    Note that 'thank' and 'you' are the only ones that appear twice.

    >>> f.process("thank you really. You help was awesome")
    tensor([1, 2])

    """

    def __init__(self,  # nosec
                 tokenizer: Optional[Tokenizer] = None,
                 lower: bool = False,
                 unk_token: str = '<unk>',
                 min_freq: int = 5,
                 normalize: bool = False,
                 scale_factor: float = None) -> None:
        """Initialize the BoW object.

        Parameters
        ----------
        tokenizer : Tokenizer, optional
            Tokenizer to use, by default NGramsTokenizer()
        lower : bool, optional
            If given, lowercase the input, by default False
        unk_token : str, optional
            The token to use for out of vocabulary tokens
            (defaults to '<unk>')
        min_freq : int, optional
            Minimum frequency to include token in the vocabulary
            (defaults to 5)
        normalize : bool, optional
            Normalize or not the bag of words using L1 norm
            (defaults to False)
        scale_factor : float, optional
            Factor to scale the resulting normalized feature value.
            Only available when normalize is True (defaults to 1.0)

        """
        self.tokenizer = tokenizer or NGramsTokenizer()
        self.lower = lower
        self.unk = unk_token

        self.min_freq = min_freq
        self.normalize = normalize
        self.scale_factor = scale_factor

        self.vocab: Dict[str, int] = odict()
        self.vocab[unk_token] = 0
        self.full_vocab: Dict[str, int] = {}

        if scale_factor and not normalize:
            raise ValueError("Cannot specify scale_factor without normalizing")

        self.register_attrs('vocab', 'full_vocab')

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary length.

        Returns
        -------
        int
            The length of the vocabulary

        """
        return len(self.vocab)

    def process(self, example):
        # Lowercase and tokenize
        example = example.lower() if self.lower else example
        tokens = self.tokenizer(example)

        # Numericalize
        numericals = [0] * len(self.vocab)
        for token in tokens:
            if token in self.vocab:
                numericals[self.vocab[token]] += 1
            else:
                if token not in self.full_vocab:
                    if self.unk is None or self.unk not in self.vocab:
                        raise ValueError("Encounterd out-of-vocabulary token \
                                          but the unk_token is either missing \
                                          or not defined in the vocabulary.")
                    else:
                        # Accumulate in numericals
                        numericals[self.vocab[self.unk]] += 1  # type: ignore

        processed = torch.tensor(numericals).float()
        if self.normalize:
            processed = torch.nn.functional.normalize(processed, dim=0, p=1)
        if self.scale_factor:
            processed = self.scale_factor * processed

        return processed

    def setup(self, *data) -> None:
        for dataset in data:
            for example in dataset:
                # Lowercase if requested
                example = example.lower() if self.lower else example
                # Tokenize and accumulate in vocabulary
                for token in self.tokenizer(example):
                    self.full_vocab[token] = self.full_vocab.get(token, 0) + 1

        # Filter only the once that have high frequency
        for k, v in self.full_vocab.items():
            if v >= self.min_freq:
                self.vocab.setdefault(k, len(self.vocab))
