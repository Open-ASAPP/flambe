from typing import Optional, Dict, Set
from collections import OrderedDict as odict

import torch
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import fasttext
from gensim.test.utils import temporary_file

from flambe.field import Field
from flambe.tokenizer import Tokenizer, WordTokenizer


class TextField(Field):
    """Featurize raw text inputs

    This class performs tokenization and numericalization, as well as
    decorating the input sequences with optional start and end tokens.

    When a vocabulary is passed during initialiazation, it is used to
    map the the words to indices. However, the vocabulary can also be
    generated from input data, through the `setup` method. Once
    a vocabulary has been built, this object can also be used to load
    external pretrained embeddings.

    The pad, unk, sos and eos tokens, when given, are assigned the
    first indices in the vocabulary, in that order. This means, that
    whenever a pad token is specified, it will always use the 0 index.

    """

    def __init__(self,  # nosec
                 tokenizer: Optional[Tokenizer] = None,
                 lower: bool = False,
                 pad_token: Optional[str] = '<pad>',
                 unk_token: Optional[str] = '<unk>',
                 sos_token: Optional[str] = None,
                 eos_token: Optional[str] = None,
                 embeddings: Optional[str] = None,
                 embeddings_format: str = 'glove',
                 embeddings_binary: bool = False,
                 unk_init_all: bool = False,
                 drop_unknown: bool = False,
                 max_seq_len: Optional[int] = None,
                 truncate_end: bool = False) -> None:
        """Initialize the TextField.

        Parameters
        ----------
        tokenizer : Tokenizer, optional
            Tokenizer to use, by default WordTokenizer()
        lower : bool, optional
            If given, lowercase the input, by default False
        pad_token : str, optional
            Reserved padding token. Note that this object does not
            perform padding. Padding is done on the fly, when sampling.
            (defaults to '<pad>')
        unk_token : str, optional
            The token to use for out of vocabulary tokens
            (defaults to '<unk>')
        sos_token : str, optional
            Start of sentence tokens to add to the start of
            each sequence (defaults to '<sos>')
        eos : Iterable[str], optional
            List of end of sentence tokens to add to the end of each
            sequence (defaults to an empty list)
        embeddings : Optional[str], optional
            Path to pretrained embeddings, by default None
        embeddings_format : str, optional
            The format of the input embeddings, should be one of:
            'glove', 'word2vec', 'fasttext' or 'gensim'. The latter can
            be used to download embeddings hosted on gensim on the fly.
            See https://github.com/RaRe-Technologies/gensim-data
            for the list of available embedding aliases.
        embeddings_binary : bool, optional
            Whether the input embeddings are provided in binary format,
            by default False
        unk_init_all : bool, optional
            If True, every token not provided in the input embeddings is
            given a random embedding from a normal distribution.
            Otherwise, all of them map to the '<unk>' token.
        drop_unknown: bool
            Whether to drop tokens that don't have embeddings
            associated. Defaults to True.
            Important: this flag will only work when using embeddings.

        """
        self.tokenizer = tokenizer or WordTokenizer()
        self.lower = lower

        self.pad = pad_token
        self.unk = unk_token
        self.sos = sos_token
        self.eos = eos_token

        self.embeddings = embeddings
        self.embeddings_format = embeddings_format
        self.embeddings_binary = embeddings_binary
        self.embedding_matrix: Optional[torch.Tensor] = None
        self.unk_init_all = unk_init_all
        self.drop_unknown = drop_unknown
        self.max_seq_len = max_seq_len
        self.truncate_end = truncate_end

        self.unk_numericals: Set[int] = set()

        self.vocab: Dict = odict()
        specials = [pad_token, unk_token, sos_token, eos_token]
        self.specials = [special for special in specials if special is not None]

        index = -1
        for token in self.specials:
            self.vocab[token] = index = index + 1

        self.register_attrs('vocab')

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary length.

        Returns
        -------
        int
            The length of the vocabulary

        """
        unique_ids = set(v for k, v in self.vocab.items())
        return len(unique_ids)

    def setup(self, *data: np.ndarray) -> None:
        """Build the vocabulary and sets embeddings.

        Parameters
        ----------
        data : Iterable[str]
            List of input strings.

        """
        if self.embeddings is not None:
            # Load embedding model
            embeddings_matrix = []
            if self.embeddings_format == 'glove':
                with temporary_file('temp.txt') as temp:
                    glove2word2vec(self.embeddings, temp)
                    model = KeyedVectors.load_word2vec_format(temp, binary=self.embeddings_binary)
            elif self.embeddings_format == 'word2vec':
                model = KeyedVectors.load_word2vec_format(self.embeddings,
                                                          binary=self.embeddings_binary)
            elif self.embeddings_format == 'fasttext':
                model = fasttext.load_facebook_vectors(self.embeddings)
            elif self.embeddings_format == 'gensim':
                try:
                    model = KeyedVectors.load(self.embeddings)
                except FileNotFoundError:
                    model = api.load(self.embeddings)
            else:
                raise ValueError("Only formats supported are word2vec, fasttext and gensim")

            # Add embeddings for special tokens
            for special in self.specials:
                if special in model:
                    embeddings_matrix.append(torch.tensor(model[special]))
                else:
                    embeddings_matrix.append(torch.randn(model.vector_size))

        # Iterate over all examples
        examples = (e for dataset in data for e in dataset if dataset is not None)

        # Get current last id
        index = len(self.vocab) - 1

        for example in examples:
            # Lowercase if requested
            example = example.lower() if self.lower else example
            # Tokenize and add to vocabulary
            for token in self.tokenizer(example):
                if token not in self.vocab:
                    if self.embeddings is not None:
                        if token in model:
                            self.vocab[token] = index = index + 1
                            embeddings_matrix.append(torch.tensor(model[token]))
                        else:
                            if self.unk_init_all:
                                # Give every OOV it's own embedding
                                self.vocab[token] = index = index + 1
                                embeddings_matrix.append(torch.randn(model.vector_size))
                            else:
                                # Collapse all OOV's to the same token
                                # id
                                self.vocab[token] = self.vocab[self.unk]
                            self.unk_numericals.add(self.vocab[token])
                    else:
                        self.vocab[token] = index = index + 1

        if self.embeddings is not None:
            self.embedding_matrix = torch.stack(embeddings_matrix)

    # TODO update when we add generics
    def process(self, example: str) -> torch.Tensor:  # type: ignore
        """Process an example, and create a Tensor.

        Parameters
        ----------
        example: str
            The example to process, as a single string

        Returns
        -------
        torch.Tensor
            The processed example, tokenized and numericalized

        """
        # Lowercase and tokenize
        example = example.lower() if self.lower else example
        tokens = self.tokenizer(example)

        # Add extra tokens
        if self.sos is not None:
            tokens = [self.sos] + list(tokens)
        if self.eos is not None:
            tokens = list(tokens) + [self.eos]

        # Numericalize
        numericals = []
        for token in tokens:
            if token not in self.vocab:
                if self.unk is None or self.unk not in self.vocab:
                    raise ValueError("Encounterd out-of-vocabulary token \
                                      but the unk_token is either missing \
                                      or not defined in the vocabulary.")
                else:
                    token = self.unk

            numerical = self.vocab[token]  # type: ignore

            if self.drop_unknown and \
                    self.embeddings is not None and numerical in self.unk_numericals:
                # Don't add unknown tokens in case the flag is activated
                continue

            numericals.append(numerical)

        ret = torch.tensor(numericals).long()

        if self.max_seq_len is not None:
            if self.truncate_end:
                ret = ret[-self.max_seq_len:]
            else:
                ret = ret[:self.max_seq_len]
        return ret
