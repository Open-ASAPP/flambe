from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import OrderedDict as odict
from itertools import chain

import torch
import warnings
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import fasttext
from gensim.test.utils import temporary_file

from flambe.compile.registrable import registrable_factory
from flambe.field import Field
from flambe.tokenizer import Tokenizer, WordTokenizer


def get_embeddings(
    embeddings: str,
    embeddings_format: str = 'glove',
    embeddings_binary: bool = False,
) -> KeyedVectors:
    """
    Get the embeddings model and matrix used in the setup function

    Parameters
    ----------
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

    Returns
    -------
    KeyedVectors
        The embeddings object specified by the parameters.
    """
    model = None

    if embeddings_format == 'glove':
        with temporary_file('temp.txt') as temp:
            glove2word2vec(embeddings, temp)
            model = KeyedVectors.load_word2vec_format(temp, binary=embeddings_binary)
    elif embeddings_format == 'word2vec':
        model = KeyedVectors.load_word2vec_format(embeddings,
                                                  binary=embeddings_binary)
    elif embeddings_format == 'fasttext':
        model = fasttext.load_facebook_vectors(embeddings)
    elif embeddings_format == 'gensim':
        try:
            model = KeyedVectors.load(embeddings)
        except FileNotFoundError:
            model = api.load(embeddings)
    else:
        raise ValueError("Only formats supported are word2vec, fasttext and gensim")

    return model


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
                 unk_token: str = '<unk>',
                 sos_token: Optional[str] = None,
                 eos_token: Optional[str] = None,
                 embeddings: Optional[str] = None,
                 embeddings_format: str = 'glove',
                 embeddings_binary: bool = False,
                 model: Optional[KeyedVectors] = None,
                 unk_init_all: bool = False,
                 drop_unknown: bool = False,
                 max_seq_len: Optional[int] = None,
                 truncate_end: bool = False,
                 setup_all_embeddings: bool = False) -> None:
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
        model : KeyedVectors, optional
            The embeddings model used for retrieving text embeddings,
            by default None
        unk_init_all : bool, optional
            If True, every token not provided in the input embeddings is
            given a random embedding from a normal distribution.
            Otherwise, all of them map to the '<unk>' token.
        drop_unknown: bool
            Whether to drop tokens that don't have embeddings
            associated. Defaults to True.
            Important: this flag will only work when using embeddings.
        max_seq_len: int, optional
            The maximum length possibly output by the process func.
            If len of input tokens is larger than this number - then
            the output will be truncated as a post processing step.
        truncate_end: bool
            Determines the window of observed text in process if the
            input is larger than max_seq_len. If this value is True
            the window starts from the end of the utterance.
            Defaults to False.

            example: max_seq_len=3, input_text=1 2 3 4 5
            truncate_end=false: output=1 2 3
            truncate_end=true: output=3 4 5
        setup_all_embeddings: bool
            Controls if all words from the optional provided
            embeddings will be added to the vocabulary and to the
            embedding matrix. Defaults to False.

        """
        if embeddings:
            if model:
                raise ValueError("Cannot submit a model and use the embeddings parameters" +
                                 "simultaneously. Use the 'from_embeddings' factory instead.")

            warnings.warn("The embeddings-exclusive parameters " +
                          "('embeddings', 'embeddings_format', 'embeddings_binary', " +
                          "'setup_all_embeddings', 'drop_unknown', 'unk_init_all') will be " +
                          "deprecated in a future release. " +
                          "Please migrate to use the 'from_embeddings' factory.")

            model = get_embeddings(embeddings, embeddings_format, embeddings_binary)

        if setup_all_embeddings and not model:
            raise ValueError("'setup_all_embeddings' cannot be enabled without passing embeddings.")

        self.tokenizer = tokenizer or WordTokenizer()
        self.lower = lower

        self.pad = pad_token
        self.unk = unk_token
        self.sos = sos_token
        self.eos = eos_token

        self.model = model
        self.embedding_matrix: Optional[torch.Tensor] = None
        self.unk_init_all = unk_init_all
        self.drop_unknown = drop_unknown
        self.setup_all_embeddings = setup_all_embeddings
        self.max_seq_len = max_seq_len
        self.truncate_end = truncate_end

        self.unk_numericals: Set[int] = set()

        self.vocab: Dict = odict()
        specials = [pad_token, unk_token, sos_token, eos_token]
        self.specials = [special for special in specials if special is not None]

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

    def _build_vocab(self, *data: np.ndarray) -> None:
        """
        Build the vocabulary for this object based on the special
        tokens and the data provided.

        This method is safe to be called multiple times.

        Parameters
        ----------
        *data: np.ndarray
            The data

        """
        examples: Iterable = (e for dataset in data for e in dataset if dataset is not None)

        index = len(self.vocab) - 1

        # First load special tokens
        for token in self.specials:
            if token not in self.vocab:
                self.vocab[token] = index = index + 1

        for example in examples:
            # Lowercase if requested
            example = example.lower() if self.lower else example
            # Tokenize and add to vocabulary
            for token in self.tokenizer(example):
                if token not in self.vocab:
                    self.vocab[token] = index = index + 1

    def _build_embeddings(self, model: KeyedVectors) -> Tuple[odict, torch.Tensor]:
        """
        Create the embeddings matrix and the new vocabulary in
        case this objects needs to use an embedding model.

        A new vocabulary needs to be built because of the parameters
        that could allow, for example, collapsing OOVs.

        Parameters
        ----------
        model: KeyedVectors
            The embeddings

        Returns
        -------
        Tuple[OrderedDict, torch.Tensor]
            A tuple with the new embeddings and the embedding matrix
        """
        embedding_matrix: List[torch.Tensor] = []
        new_vocab: odict[str, int] = odict()

        new_index = -1

        tokens: Iterable[str] = self.vocab.keys()

        if self.setup_all_embeddings:
            tokens = chain(tokens, model.vocab.keys())

        for token in tokens:
            if token not in new_vocab:
                if token in model:
                    embedding_matrix.append(torch.tensor(model[token]))
                    new_vocab[token] = new_index = new_index + 1
                elif token in self.specials:
                    embedding_matrix.append(torch.randn(model.vector_size))
                    new_vocab[token] = new_index = new_index + 1
                else:
                    self.unk_numericals.add(self.vocab[token])

                    if self.unk_init_all:
                        embedding_matrix.append(torch.randn(model.vector_size))
                        new_vocab[token] = new_index = new_index + 1
                    else:
                        # Collapse all OOV's to the same <unk> token id
                        new_vocab[token] = new_vocab[self.unk]

        return new_vocab, torch.stack(embedding_matrix)

    def setup(self, *data: np.ndarray) -> None:
        """Build the vocabulary and sets embeddings.

        Parameters
        ----------
        data : Iterable[str]
            List of input strings.

        """
        self._build_vocab(*data)
        if self.model:
            self.vocab, self.embedding_matrix = self._build_embeddings(self.model)

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
                    self.model is not None and numerical in self.unk_numericals:
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

    @registrable_factory
    @classmethod
    def from_embeddings(
        cls,
        embeddings: str,
        embeddings_format: str = 'glove',
        embeddings_binary: bool = False,
        setup_all_embeddings: bool = False,
        unk_init_all: bool = False,
        drop_unknown: bool = False,
        **kwargs,
    ):
        """
        Optional constructor to create TextField from embeddings params.

        Parameters
        ----------
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
        setup_all_embeddings: bool
            Controls if all words from the optional provided
            embeddings will be added to the vocabulary and to the
            embedding matrix. Defaults to False.
        unk_init_all : bool, optional
            If True, every token not provided in the input embeddings is
            given a random embedding from a normal distribution.
            Otherwise, all of them map to the '<unk>' token.
        drop_unknown: bool
            Whether to drop tokens that don't have embeddings
            associated. Defaults to True.
            Important: this flag will only work when using embeddings.

        Returns
        -------
        TextField
            The constructed text field with the requested model.
        """
        model = get_embeddings(
            embeddings,
            embeddings_format,
            embeddings_binary,
        )
        return cls(
            model=model,
            setup_all_embeddings=setup_all_embeddings,
            unk_init_all=unk_init_all,
            drop_unknown=drop_unknown,
            **kwargs,
        )
