from typing import List, Tuple, Optional, Union, Dict, Sequence
from zipfile import ZipFile
from io import BytesIO
import requests

import nltk

from flambe.dataset import TabularDataset
from flambe.field import Field


class PTBDataset(TabularDataset):
    """The official PTB dataset."""

    PTB_URL = "https://raw.githubusercontent.com/yoonkim/lstm-char-cnn/master/data/ptb/"

    def __init__(self,
                 cache: bool = False,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:
        """Initialize the PTBDataset builtin."""
        train_path = self.PTB_URL + "train.txt"
        val_path = self.PTB_URL + "valid.txt"
        test_path = self.PTB_URL + "test.txt"

        train, _ = self._load_file(train_path)
        val, _ = self._load_file(val_path)
        test, _ = self._load_file(test_path)

        super().__init__(train, val, test, cache=cache, transform=transform)

    @classmethod
    def _load_file(cls,
                   path: str,
                   sep: Optional[str] = '\t',
                   header: Optional[str] = None,
                   columns: Optional[Union[List[str], List[int]]] = None,
                   encoding: Optional[str] = 'utf-8') -> Tuple[List[Tuple], Optional[List[str]]]:
        """Load data from the given path."""
        data, named_cols = super()._load_file(path, sep, header, columns)
        return [(d[0][:],) for d in data], named_cols


class Wiki103(TabularDataset):
    """The official WikiText103 dataset."""

    WIKI_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"

    def __init__(self,  # nosec
                 split_by_sentence: bool = False,
                 end_of_line_token: Optional[str] = '</s>',
                 cache: bool = False,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:
        """Initialize the Wiki103 built-in.

        Parameters
        ----------
        split_by_sentence: bool, Optional
            If true, tokenizes per sentence. Default ``False``.
        end_of_line_token: str, Optional
            Token added at the end of every line.

        see TabularDataset for other arguments.

        """
        nltk.download('punkt', quiet=True)

        self.split_by_sentence = split_by_sentence
        self.eol = end_of_line_token
        response = requests.get(self.WIKI_URL, stream=True)
        with ZipFile(BytesIO(response.content), 'r') as z:
            train = self._process(z.read('wikitext-103/wiki.train.tokens'))
            val = self._process(z.read('wikitext-103/wiki.valid.tokens'))
            test = self._process(z.read('wikitext-103/wiki.test.tokens'))

        super().__init__(train, val, test, cache=cache, transform=transform)

    def _process(self, file: bytes) -> List[Tuple[str]]:
        """Process the input file.

        Parameters
        ----------
        file: bytes
            The input file, as a byte string

        Returns
        -------
        List[Tuple[str]]
            List of examples, where each example is a single
            element tuple containing the text.

        """
        decoded_text = file.decode('utf-8')
        # Replace end of line tokens
        if self.eol is not None and not self.split_by_sentence:
            decoded_text = decoded_text.replace('\n', self.eol)

        # Split by sentence or unroll
        if self.split_by_sentence:
            text = [(sent.strip(),) for sent in nltk.tokenize.sent_tokenize(decoded_text)]
        else:
            text = [(decoded_text,)]

        return text


class Enwiki8(TabularDataset):
    """The official WikiText103 dataset."""

    ENWIKI_URL = "http://mattmahoney.net/dc/enwik8.zip"

    def __init__(self,
                 unroll_size: Optional[int] = 100,
                 num_test_symbols: int = 5000000,
                 remove_end_of_line: bool = True,
                 cache: bool = False,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:
        """Initialize the Wiki103 built-in.

        Parameters
        ----------
        split_by_sentence: bool, Optional
            If true, tokenizes per sentence. Default ``False``.
        unroll_size: int, Optional
            Make every sequence of this length. Default ``128``.
            Only used if split_be_sentence is False
        end_of_line_token: str, Optional
            Token added at the end of every line.

        see TabularDataset for other arguments.

        """
        self.unroll_size = unroll_size
        self.num_test_symbols = num_test_symbols
        self.remove_end_of_line = remove_end_of_line
        response = requests.get(self.ENWIKI_URL, stream=True)
        with ZipFile(BytesIO(response.content), 'r') as z:
            train, val, test = self._process(z.read('enwik8'))

        super().__init__(train, val, test, cache=cache, transform=transform)

    def _process(self, file: bytes) -> Sequence[List[Tuple[str]]]:
        """Process the input file.

        Parameters
        ----------
        file: bytes
            The input file, as a byte string

        Returns
        -------
        List[Tuple[str]]
            List of examples, where each example is a single
            element tuple containing the text.

        """
        train_data = file[: -2 * self.num_test_symbols]
        val_data = file[-2 * self.num_test_symbols: -self.num_test_symbols]
        test_data = file[-self.num_test_symbols:]

        symbol = '' if self.remove_end_of_line else str(ord('\n'))
        train = ' '.join([str(c) if c != ord('\n') else symbol for c in train_data])
        val = ' '.join([str(c) if c != ord('\n') else symbol for c in val_data])
        test = ' '.join([str(c) if c != ord('\n') else symbol for c in test_data])

        return [(train,)], [(val,)], [(test,)]
