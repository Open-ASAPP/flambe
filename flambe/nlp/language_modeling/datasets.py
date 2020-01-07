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

    def __init__(self,  # nosec
                 split_by_sentence: bool = False,
                 end_of_line_token: Optional[str] = '<eol>',  # nosec
                 cache: bool = False,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:
        """Initialize the PTBDataset builtin.

        Parameters
        ----------
        split_by_sentence: bool, Optional
            If true, tokenizes per sentence. Default ``False``.
        end_of_line_token: str, Optional
            Token added at the end of every line.

        see TabularDataset for other arguments.

        """
        self.split_by_sentence = split_by_sentence
        self.eol = end_of_line_token

        train_path = self.PTB_URL + "train.txt"
        val_path = self.PTB_URL + "valid.txt"
        test_path = self.PTB_URL + "test.txt"

        train = self._process(requests.get(train_path).content)
        val = self._process(requests.get(val_path).content)
        test = self._process(requests.get(test_path).content)

        super().__init__(train, val, test, cache=cache, transform=transform)

    def _process(self, file: bytes) -> List[Tuple[str]]:
        """Process the input file.

        Parameters
        ----------
        field: str
            The input file, as bytes

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
            nltk.download('punkt', quiet=True)
            text = [(sent.strip(),) for sent in nltk.tokenize.sent_tokenize(decoded_text)]
        else:
            text = [(decoded_text,)]

        return text


class Wiki103(TabularDataset):
    """The official WikiText103 dataset."""

    WIKI_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"

    def __init__(self,  # nosec
                 split_by_line: bool = False,
                 end_of_line_token: Optional[str] = '<eol>',  # nosec
                 remove_headers: bool = False,
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
        self.split_by_line = split_by_line
        self.eol = end_of_line_token
        self.remove_headers = remove_headers
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
        decoded_lines = decoded_text.split('\n')

        # Remove titles of Wikipedia articles if desired
        if self.remove_headers:
            filtered_lines = []
            for line in decoded_lines:
                line_strip = line.strip()
                if len(line_strip) > 0:
                    if line_strip[0] != '=' and line_strip[-1] != '=':
                        filtered_lines.append(line)
            decoded_lines = filtered_lines

        eol = self.eol or ''
        if self.split_by_line:
            text = [(line.lstrip() + eol,) for line in decoded_lines]
        else:
            text = [(eol.join(decoded_lines),)]

        return text


class Enwiki8(TabularDataset):
    """The official WikiText103 dataset."""

    ENWIKI_URL = "http://mattmahoney.net/dc/enwik8.zip"

    def __init__(self,
                 num_eval_symbols: int = 5000000,
                 remove_end_of_line: bool = False,
                 cache: bool = False,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:
        """Initialize the Wiki103 built-in.

        Parameters
        ----------.
        num_eval_symbols: int, optional
            The number of symbols to use for seach of validation,
            and testing. Default ``5000000``.
        remove_end_of_line: bool, optional
            If True, remove end of line tokens. Default ``True``.

        see TabularDataset for other arguments.

        """
        self.num_eval_symbols = num_eval_symbols
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
        train_data = file[: -2 * self.num_eval_symbols]
        val_data = file[-2 * self.num_eval_symbols: -self.num_eval_symbols]
        test_data = file[-self.num_eval_symbols:]

        symbol = '' if self.remove_end_of_line else str(ord('\n'))
        train = ' '.join([str(c) if c != ord('\n') else symbol for c in train_data])
        val = ' '.join([str(c) if c != ord('\n') else symbol for c in val_data])
        test = ' '.join([str(c) if c != ord('\n') else symbol for c in test_data])

        return [(train,)], [(val,)], [(test,)]
