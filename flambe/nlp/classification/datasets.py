from typing import List, Tuple, Optional, Dict, Union

from flambe.dataset import TabularDataset
from flambe.field import Field


class SSTDataset(TabularDataset):
    """The official SST-1 dataset."""

    URL = "https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/"

    def __init__(self,
                 binary: bool = True,
                 phrases: bool = False,
                 cache: bool = True,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:
        """Initialize the SSTDataset builtin.

        Parameters
        ----------
        binary: bool
            Set to true to train and evaluate in binary mode.
            Defaults to True.
        phrases: bool
            Set to true to train on phrases. Defaults to False.

        """
        binary_str = 'binary' if binary else 'fine'
        phrases_str = '.phrases' if phrases else ''

        train_path = self.URL + f"stsa.{binary_str}{phrases_str}.train"
        dev_path = self.URL + f"stsa.{binary_str}.dev"
        test_path = self.URL + f"stsa.{binary_str}.test"

        train, _ = self._load_file(train_path, sep='\t', header=None)
        val, _ = self._load_file(dev_path, sep='\t', header=None)
        test, _ = self._load_file(test_path, sep='\t', header=None)

        named_cols = ['text', 'label']
        super().__init__(train, val, test, cache, named_cols, transform)

    @classmethod
    def _load_file(cls,
                   path: str,
                   sep: Optional[str] = '\t',
                   header: Optional[str] = None,
                   columns: Optional[Union[List[str], List[int]]] = None,
                   encoding: Optional[str] = 'utf-8') -> Tuple[List[Tuple], Optional[List[str]]]:
        """Load data from the given path."""
        data, named_cols = super()._load_file(path, sep, header, columns)
        return [("".join(d[0][2:]), d[0][0]) for d in data], named_cols


class TRECDataset(TabularDataset):
    """The official TREC dataset."""

    URL = "https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/"

    def __init__(self, cache: bool = True,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:
        """Initialize the SSTDataset builtin."""
        train_path = self.URL + "TREC.train.all"
        test_path = self.URL + "TREC.test.all"

        train, _ = self._load_file(train_path, sep='\t', header=None, encoding='latin-1')
        test, _ = self._load_file(test_path, sep='\t', header=None, encoding='latin-1')

        named_cols = ['text', 'label']
        super().__init__(
            train=train,
            val=None,
            test=test,
            cache=cache,
            named_columns=named_cols,
            transform=transform
        )

    @classmethod
    def _load_file(cls,
                   path: str,
                   sep: Optional[str] = '\t',
                   header: Optional[str] = None,
                   columns: Optional[Union[List[str], List[int]]] = None,
                   encoding: Optional[str] = 'latin-1') -> Tuple[List[Tuple], Optional[List[str]]]:
        """Load data from the given path."""
        data, named_cols = super()._load_file(path, sep, header, columns, encoding)
        return [("".join(d[0][2:]), d[0][0]) for d in data], named_cols


class NewsGroupDataset(TabularDataset):
    """The official 20 news group dataset."""

    def __init__(self,
                 cache: bool = False,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:
        """Initialize the NewsGroupDataset builtin."""
        try:
            from sklearn.datasets import fetch_20newsgroups
        except ImportError:
            raise ImportError("Install sklearn to use the NewsGroupDataset")

        train = fetch_20newsgroups(subset='train')
        test = fetch_20newsgroups(subset='test')

        train = [(' '.join(d.split()), str(t)) for d, t in zip(train['data'], train['target'])]
        test = [(' '.join(d.split()), str(t)) for d, t in zip(test['data'], test['target'])]

        named_cols = ['text', 'label']
        super().__init__(
            train=train,
            val=None,
            test=test,
            cache=cache,
            named_columns=named_cols,
            transform=transform
        )
