import os
from typing import Optional, List, Tuple, Iterable, Dict, Union, Any

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from flambe.dataset import Dataset
from flambe.compile import registrable_factory
from flambe.field import Field


class DataView:
    """TabularDataset view for the train, val or test split. This class
    must be used only internally in the TabularDataset class.

    A DataView is a lazy Iterable that receives the operations
    from the TabularDataset object. When __getitem__ is called, then
    all the fields defined in the transform are applied.

    This object can cache examples already transformed.
    To enable this, make sure to use this view under a Singleton pattern
    (there must only be one DataView per split in the TabularDataset).

    """
    def __init__(self,
                 data: np.ndarray,
                 transform_hooks: List[Tuple[Field, Union[int, List[int]]]],
                 cache: bool) -> None:
        """
        Parameters
        ----------
        data: np.ndarray
            A 2d numpy array holding the data
        transform_hooks: List[Tuple[Field, Union[int, List[int]]]]
            The transformations that will be applied to each example.
        cache: bool
            To apply cache or not.

        """
        self.data = data  # Stores the raw data
        self.transform_hooks = transform_hooks
        self.cache = cache

        # Caches the transformed data
        self.cached_data: Dict[int, Any] = {}

    @property
    def raw(self):
        """Returns an subscriptable version of the data"""
        return self.data

    def __getitem__(self, index):
        """
        Get an item from an index and apply the transformations
        dinamically.

        """
        if self.data is None:
            raise IndexError()

        if self.cache and index in self.cached_data:
            return self.cached_data[index]

        ex = self.data[index]
        if len(self.transform_hooks) > 0:
            ret = []
            for field, cols in self.transform_hooks:
                _ex = ex[cols]
                if isinstance(cols, List):
                    processed_ex = field.process(*_ex)
                else:
                    processed_ex = field.process(_ex)

                if isinstance(processed_ex, tuple):
                    ret.extend(processed_ex)
                else:
                    ret.append(processed_ex)
            ret = tuple(ret)
        else:
            ret = tuple(ex)

        if self.cache:
            self.cached_data[index] = ret

        return ret

    def is_empty(self) -> bool:
        """
        Return if the DataView has data

        """
        return len(self) == 0

    def cols(self) -> int:
        """ Return the amount of columns the DataView has."""
        if self.is_empty():
            raise ValueError("Empty DataView contains no columns")

        return len(self[0])

    def __len__(self) -> int:
        """
        Return the length of the dataview, ie the amount
        of examples it contains.

        """
        if self.data is None:
            return 0
        return len(self.data)

    def __setitem__(self):
        """Raise an error as DataViews are immutable."""
        raise ValueError("Dataset objects are immutable")

    def __delitem__(self):
        """Raise an error as DataViews are immutable."""
        raise ValueError("Dataset objects are immutable")


class TabularDataset(Dataset):
    """Loader for tabular data, usually in `csv` or `tsv` format.

    A TabularDataset can represent any data that can be organized
    in a table. Internally, we store all information in a 2D numpy
    generic array. This object also behaves
    as a sequence over the whole dataset, chaining the training,
    validation and test data, in that order. This is useful in creating
    vocabularies or loading embeddings over the full datasets.

    Attributes
    ----------
    train: np.ndarray
        The list of training examples
    val: np.ndarray
        The list of validation examples
    test: np.ndarray
        The list of text examples

    """

    def __init__(self,
                 train: Iterable[Iterable],
                 val: Optional[Iterable[Iterable]] = None,
                 test: Optional[Iterable[Iterable]] = None,
                 cache: bool = True,
                 named_columns: Optional[List[str]] = None,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:
        """Initialize the TabularDataset.

        Parameters
        ----------
        train: Iterable[Iterable]
            The train data
        val: Iterable[Iterable], optional
            The val data, optional
        test: Iterable[Iterable], optional
            The test data, optional
        cache: bool
            Whether to cache fetched examples. Only use True if the
            dataset fits in memory. Defaults to False.
        named_columns: Optional[List[Union[str, int]]]
            The columns' names of the dataset, in order.
        transform: Dict[str, Dict[str, Any]]
            The fields to be applied to the columns. Each field is
            identified with a name for easy linking.
            For example:
            {
               'text': {'field': SomeField(), 'columns': [0, 1]},
               'label': {'field': SomeOtherField(), 'columns': 2}
            }

        """
        self._train = np.array(train, dtype=np.object)
        self._val = None
        self._test = None

        if val is not None:
            self._val = np.array(val, dtype=np.object)
        if test is not None:
            self._test = np.array(test, dtype=np.object)

        self.cache = cache

        self.named_columns = named_columns

        cols = []
        # All datasets should be 2-dimensional
        for k, d in {"val": self._val, "test": self._test, "train": self._train}.items():
            if d is not None:
                cols.append(d.shape[-1])
                if len(d.shape) != 2:
                    # This happens when examples differ in the amount of
                    # columns and numpy stores them in a 1-D tensor
                    # (with tuples as values)
                    raise ValueError(
                        f"{k} dataset contains examples with different amount of columns"
                    )

        # Check that all splits contain same columns
        if np.unique(cols).shape != (1,):
            raise ValueError("All splits containing data should have same amount of columns")

        if named_columns and len(named_columns) != cols[0]:
            raise ValueError("Columns parameter should have same size as the dataset's amount " +
                             " of columns")

        # Store the hooks for lazy loading
        self.transform_hooks: List[Tuple[Field, Union[int, List[int]]]] = []
        self.transform = transform

        if transform:
            self._set_transforms(transform)

        self.train_view: Optional[DataView] = None
        self.val_view: Optional[DataView] = None
        self.test_view: Optional[DataView] = None

    def _set_transforms(self, transform: Dict[str, Union[Field, Dict]]) -> None:
        """Set transformations attributes and hooks to the data splits.

        This method adds attributes for each field in the transform
        dict. It also adds hooks for the 'process' call in each field.

        ATTENTION: This method works with the _train, _val and _test
        hidden attributes as this runs in the constructor and creates
        the hooks to be used in creating the properties.

        """
        columns: Union[int, List[int]]

        for k, t in enumerate(transform.items()):
            name, value = t
            if isinstance(value, Field):
                field = value
                columns = k
            else:
                try:
                    field, tmp_cols = value['field'], value.get('columns', k)

                    # Process as list to avoid repeating code
                    if not isinstance(tmp_cols, List):
                        tmp_cols = [tmp_cols]

                    for i, c in enumerate(tmp_cols[:]):
                        if isinstance(c, str):
                            if not self.named_columns:
                                raise ValueError(
                                    "Columns parameter is required for str-based indexing"
                                )
                            try:
                                tmp_cols[i] = self.named_columns.index(c)
                            except ValueError:
                                raise ValueError(
                                    f"Dataset has no column name {c}. " +
                                    f"Available columns: {self.named_columns}"
                                )

                    columns = tmp_cols

                    # If it was a value originally then process
                    # it as a single value
                    if len(tmp_cols) == 1:
                        columns = tmp_cols[0]

                except KeyError:
                    raise ValueError(
                        f"If a dict is provided in 'transform', then it must have the 'field' key."
                        f" transform item = {k, t}"
                    )

            setattr(self, name, field)
            args = [self._train[:, columns]]
            if self._val is not None:
                args.append(self._val[:, columns])
            if self._test is not None:
                args.append(self._test[:, columns])
            field.setup(*args)
            self.transform_hooks.append((field, columns))

    @registrable_factory
    @classmethod
    def from_path(cls,
                  train_path: str,
                  val_path: Optional[str] = None,
                  test_path: Optional[str] = None,
                  sep: Optional[str] = '\t',
                  header: Optional[str] = 'infer',
                  columns: Optional[Union[List[str], List[int]]] = None,
                  encoding: Optional[str] = 'utf-8',
                  transform: Dict[str, Union[Field, Dict]] = None) -> 'TabularDataset':
        """Load a TabularDataset from the given file paths.

        Parameters
        ----------
        train_path : str
            The path to the train data
        val_path : str, optional
            The path to the optional validation data
        test_path : str, optional
            The path to the optional test data
        sep: str
            Separator to pass to the `read_csv` method
        header: Optional[Union[str, int]]
            Use 0 for first line, None for no headers, and 'infer' to
            detect it automatically, defaults to 'infer'
        columns: List[str]
            List of columns to load, can be used to select a subset
            of columns, or change their order at loading time
        encoding: str
            The encoding format passed to the pandas reader
        transform: Dict[str, Union[Field, Dict]]
            The fields to be applied to the columns. Each field is
            identified with a name for easy linking.

        """

        if (
            columns and
            any(isinstance(c, int) for c in columns) and
            any(isinstance(c, str) for c in columns)
        ):
            raise ValueError("Columns parameters need to be all string or all integers.")

        train, cols = cls._load_file(train_path, sep, header, columns, encoding)

        val, test = None, None
        if val_path is not None:
            val, _ = cls._load_file(val_path, sep, header, columns, encoding)
        if test_path is not None:
            test, _ = cls._load_file(test_path, sep, header, columns, encoding)

        return cls(train=train, val=val, test=test, transform=transform, named_columns=cols)

    @registrable_factory
    @classmethod
    def autogen(cls,
                data_path: str,
                test_path: Optional[str] = None,
                seed: Optional[int] = None,
                test_ratio: Optional[float] = 0.2,
                val_ratio: Optional[float] = 0.2,
                sep: Optional[str] = '\t',
                header: Optional[str] = 'infer',
                columns: Optional[Union[List[str], List[int]]] = None,
                encoding: Optional[str] = 'utf-8',
                transform: Dict[str, Union[Field, Dict]] = None) -> 'TabularDataset':
        """Generate a test and validation set from the given file
        paths, then load a TabularDataset.

        Parameters
        ----------
        data_path: str
            The path to the data
        test_path: Optional[str]
            The path to the test data
        seed: Optional[int]
            Random seed to be used in test/val generation
        test_ratio: Optional[float]
            The ratio of the test dataset in relation to
            the whole dataset. If `test_path` is specified, this field
            has no effect.
        val_ratio: Optional[float]
            The ratio of the validation dataset in relation to
            the training dataset (whole - test)
        sep: str
            Separator to pass to the `read_csv` method
        header: Optional[Union[str, int]]
            Use 0 for first line, None for no headers, and 'infer' to
            detect it automatically, defaults to 'infer'
        columns: List[str]
            List of columns to load, can be used to select a subset
            of columns, or change their order at loading time
        encoding: str
            The encoding format passed to the pandas reader
        transform: Dict[str, Union[Field, Dict]]
            The fields to be applied to the columns. Each field is
            identified with a name for easy linking.

        """
        if (
            columns and
            any(isinstance(c, int) for c in columns) and
            any(isinstance(c, str) for c in columns)
        ):
            raise ValueError("Columns parameters need to be all string or all integers.")

        data, cols = cls._load_file(data_path,
                                    sep=sep,
                                    header=header,
                                    columns=columns,
                                    encoding=encoding)

        train, val, test = None, None, None
        if test_path is not None:
            train, val = train_test_split(data, test_size=val_ratio, random_state=seed)
            test, _ = cls._load_file(test_path,
                                     sep=sep,
                                     header=header,
                                     columns=columns,
                                     encoding=encoding)
        else:
            train_val, test = train_test_split(data, test_size=test_ratio, random_state=seed)
            train, val = train_test_split(train_val, test_size=val_ratio, random_state=seed)

        return cls(train=train, val=val, test=test, transform=transform, named_columns=cols)

    @classmethod
    def _load_file(cls,
                   path: str,
                   sep: Optional[str] = '\t',
                   header: Optional[str] = 'infer',
                   columns: Optional[Union[List[str], List[int]]] = None,
                   encoding: Optional[str] = 'utf-8') -> Tuple[List[Tuple], Optional[List[str]]]:
        """Load data from the given path.

        The path may be either a single file or a directory. If it is
        a directory, each file is loaded according to the specified
        options and all the data is concatenated into a single list.
        The files will be processed in order based on file name.

        Parameters
        ----------
        path : str
            Path to data, could be a directory, a file, or a
            smart_open link
        sep: str
            Separator to pass to the `read_csv` method
        header: Optional[Union[str, int]]
            Use 0 for first line, None for no headers, and 'infer' to
            detect it automatically, defaults to 'infer'
        columns: Optional[Union[List[str], List[int]]]
            List of columns to load, can be used to select a subset
            of columns, or change their order at loading time
        encoding: str
            The encoding format passed to the pandas reader

        Returns
        -------
        Tuple[List[Tuple], Optional[List[str]]]
            A tuple containing the list of examples (where each example
            is itself also a list or tuple of entries in the dataset)
            and an optional list of named columns (one string for each
            column in the dataset)

        """
        # Get all paths
        if isinstance(path, str) and os.path.isdir(path):
            file_paths = [os.path.join(path, name) for name in os.listdir(path)]
            file_paths = sorted(file_paths)
        else:
            file_paths = [path]

        data: List = []
        for file_path in file_paths:
            # Don't fail on buggy files
            try:
                examples = pd.read_csv(file_path,
                                       sep=sep,
                                       header=header,
                                       index_col=False,
                                       dtype=str,
                                       encoding=encoding,
                                       keep_default_na=False)
                # Select columns
                if columns is not None:
                    examples = examples[columns]
                data.extend(examples.values.tolist())
            except Exception as e:
                print("Warning: failed to load file {file_path}")
                print(e)

        if len(data) == 0:
            raise ValueError(f"No data found at {path}")

        # Take the named columns from the columns parameter
        # if they are strings or try to use the pd.DataFrame
        # column names if they are strings.
        named_cols: List[str] = []
        if columns:
            for i, c in enumerate(columns):  # type: ignore
                if isinstance(c, str):
                    named_cols.append(c)
        elif all(isinstance(c, str) for c in examples.columns):
            named_cols = examples.columns.tolist()

        return data, named_cols if len(named_cols) > 0 else None

    @property
    def train(self) -> np.ndarray:
        """Returns the training data as a numpy nd array"""
        if self.train_view is None:
            self.train_view = DataView(self._train, self.transform_hooks, self.cache)

        return self.train_view

    @property
    def val(self) -> np.ndarray:
        """Returns the validation data as a numpy nd array"""
        if self.val_view is None:
            self.val_view = DataView(self._val, self.transform_hooks, self.cache)

        return self.val_view

    @property
    def test(self) -> np.ndarray:
        """Returns the test data as a numpy nd array"""
        if self.test_view is None:
            self.test_view = DataView(self._test, self.transform_hooks, self.cache)

        return self.test_view

    @property
    def raw(self) -> np.ndarray:
        """Returns all partitions of the data as a numpy nd array"""
        args = [self._train]
        if not self.val.is_empty():
            args.append(self.val.raw)
        if not self.test.is_empty():
            args.append(self.test.raw)
        return np.concatenate(args, axis=0)

    @property
    def cols(self) -> int:
        """Returns the amount of columns in the tabular dataset"""
        return self.train.cols()

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.train) + len(self.val) + len(self.test)

    def __iter__(self):
        """Iterate through the dataset."""
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        """Get the item at the given index."""
        ceiling = len(self.train)
        if index < ceiling:
            return self.train[index]

        offset = ceiling
        ceiling += len(self.val)
        if index < ceiling:
            return self.val[index - offset]

        offset = ceiling
        ceiling += len(self.test)
        if index < ceiling:
            return self.test[index - offset]
