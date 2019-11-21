import pytest
import numpy as np
import torch
from flambe.dataset import TabularDataset
from flambe.field import Field, TextField, LabelField


@pytest.fixture
def train_dataset():
    """Dummy dataset from file"""
    return TabularDataset.from_path('tests/data/dummy_tabular/train.csv', sep=',')


@pytest.fixture
def train_dataset_no_header():
    """Dummy dataset from file"""
    return TabularDataset.from_path('tests/data/no_header_dataset.csv', sep=',',
                                    header=None)


@pytest.fixture
def train_dataset_reversed():
    """Dummy dataset from file"""
    return TabularDataset.from_path('tests/data/dummy_tabular/train.csv', sep=',',
                                    columns=['label', 'text'])


@pytest.fixture
def full_dataset():
    """Dummy dataset from file"""
    return TabularDataset.from_path(train_path='tests/data/dummy_tabular/train.csv',
                                    val_path='tests/data/dummy_tabular/val.csv', sep=',')


@pytest.fixture
def dir_dataset():
    """Dummy dataset from directory"""
    return TabularDataset.from_path('tests/data/dummy_tabular', sep=',')


@pytest.fixture
def autogen_dataset():
    """Dummy dataset from file with auto-generated val and test"""
    return TabularDataset.autogen('tests/data/dummy_tabular/train.csv',
                                  seed=42,
                                  sep=',')


@pytest.fixture
def autogen_dataset_with_test():
    """Dummy dataset from file with auto-generated val and given test"""
    return TabularDataset.autogen('tests/data/dummy_tabular/train.csv',
                                  test_path='tests/data/dummy_tabular_test/test.csv',
                                  seed=42,
                                  sep=',')


@pytest.fixture
def autogen_dataset_dir():
    """Dummy dataset from directory with auto-generated val and test"""
    return TabularDataset.autogen('tests/data/dummy_tabular',
                                  seed=42,
                                  sep=',')


@pytest.fixture
def autogen_dataset_dir_with_test():
    """Dummy dataset from dir with auto-generated val and given test"""
    return TabularDataset.autogen('tests/data/dummy_tabular',
                                  test_path='tests/data/dummy_tabular_test',
                                  seed=42,
                                  sep=',')


@pytest.fixture
def autogen_dataset_ratios():
    """Dummy dataset from file with auto-generated val and test with
    different ratios
    """
    return TabularDataset.autogen('tests/data/dummy_tabular/train.csv',
                                  seed=42,
                                  sep=',',
                                  test_ratio=0.5,
                                  val_ratio=0.5)


@pytest.fixture
def autogen_dataset_ratios_with_test():
    """Dummy dataset from file with auto-generated val and given test
    with different ratios
    """
    return TabularDataset.autogen('tests/data/dummy_tabular/train.csv',
                                  test_path='tests/data/dummy_tabular_test/test.csv',
                                  seed=42,
                                  sep=',',
                                  test_ratio=0.5,  # no effect
                                  val_ratio=0.5)


@pytest.fixture
def autogen_dataset_dir_ratios():
    """Dummy dataset from directory with auto-generated val and test
    with different ratios
    """
    return TabularDataset.autogen('tests/data/dummy_tabular',
                                  seed=42,
                                  sep=',',
                                  test_ratio=0.5,
                                  val_ratio=0.5)


@pytest.fixture
def autogen_dataset_dir_ratios_with_test():
    """Dummy dataset from directory with auto-generated val
    and given test with different ratios
    """
    return TabularDataset.autogen('tests/data/dummy_tabular',
                                  test_path='tests/data/dummy_tabular_test',
                                  seed=42,
                                  sep=',',
                                  test_ratio=0.5,  # no effect
                                  val_ratio=0.5)


def test_valid_dataset():
    """Test trivial dataset build process"""
    train = (("Lorem ipsum dolor sit amet", 3, 4.5),
             ("Sed ut perspiciatis unde", 5, 5.5))
    val = (("ipsum quia dolor sit", 10, 3.5),)
    test = (("Ut enim ad minima veniam", 100, 35),)

    t = TabularDataset(train, val, test)

    assert len(t) == 4
    assert len(t.train) == 2
    assert len(t.val) == 1
    assert len(t.test) == 1

    def check(d, t):
        for i, tu in enumerate(d):
            v0, v1, v2 = tu
            assert t[i][0] == v0
            assert t[i][1] == v1
            assert t[i][2] == v2

    check(train, t.train)
    check(val, t.val)
    check(test, t.test)


def test_invalid_dataset():
    """Test dataset is invalid as it has different columns"""
    train = (("Lorem ipsum dolor sit amet", 3, 4.5),
             ("Sed ut perspiciatis unde", 5.5))
    with pytest.raises(ValueError):
        TabularDataset(train)


def test_invalid_dataset2():
    """Test dataset is invalid as different splits contain different
    columns
    """
    train = (("Lorem ipsum dolor sit amet", 3, 4.5),
             ("Sed ut perspiciatis unde", 4, 5.5))
    val = (("ipsum quia dolor sit", 3.5),)
    with pytest.raises(ValueError):
        TabularDataset(train, val)


def test_incomplete_dataset():
    """Test dataset missing either val or test"""
    train = (("Lorem ipsum dolor sit amet", 3, 4.5),
             ("Sed ut perspiciatis unde", 4, 5.5))
    t = TabularDataset(train)

    assert len(t.val) == 0
    assert len(t.test) == 0


def test_cache_dataset():
    """Test caching the dataset"""
    train = (
            ("Lorem ipsum dolor sit amet", 3, 4.5),
            ("Sed ut perspiciatis unde", 5, 5.5),
            ("Lorem ipsum dolor sit amet", 3, 4.5),
            ("Sed ut perspiciatis unde", 5, 5.5),
            ("Lorem ipsum dolor sit amet", 3, 4.5),
            ("Sed ut perspiciatis unde", 5, 5.5),
            ("Lorem ipsum dolor sit amet", 3, 4.5),
            ("Sed ut perspiciatis unde", 5, 5.5),
            ("Lorem ipsum dolor sit amet", 3, 4.5),
            ("Sed ut perspiciatis unde", 5, 5.5))

    t = TabularDataset(train, cache=True)

    assert len(t.train.cached_data) == 0
    for i, _ in enumerate(t.train):
        assert len(t.train.cached_data) == i + 1


def test_column_attr(train_dataset):
    assert len(train_dataset.named_columns) == 2
    assert train_dataset.named_columns == ['text', 'label']


def test_column_attr2(train_dataset_no_header):
    assert train_dataset_no_header.named_columns is None


def test_named_columns():
    """Test dataset is invalid as it has different columns"""
    train = (("Lorem ipsum dolor sit amet", 3),
             ("Sed ut perspiciatis unde", 5.5))
    TabularDataset(train, named_columns=['col1', 'col2'])


def test_invalid_columns():
    """Test dataset is invalid as it has different columns"""
    train = (("Lorem ipsum dolor sit amet", 3),
             ("Sed ut perspiciatis unde", 5.5))
    with pytest.raises(ValueError):
        TabularDataset(train, named_columns=['some_random_col'])


def test_dataset_from_file(train_dataset):
    """Test loading a dataset from file"""
    dummy = "justo. Praesent luctus. Curabitur egestas nunc sed libero. Proin sed"
    assert train_dataset[0][0] == dummy
    assert train_dataset[0][1] == '6'


def test_dataset_from_file_reversed(train_dataset_reversed):
    """Test loading a dataset from file"""
    dummy = "justo. Praesent luctus. Curabitur egestas nunc sed libero. Proin sed"
    assert train_dataset_reversed[0][0] == '6'
    assert train_dataset_reversed[0][1] == dummy


def test_full_dataset_from_file(full_dataset):
    """Test loading a dataset from file"""
    train_dummy = "justo. Praesent luctus. Curabitur egestas nunc sed libero. Proin sed"
    val_dummy = "malesuada. Integer id magna et ipsum cursus vestibulum. Mauris magna."

    assert full_dataset.train[0][0] == train_dummy
    assert full_dataset.train[0][1] == '6'

    assert full_dataset.val[0][0] == val_dummy
    assert full_dataset.val[0][1] == '8'

    assert full_dataset[0][0] == train_dummy
    assert full_dataset[100][0] == val_dummy


def test_dataset_from_dir(dir_dataset):
    """Test loading multiple datasets at once"""
    dummy = "malesuada. Integer id magna et ipsum cursus vestibulum. Mauris magna."
    assert dir_dataset[0][0] == dummy
    assert dir_dataset[0][1] == '8'

    dummy = "Sed molestie. Sed id risus quis diam luctus lobortis. Class"
    assert dir_dataset[100][0] == dummy
    assert dir_dataset[100][1] == '6'


def test_dataset_autogen(autogen_dataset):
    """Test autogenerating val and test sets from a file"""
    train_dummy = "eget, venenatis a, magna. Lorem ipsum dolor sit amet, consectetuer"
    val_dummy = "leo. Vivamus nibh dolor, nonummy ac, feugiat non, lobortis quis,"
    test_dummy = "turpis egestas. Aliquam fringilla cursus purus. Nullam scelerisque neque sed"

    assert autogen_dataset.train[0][0] == train_dummy
    assert autogen_dataset.train[0][1] == '8'
    assert len(autogen_dataset.train) == 64

    assert autogen_dataset.val[0][0] == val_dummy
    assert autogen_dataset.val[0][1] == '1'
    assert len(autogen_dataset.val) == 16

    assert autogen_dataset.test[0][0] == test_dummy
    assert autogen_dataset.test[0][1] == '6'
    assert len(autogen_dataset.test) == 20


def test_dataset_autogen_with_test(autogen_dataset_with_test):
    """Test autogenerating val and test sets from a file"""
    train_dummy = "Etiam ligula tortor, dictum eu, placerat eget, venenatis a, magna."
    val_dummy = "turpis egestas. Aliquam fringilla cursus purus. Nullam scelerisque neque sed"
    test_dummy = "a sollicitudin orci sem eget massa. Suspendisse eleifend. Cras sed"

    assert autogen_dataset_with_test.train[0][0] == train_dummy
    assert autogen_dataset_with_test.train[0][1] == '6'
    assert len(autogen_dataset_with_test.train) == 80

    assert autogen_dataset_with_test.val[0][0] == val_dummy
    assert autogen_dataset_with_test.val[0][1] == '6'
    assert len(autogen_dataset_with_test.val) == 20

    assert autogen_dataset_with_test.test[0][0] == test_dummy
    assert autogen_dataset_with_test.test[0][1] == '3'
    assert len(autogen_dataset_with_test.test) == 50


def test_dataset_autogen_ratios(autogen_dataset_ratios):
    """Test autogenerating val and test sets from a file with ratios"""
    train_dummy = "leo. Vivamus nibh dolor, nonummy ac, feugiat non, lobortis quis,"
    val_dummy = "ac turpis egestas. Aliquam fringilla cursus purus. Nullam scelerisque neque"
    test_dummy = "turpis egestas. Aliquam fringilla cursus purus. Nullam scelerisque neque sed"

    assert autogen_dataset_ratios.train[0][0] == train_dummy
    assert autogen_dataset_ratios.train[0][1] == '1'
    assert len(autogen_dataset_ratios.train) == 25

    assert autogen_dataset_ratios.val[0][0] == val_dummy
    assert autogen_dataset_ratios.val[0][1] == '6'
    assert len(autogen_dataset_ratios.val) == 25

    assert autogen_dataset_ratios.test[0][0] == test_dummy
    assert autogen_dataset_ratios.test[0][1] == '6'
    assert len(autogen_dataset_ratios.test) == 50


def test_dataset_autogen_ratios_with_test(autogen_dataset_ratios_with_test):
    """Test autogenerating val set from a file with ratios with test"""
    train_dummy = "leo. Vivamus nibh dolor, nonummy ac, feugiat non, lobortis quis,"
    val_dummy = "turpis egestas. Aliquam fringilla cursus purus. Nullam scelerisque neque sed"
    test_dummy = "a sollicitudin orci sem eget massa. Suspendisse eleifend. Cras sed"

    assert autogen_dataset_ratios_with_test.train[0][0] == train_dummy
    assert autogen_dataset_ratios_with_test.train[0][1] == '1'
    assert len(autogen_dataset_ratios_with_test.train) == 50

    assert autogen_dataset_ratios_with_test.val[0][0] == val_dummy
    assert autogen_dataset_ratios_with_test.val[0][1] == '6'
    assert len(autogen_dataset_ratios_with_test.val) == 50

    assert autogen_dataset_ratios_with_test.test[0][0] == test_dummy
    assert autogen_dataset_ratios_with_test.test[0][1] == '3'
    assert len(autogen_dataset_ratios_with_test.test) == 50


def test_dataset_autogen_dir_val_test(autogen_dataset_dir):
    """Test autogenerating val and test sets from a dir"""
    train_dummy = "egestas blandit. Nam nulla magna, malesuada vel, convallis in, cursus"
    val_dummy = "turpis egestas. Aliquam fringilla cursus purus. Nullam scelerisque neque sed"
    test_dummy = "nibh. Aliquam ornare, libero at auctor ullamcorper, nisl arcu iaculis"

    assert autogen_dataset_dir.train[0][0] == train_dummy
    assert autogen_dataset_dir.train[0][1] == '1'
    assert len(autogen_dataset_dir.train) == 160

    assert autogen_dataset_dir.val[0][0] == val_dummy
    assert autogen_dataset_dir.val[0][1] == '6'
    assert len(autogen_dataset_dir.val) == 41

    assert autogen_dataset_dir.test[0][0] == test_dummy
    assert autogen_dataset_dir.test[0][1] == '10'
    assert len(autogen_dataset_dir.test) == 51


def test_dataset_autogen_dir_with_test(autogen_dataset_dir_with_test):
    """Test autogenerating val from a dir and given test"""
    train_dummy = "Donec non justo. Proin non massa non ante bibendum ullamcorper."
    val_dummy = "nibh. Aliquam ornare, libero at auctor ullamcorper, nisl arcu iaculis"
    test_dummy = "a sollicitudin orci sem eget massa. Suspendisse eleifend. Cras sed"

    assert autogen_dataset_dir_with_test.train[0][0] == train_dummy
    assert autogen_dataset_dir_with_test.train[0][1] == '4'
    assert len(autogen_dataset_dir_with_test.train) == 201

    assert autogen_dataset_dir_with_test.val[0][0] == val_dummy
    assert autogen_dataset_dir_with_test.val[0][1] == '10'
    assert len(autogen_dataset_dir_with_test.val) == 51

    assert autogen_dataset_dir_with_test.test[0][0] == test_dummy
    assert autogen_dataset_dir_with_test.test[0][1] == '3'
    assert len(autogen_dataset_dir_with_test.test) == 50


def test_dataset_autogen_dir_val_test_ratios(autogen_dataset_dir_ratios):
    """Test autogenerating val set from a dir with given test and
    different ratios
    """
    train_dummy = "Aenean euismod mauris eu elit. Nulla facilisi. Sed neque. Sed"
    val_dummy = "ut quam vel sapien imperdiet ornare. In faucibus. Morbi vehicula."
    test_dummy = "nibh. Aliquam ornare, libero at auctor ullamcorper, nisl arcu iaculis"

    assert autogen_dataset_dir_ratios.train[0][0] == train_dummy
    assert autogen_dataset_dir_ratios.train[0][1] == '9'
    assert len(autogen_dataset_dir_ratios.train) == 63

    assert autogen_dataset_dir_ratios.val[0][0] == val_dummy
    assert autogen_dataset_dir_ratios.val[0][1] == '10'
    assert len(autogen_dataset_dir_ratios.val) == 63

    assert autogen_dataset_dir_ratios.test[0][0] == test_dummy
    assert autogen_dataset_dir_ratios.test[0][1] == '10'
    assert len(autogen_dataset_dir_ratios.test) == 126


def test_dataset_autogen_dir_val_test_ratios_with_test(autogen_dataset_dir_ratios_with_test):
    """Test autogenerating val and test sets from a file"""
    train_dummy = "Donec egestas. Aliquam nec enim. Nunc ut erat. Sed nunc"
    val_dummy = "nibh. Aliquam ornare, libero at auctor ullamcorper, nisl arcu iaculis"
    test_dummy = "a sollicitudin orci sem eget massa. Suspendisse eleifend. Cras sed"

    assert autogen_dataset_dir_ratios_with_test.train[0][0] == train_dummy
    assert autogen_dataset_dir_ratios_with_test.train[0][1] == '1'
    assert len(autogen_dataset_dir_ratios_with_test.train) == 126

    assert autogen_dataset_dir_ratios_with_test.val[0][0] == val_dummy
    assert autogen_dataset_dir_ratios_with_test.val[0][1] == '10'
    assert len(autogen_dataset_dir_ratios_with_test.val) == 126

    assert autogen_dataset_dir_ratios_with_test.test[0][0] == test_dummy
    assert autogen_dataset_dir_ratios_with_test.test[0][1] == '3'
    assert len(autogen_dataset_dir_ratios_with_test.test) == 50


def test_dataset_length(train_dataset, full_dataset):
    """Test dataset length."""
    assert len(train_dataset) == 100
    assert len(full_dataset) == 200
    assert len(full_dataset.train) == 100
    assert len(full_dataset.val) == 100


def test_dataset_iter(train_dataset):
    """Test that the dataset is iterable."""
    for i, ex in enumerate(train_dataset):
        assert np.array_equal(ex, train_dataset[i])


def test_dataset_setitem(train_dataset):
    """Test dataset is immutable"""
    with pytest.raises(Exception):
        train_dataset[0] = 0


def test_dataset_deltitem(train_dataset):
    """Test dataset is immutable"""
    with pytest.raises(Exception):
        del train_dataset[0]


def test_dataset_transform():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "text": TextField(),
        "label": LabelField()
    }

    t = TabularDataset(train, transform=transform)

    assert hasattr(t, "text")
    assert hasattr(t, "label")

    assert t.label.vocab_size == 2
    assert t.text.vocab_size == 11


def test_dataset_transform_2():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "text": {
            "field": TextField()
        },
        "label": {
            "field": LabelField(),
            "columns": 1
        }
    }

    t = TabularDataset(train, transform=transform)

    assert hasattr(t, "text")
    assert hasattr(t, "label")

    assert t.label.vocab_size == 2


def test_dataset_transform_3():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "text": {
            "columns": 0
        },
        "label": {
            "field": LabelField(),
            "columns": 1
        }
    }

    with pytest.raises(ValueError):
        TabularDataset(train, transform=transform)


def test_dataset_transform_4():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "t1": {
            "field": TextField(),
            "columns": 1
        },
        "t2": {
            "field": TextField(),
            "columns": 1
        }
    }

    t = TabularDataset(train, transform=transform)

    assert t.train.cols() == 2


def test_dataset_transform_5():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "t1": {
            "field": TextField(),
            "columns": 0
        },
        "t2": {
            "field": TextField(),
            "columns": 0
        }
    }

    t = TabularDataset(train, transform=transform)
    assert t.train.cols() == 2


def test_dataset_transform_6():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    class DummyField(Field):
        def setup(self, *data: np.ndarray) -> None:
            pass

        def process(self, ex1, ex2):
            return torch.tensor(0)

    transform = {
        "text": {
            "field": DummyField(),
            "columns": [0, 1]
        }
    }

    t = TabularDataset(train, transform=transform)
    assert t.train.cols() == 1


def test_dataset_transform_7():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    class DummyField(Field):
        def setup(self, *data: np.ndarray) -> None:
            pass

        def process(self, ex1, ex2):
            return torch.tensor(0)

    transform = {
        "text": {
            "field": DummyField(),
            "columns": [0, 1]
        },
        "other": {
            "field": DummyField(),
            "columns": [0, 1]
        },
        "other2": {
            "field": LabelField(),
            "columns": 0
        }
    }

    t = TabularDataset(train, transform=transform)
    assert t.train.cols() == 3


def test_dataset_transform_8():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "tx": {
            "field": LabelField(),
            "columns": [0, 1]
        }
    }

    with pytest.raises(TypeError):
        t = TabularDataset(train, transform=transform)
        t.train.cols()


def test_dataset_transform_with_named_cols():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "tx": {
            "field": LabelField(),
            "columns": 'label'
        }
    }

    t = TabularDataset(train, transform=transform, named_columns=['text', 'label'])
    assert len(t.train[0]) == 1


def test_dataset_transform_with_invalid_named_cols():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "tx": {
            "field": LabelField(),
            "columns": 'none_existent'
        }
    }

    with pytest.raises(ValueError):
        TabularDataset(train, transform=transform, named_columns=['text', 'label'])


def test_dataset_transform_with_mixed_cols():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "label": {
            "field": LabelField(),
            "columns": 1,
        },
        "text": {
            "field": TextField(),
            "columns": 'text',
        }
    }

    t = TabularDataset(train, transform=transform, named_columns=['text', 'label'])
    assert len(t.train) == 2
    assert len(t.train[0]) == 2


def test_dataset_transform_mixed_multiple_named_cols():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    class DummyField(Field):
        def setup(self, *data: np.ndarray) -> None:
            pass

        def process(self, ex1, ex2):
            return torch.tensor(0)

    transform = {
        "text": {
            "field": DummyField(),
            "columns": ['text', 'label']
        },
        "other": {
            "field": DummyField(),
            "columns": [0, 1]
        },
        "other2": {
            "field": DummyField(),
            "columns": [0, 'label']
        }
    }

    t = TabularDataset(train, transform=transform, named_columns=['text', 'label'])
    assert t.train.cols() == 3
