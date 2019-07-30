from flambe.nlp.classification import SSTDataset, TRECDataset, NewsGroupDataset


def test_dataset_sst():
    dataset = SSTDataset()
    assert len(dataset.train) == 6920 
    assert len(dataset.train[0]) == 2

    assert len(dataset.val) == 872
    assert len(dataset.val[0]) == 2

    assert len(dataset.test) == 1821
    assert len(dataset.test[0]) == 2


def test_dataset_trec():
    dataset = TRECDataset()
    assert len(dataset.train) == 5452
    assert len(dataset.train[0]) == 2

    assert len(dataset.val) == 0

    assert len(dataset.test) == 500
    assert len(dataset.test[0]) == 2

def test_dataset_news():
    try:
        from sklearn.datasets import fetch_20newsgroups

        dataset = NewsGroupDataset()

        assert len(dataset.train) == 11314
        assert len(dataset.train[0]) == 2

        assert len(dataset.val) == 0

        assert len(dataset.test) == 7532
        assert len(dataset.test[0]) == 2

    except ImportError:
        pass
