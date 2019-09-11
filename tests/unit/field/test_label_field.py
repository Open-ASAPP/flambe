import pytest

from numpy import isclose

from flambe.field import LabelField
from flambe.tokenizer import LabelTokenizer


NUMERIC_PRECISION = 1e-2


def test_tokenizer():
    """Test one hot featurizer."""
    tokenizer = LabelTokenizer()

    dummy = 'LABEL1'
    assert tokenizer.tokenize(dummy) == ['LABEL1']

    dummy = 'LABEL1,LABEL2'
    assert tokenizer.tokenize(dummy) == ['LABEL1,LABEL2']

    dummy = 'LABEL1 LABEL2'
    assert tokenizer.tokenize(dummy) == ['LABEL1 LABEL2']


def test_tokenizer_multilabel():
    """Test one hot featurizer."""
    tokenizer = LabelTokenizer(multilabel_sep=',')

    dummy = 'LABEL1'
    assert tokenizer.tokenize(dummy) == ['LABEL1']

    dummy = 'LABEL1,LABEL2'
    assert tokenizer.tokenize(dummy) == ['LABEL1', 'LABEL2']

    dummy = 'LABEL1 LABEL2'
    assert tokenizer.tokenize(dummy) == ['LABEL1 LABEL2']


def test_label_process():
    """Test label nuemricalization."""
    dummy = ['LABEL1', 'LABEL3', 'LABEL2', 'LABEL2']

    field = LabelField()
    field.setup(dummy)

    assert len(field.vocab) == 3
    assert int(field.process('LABEL1')) == 0
    assert int(field.process('LABEL2')) == 2
    assert int(field.process('LABEL3')) == 1


def test_label_process_multilabel():
    """Test label nuemricalization."""
    dummy = ['LABEL1,LABEL2', 'LABEL3', 'LABEL2,LABEL1', 'LABEL2']

    field = LabelField()
    field.setup(dummy)
    assert len(field.vocab) == 4

    field = LabelField(multilabel_sep=',')
    field.setup(dummy)
    assert len(field.vocab) == 3

    assert list(field.process('LABEL1,LABEL2')) == [0, 1]
    assert list(field.process('LABEL2,LABEL1')) == [1, 0]
    assert int(field.process('LABEL2')) == 1
    assert int(field.process('LABEL3')) == 2


def test_label_process_one_hot():
    """Test label numericalization."""
    dummy = ['LABEL1', 'LABEL3', 'LABEL2', 'LABEL2']

    field = LabelField(one_hot=True)
    field.setup(dummy)

    assert len(field.vocab) == 3
    assert list(field.process('LABEL1')) == [1, 0, 0]
    assert list(field.process('LABEL2')) == [0, 0, 1]
    assert list(field.process('LABEL3')) == [0, 1, 0]


def test_label_process_multilabel_one_hot():
    """Test label numericalization."""
    dummy = ['LABEL1,LABEL2', 'LABEL3', 'LABEL2,LABEL1', 'LABEL2']

    field = LabelField(multilabel_sep=',', one_hot=True)
    field.setup(dummy)

    assert len(field.vocab) == 3
    assert list(field.process('LABEL1,LABEL2')) == [1, 1, 0]
    assert list(field.process('LABEL2,LABEL1')) == [1, 1, 0]
    assert list(field.process('LABEL2')) == [0, 1, 0]
    assert list(field.process('LABEL3')) == [0, 0, 1]


def test_label_frequencies():
    """Test label frequencies."""
    dummy = ['LABEL1'] * 80
    dummy.extend(['LABEL2'] * 20)

    field = LabelField()
    field.setup(dummy)

    assert len(field.vocab) == 2

    assert len(field.label_freq) == 2
    assert isclose(field.label_freq[0].item(), 0.8, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_freq[1].item(), 0.2, rtol=NUMERIC_PRECISION)

    assert len(field.label_count) == 2
    assert isclose(field.label_count[0].item(), 80, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_count[1].item(), 20, rtol=NUMERIC_PRECISION)

    assert len(field.label_inv_freq) == 2
    assert isclose(field.label_inv_freq[0].item(), 1.25, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_inv_freq[1].item(), 5, rtol=NUMERIC_PRECISION)


def test_label_frequencies_2():
    """Test label frequencies."""
    dummy = ['LABEL1'] * 80

    field = LabelField()
    field.setup(dummy)

    assert len(field.vocab) == 1

    assert len(field.label_freq) == 1
    assert isclose(field.label_freq[0].item(), 1, rtol=NUMERIC_PRECISION)

    assert len(field.label_count) == 1
    assert isclose(field.label_count[0].item(), 80, rtol=NUMERIC_PRECISION)

    assert len(field.label_inv_freq) == 1
    assert isclose(field.label_inv_freq[0].item(), 1, rtol=NUMERIC_PRECISION)


def test_label_frequencies_3():
    """Test label frequencies."""
    dummy = []

    field = LabelField()
    field.setup(dummy)

    assert len(field.vocab) == 0

    assert len(field.label_freq) == 0
    assert len(field.label_count) == 0
    assert len(field.label_inv_freq) == 0


def test_label_process_one_hot_frequencies():
    """Test label numericalization."""
    dummy = ['LABEL1', 'LABEL3', 'LABEL2', 'LABEL2']

    field = LabelField(one_hot=True)
    field.setup(dummy)

    assert len(field.label_freq) == 3
    assert isclose(field.label_freq[0].item(), 0.25, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_freq[1].item(), 0.25, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_freq[2].item(), 0.5, rtol=NUMERIC_PRECISION)

    assert len(field.label_count) == 3
    assert isclose(field.label_count[0].item(), 1, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_count[1].item(), 1, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_count[2].item(), 2, rtol=NUMERIC_PRECISION)

    assert len(field.label_inv_freq) == 3
    assert isclose(field.label_inv_freq[0].item(), 4, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_inv_freq[1].item(), 4, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_inv_freq[2].item(), 2, rtol=NUMERIC_PRECISION)


def test_label_process_multilabel_one_hot_frequencies():
    """Test label numericalization."""
    dummy = ['LABEL1,LABEL2', 'LABEL3', 'LABEL2,LABEL1', 'LABEL2']

    field = LabelField(multilabel_sep=',', one_hot=True)
    field.setup(dummy)

    assert len(field.label_freq) == 3
    assert isclose(field.label_freq[0].item(), 0.333, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_freq[1].item(), 0.5, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_freq[2].item(), 0.166, rtol=NUMERIC_PRECISION)

    assert len(field.label_count) == 3
    assert isclose(field.label_count[0].item(), 2, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_count[1].item(), 3, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_count[2].item(), 1, rtol=NUMERIC_PRECISION)

    assert len(field.label_inv_freq) == 3
    assert isclose(field.label_inv_freq[0].item(), 3, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_inv_freq[1].item(), 2, rtol=NUMERIC_PRECISION)
    assert isclose(field.label_inv_freq[2].item(), 6, rtol=NUMERIC_PRECISION)


def test_pass_bool_labels():
    """Test labels specified in the init"""
    dummy = [True, False, True, True]

    field = LabelField(labels=[False, True])
    field.setup(dummy)

    assert len(field.vocab) == 2
    assert int(field.process(False)) == 0
    assert int(field.process(True)) == 1

    field = LabelField(labels=[True, False])
    field.setup(dummy)

    assert len(field.vocab) == 2
    assert int(field.process(False)) == 1
    assert int(field.process(True)) == 0


def test_pass_labels():
    """Test labels specified in the init"""
    dummy = ['LABEL1', 'LABEL3', 'LABEL2', 'LABEL2']

    field = LabelField(labels=['LABEL1', 'LABEL2', 'LABEL3'])
    field.setup(dummy)

    assert len(field.vocab) == 3
    assert int(field.process('LABEL1')) == 0
    assert int(field.process('LABEL2')) == 1
    assert int(field.process('LABEL3')) == 2

    field = LabelField(labels=['LABEL3', 'LABEL1', 'LABEL2'])
    field.setup(dummy)

    assert len(field.vocab) == 3
    assert int(field.process('LABEL1')) == 1
    assert int(field.process('LABEL2')) == 2
    assert int(field.process('LABEL3')) == 0


def test_pass_labels_with_unkown_1():
    """Test labels specified in the init"""
    dummy = ['LABEL1', 'LABEL3', 'LABEL2', 'LABEL2']

    field = LabelField(labels=['LABEL1', 'LABEL2'])
    with pytest.raises(ValueError):
        field.setup(dummy)


def test_pass_labels_with_unkown_2():
    """Test labels specified in the init"""
    dummy = ['LABEL1', 'LABEL3', 'LABEL2', 'LABEL2']

    field = LabelField(labels=['LABEL1', 'LABEL2', 'LABEL3'])
    field.setup(dummy)
    with pytest.raises(ValueError):
        list(field.process('LABEL4'))
