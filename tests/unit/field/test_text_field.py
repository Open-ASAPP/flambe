from collections import OrderedDict as odict
from unittest.mock import MagicMock

import pytest
import torch
from flambe.field import TextField, BoWField
from flambe.tokenizer import WordTokenizer, CharTokenizer, NGramsTokenizer, NLTKWordTokenizer


example = (
"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt "
"ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco "
"laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in "
"voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat "
"non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
)

def test_word_tokenizer():
    tokenizer = WordTokenizer()

    dummy = "justo. Praesent luctus."
    assert tokenizer(dummy) == ['justo.', 'Praesent', 'luctus.']
    dummy = ""
    assert tokenizer(dummy) == []

def test_nltk_word_tokenizer():
    tokenizer = NLTKWordTokenizer()

    dummy = "justo. Praesent luctus."
    assert tokenizer(dummy) == ['justo', '.', 'Praesent', 'luctus', '.']
    dummy = ""
    assert tokenizer(dummy) == []

def test_ngram_tokenizer():
    tokenizer = NGramsTokenizer(2)

    dummy = "justo. Praesent luctus."
    assert tokenizer(dummy) == ['justo .', '. Praesent', 'Praesent luctus', 'luctus .']
    dummy = ""
    assert tokenizer(dummy) == []

def test_ngram_tokenizer_equivalence():
    t1 = NGramsTokenizer(1)
    t2 = NLTKWordTokenizer()

    assert t1(example) == t2(example)

def test_ngram_tokenizer_equivalence_2():
    t = NGramsTokenizer([1,2,3])

    ret = []
    for i in [1, 2, 3]:
        ret.extend(NGramsTokenizer(i)(example))

    assert t(example) == ret


def test_ngram_tokenizer_stopwords():
    tokenizer = NGramsTokenizer(2, exclude_stopwords=True)

    dummy = "justo. Praesent the luctus."
    assert tokenizer(dummy) == ['justo .', '. Praesent', 'Praesent luctus', 'luctus .']

    tokenizer = NGramsTokenizer(1, exclude_stopwords=True)

    dummy = "justo. Praesent the luctus."
    assert tokenizer(dummy) == ['justo', '.', 'Praesent', 'luctus', '.']

    tokenizer = NGramsTokenizer(2, exclude_stopwords=True, stop_words=["Praesent", "the"])

    dummy = "justo. Praesent the luctus."
    assert tokenizer(dummy) == ['justo .', '. luctus', 'luctus .']


def test_char_tokenizer():
    tokenizer = CharTokenizer()

    dummy = "justo. Praesent"
    assert tokenizer(dummy) == ["j", "u", "s", "t", "o", ".", " ",
                                "P", "r", "a", "e", "s", "e", "n", "t"]
    dummy = ""
    assert tokenizer(dummy) == []


def test_build_vocab():
    field = TextField(pad_token='<pad>', unk_token='<unk>')
    field._build_vocab()
    assert field.vocab == {'<pad>': 0, '<unk>': 1}

    dummy = ["justo Praesent luctus", "luctus praesent"]
    field._build_vocab(dummy)

    vocab = {'<pad>': 0, '<unk>': 1, 'justo': 2, 'Praesent': 3,
             'luctus': 4, 'praesent': 5}
    assert field.vocab == vocab


def test_bow_build_vocab():
    field = BoWField(min_freq=2, unk_token='<unk>')
    assert field.vocab == {'<unk>': 0}

    dummy = ["justo luctus Praesent luctus", "luctus praesent"]
    field.setup(dummy)

    vocab = {'<unk>': 0, 'luctus': 1}
    assert field.vocab == vocab


def test_build_vocab_lower():
    field = TextField(lower=True, pad_token=None, unk_token=None)

    dummy = ["justo Praesent luctus", "luctus praesent"]
    field.setup(dummy)

    vocab = {'justo': 0, 'praesent': 1, 'luctus': 2}
    assert field.vocab == vocab


def test_bow_build_vocab_lower():
    field = BoWField(min_freq=2)
    assert field.vocab == {'<unk>': 0}

    dummy = ["justo luctus Praesent Luctus", "luctus praesent"]
    field.setup(dummy)

    vocab = {'<unk>': 0, 'luctus': 1}
    assert field.vocab == vocab


def test_build_vocab_empty():
    field = TextField(pad_token=None, unk_token=None)
    assert field.vocab == dict()

    dummy = ["justo Praesent luctus", "luctus praesent"]
    field.setup(dummy)

    vocab = {'justo': 0, 'Praesent': 1, 'luctus': 2, 'praesent': 3}
    assert field.vocab == vocab


def test_build_vocab_setup_all_embeddings():
    """
    This test shows that all fields in the embeddings will be included.

    In embeddings and data:
        blue
        green
        yellow
    In embeddings only:
        purple
        gold
    In data only:
        white

    Expected vocab:
        blue
        green
        yellow
        purple
        gold
        white
    """

    model = MagicMock()
    model.vocab = {
        'purple': torch.rand(1),
        'gold': torch.rand(1),
        '<unk>': torch.rand(1),
        'blue': torch.rand(1),
        'green': torch.rand(1),
        '<pad>': torch.rand(1),
        'yellow': torch.rand(1),
    }
    model.__getitem__.side_effect = model.vocab.__getitem__
    model.__contains__.side_effect = model.vocab.__contains__

    field = TextField(
        model=model,
        setup_all_embeddings=True,
    )

    dummy = ["blue green", "yellow", 'white']

    field.setup(dummy)

    # assert vocab setup in expected order
    assert field.vocab == odict([
        ('<pad>', 0), ('<unk>', 1), ('blue', 2), ('green', 3),
        ('yellow', 4), ('white', 1), ('purple', 5), ('gold', 6),
    ])

    # assert embedding matrix organized in expected order
    assert torch.equal(
        field.embedding_matrix,
        torch.stack([
            model['<pad>'], model['<unk>'], model['blue'], model['green'],
            model['yellow'], model['purple'], model['gold'],
        ]),
    )


def test_build_vocab_decorators_specials():
    field = TextField(pad_token='<pad>', unk_token='<unk>',
                      sos_token='<sos>', eos_token='<eos>')

    field._build_vocab()

    assert field.vocab == {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    dummy = ["justo Praesent luctus", "luctus praesent"]
    field._build_vocab(dummy)

    vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3,
             'justo': 4, 'Praesent': 5, 'luctus': 6, 'praesent': 7}
    assert field.vocab == vocab


def test_build_vocab_decorators_missing_specials():
    field = TextField(pad_token=None, unk_token=None,
                      sos_token='<sos>', eos_token='<eos>')
    field._build_vocab()

    assert field.vocab == {'<sos>': 0, '<eos>': 1}
    dummy = ["justo Praesent luctus", "luctus praesent"]
    field._build_vocab(dummy)

    vocab = {'<sos>': 0, '<eos>': 1, 'justo': 2, 'Praesent': 3, 'luctus': 4, 'praesent': 5}
    assert field.vocab == vocab


def test_load_embeddings():
    field = TextField.with_embeddings(
        embeddings="tests/data/dummy_embeddings/test.txt",
        pad_token=None,
        unk_init_all=False,
    )
    dummy = "a test !"
    field.setup([dummy])

    # Now we have embeddings to check against
    true_embeddings = torch.tensor([[0.9, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]])
    assert len(field.embedding_matrix) == 3
    assert torch.all(torch.eq(field.embedding_matrix[1:3], true_embeddings))


def test_load_embeddings_empty_voc():
    field = TextField.with_embeddings(
        embeddings="tests/data/dummy_embeddings/test.txt",
        pad_token=None,
        unk_init_all=True,
    )

    dummy = "justo Praesent luctus justo praesent"
    field.setup([dummy])

    # No embeddings in the data, so get zeros
    assert len(field.embedding_matrix) == 5

    field = TextField.with_embeddings(
        embeddings="tests/data/dummy_embeddings/test.txt",
        pad_token=None,
        unk_init_all=False,
    )

    dummy = "justo Praesent luctus justo praesent"
    field.setup([dummy])

    # No embeddings in the data, so get zeros
    assert len(field.embedding_matrix) == 1


def test_text_process():
    field = TextField()
    field.setup()

    dummy = "justo Praesent luctus justo praesent"
    assert list(field.process(dummy)) == [1, 1, 1, 1, 1]

    field.setup([dummy])
    assert list(field.process(dummy)) == [2, 3, 4, 2, 5]


def test_bow_text_process():
    field = BoWField(min_freq=2)

    dummy = "justo praesent luctus justo praesent"
    field.setup([dummy])
    assert list(field.process(dummy)) == [0, 2, 2]


def test_bow_text_process_normalize():
    field = BoWField(min_freq=2, normalize=True)

    dummy = "justo praesent luctus justo praesent"
    field.setup([dummy])
    assert list(field.process(dummy)) == [0, 0.5, 0.5]


def test_bow_text_process_normalize_scale():
    field = BoWField(min_freq=2, normalize=True, scale_factor=10)

    dummy = "justo praesent luctus justo praesent"
    field.setup([dummy])
    assert list(field.process(dummy)) == [0, 5, 5]


def test_bow_text_process_scale():
    with pytest.raises(ValueError):
        _ = BoWField(min_freq=2, scale_factor=10)


def test_text_process_lower():
    field = TextField(lower=True)
    field.setup()

    dummy = "justo Praesent luctus justo praesent"
    assert list(field.process(dummy)) == [1, 1, 1, 1, 1]

    field.setup([dummy])
    assert list(field.process(dummy)) == [2, 3, 4, 2, 3]


def test_text_process_unk():
    field = TextField(unk_token=None)

    dummy = "justo Praesent luctus justo praesent"
    with pytest.raises(Exception):
        field.process(dummy)
