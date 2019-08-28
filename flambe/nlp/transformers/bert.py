"""
Intergation of the pytorch_transformers bert module.

Note that these objects are only to be used to load
pretrained models. The pytorch-transformers library
wasn't designed to train these models from scratch.

"""

import pytorch_transformers as pt

from flambe.nlp.transformers.utils import TransformerTextField, TransformerEmbedder


class BertTextField(TransformerTextField):
    """Integrate the pytorch_transformers BertTokenizer.

    Currently available aliases:
        . `bert-base-uncased`
        . `bert-large-uncased`
        . `bert-base-cased`
        . `bert-large-cased`
        . `bert-base-multilingual-uncased`
        . `bert-base-multilingual-cased`
        . `bert-base-chinese`
        . `bert-base-german-cased`
        . `bert-large-uncased-whole-word-masking`
        . `bert-large-cased-whole-word-masking`
        . `bert-large-uncased-whole-word-masking-finetuned-squad`
        . `bert-large-cased-whole-word-masking-finetuned-squad`
        . `bert-base-cased-finetuned-mrpc`

    """

    _cls = pt.BertTokenizer


class BertEmbedder(TransformerEmbedder):
    """Integrate the pytorch_transformers BertModel.

    Currently available aliases:
        . `bert-base-uncased`
        . `bert-large-uncased`
        . `bert-base-cased`
        . `bert-large-cased`
        . `bert-base-multilingual-uncased`
        . `bert-base-multilingual-cased`
        . `bert-base-chinese`
        . `bert-base-german-cased`
        . `bert-large-uncased-whole-word-masking`
        . `bert-large-cased-whole-word-masking`
        . `bert-large-uncased-whole-word-masking-finetuned-squad`
        . `bert-large-cased-whole-word-masking-finetuned-squad`
        . `bert-base-cased-finetuned-mrpc`

    """

    _cls = pt.BertModel
