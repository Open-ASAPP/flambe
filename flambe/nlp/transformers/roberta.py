"""
Intergation of the pytorch_transformers roberta module.

Note that these objects are only to be used to load
pretrained models. The pytorch-transformers library
wasn't designed to train these models from scratch.

"""

import pytorch_transformers as pt

from flambe.nlp.transformers.utils import TransformerTextField, TransformerEmbedder


class RobertaTextField(TransformerTextField):
    """Integrate the pytorch_transformers RobertaTokenizer.

    Currently available aliases:
        . `roberta-base`
        . `roberta-large`
        . `roberta-large-mnli`

    """

    _cls = pt.RobertaTokenizer


class RobertaEmbedder(TransformerEmbedder):
    """Integrate the pytorch_transformers RobertaModel.

    Currently available aliases:
        . `roberta-base`
        . `roberta-large`
        . `roberta-large-mnli`

    """

    _cls = pt.RobertaModel
