"""
Intergation of the pytorch_transformers xlnet module.

Note that these objects are only to be used to load
pretrained models. The pytorch-transformers library
wasn't designed to train these models from scratch.

"""

import pytorch_transformers as pt

from flambe.nlp.transformers.utils import TransformerTextField, TransformerEmbedder


class XLNetTextField(TransformerTextField):
    """Integrate the pytorch_transformers XLNetTokenizer.

    Currently available aliases:
        . `xlnet-base-cased`
        . `xlnet-large-cased`

    """

    _cls = pt.XLNetTokenizer


class XLNetEmbedder(TransformerEmbedder):
    """Integrate the pytorch_transformers XLNetModel.

    Currently available aliases:
        . `xlnet-base-cased`
        . `xlnet-large-cased`

    """

    _cls = pt.XLNetModel
