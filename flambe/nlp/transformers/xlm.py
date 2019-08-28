"""
Intergation of the pytorch_transformers xlm module.

Note that these objects are only to be used to load
pretrained models. The pytorch-transformers library
wasn't designed to train these models from scratch.

"""

import pytorch_transformers as pt

from flambe.nlp.transformers.utils import TransformerTextField, TransformerEmbedder


class XLMTextField(TransformerTextField):
    """Integrate the pytorch_transformers XLMTokenizer.

    Currently available aliases:
        . `xlm-mlm-en-2048`
        . `xlm-mlm-ende-1024`
        . `xlm-mlm-enfr-1024`
        . `xlm-mlm-enro-1024`
        . `xlm-mlm-tlm-xnli15-1024`
        . `xlm-mlm-xnli15-1024`
        . `xlm-clm-enfr-1024`
        . `xlm-clm-ende-1024`

    """

    _cls = pt.XLMTokenizer


class XLMEmbedder(TransformerEmbedder):
    """Integrate the pytorch_transformers XLMModel.

    Currently available aliases:
        . `xlm-mlm-en-2048`
        . `xlm-mlm-ende-1024`
        . `xlm-mlm-enfr-1024`
        . `xlm-mlm-enro-1024`
        . `xlm-mlm-tlm-xnli15-1024`
        . `xlm-mlm-xnli15-1024`
        . `xlm-clm-enfr-1024`
        . `xlm-clm-ende-1024`

    """

    _cls = pt.XLMModel
