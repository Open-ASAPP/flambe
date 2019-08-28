"""
Intergation of the pytorch_transformers transfo_xl module.

Note that these objects are only to be used to load
pretrained models. The pytorch-transformers library
wasn't designed to train these models from scratch.

"""

import pytorch_transformers as pt

from flambe.nlp.transformers.utils import TransformerTextField, TransformerEmbedder


class TransfoXLTextField(TransformerTextField):
    """Integrate the pytorch_transformers TransfoXLTokenizer.

    Currently available aliases:
        . `transfo-xl-wt103`

    """

    _cls = pt.TransfoXLTokenizer


class TransfoXLEmbedder(TransformerEmbedder):
    """Integrate the pytorch_transformers TransfoXLModel.

    Currently available aliases:
        . `transfo-xl-wt103`

    """

    _cls = pt.TransfoXLModel
