"""
Intergation of the pytorch_transformers openai and gpt2 modules.

Note that these objects are only to be used to load
pretrained models. The pytorch-transformers library
wasn't designed to train these models from scratch.

"""

import pytorch_transformers as pt

from flambe.nlp.transformers.utils import TransformerTextField, TransformerEmbedder


class GPTTextField(TransformerTextField):
    """Integrate the pytorch_transformers OpenAIGPTTokenizer.

    Currently available aliases:
        . `openai-gpt`

    """

    _cls = pt.OpenAIGPTTokenizer


class GPTEmbedder(TransformerEmbedder):
    """Integrate the pytorch_transformers OpenAIGPTmodel.

    Currently available aliases:
        . `openai-gpt`

    """

    _cls = pt.OpenAIGPTModel


class GPT2TextField(TransformerTextField):
    """Integrate the pytorch_transformers GPT2Tokenizer.

    Currently available aliases:
        . `gpt2`
        . `gpt2-medium`
        . `gpt2-large`

    """

    _cls = pt.GPT2Tokenizer


class GPT2Embedder(TransformerEmbedder):
    """Integrate the pytorch_transformers GPT2Model.

    Currently available aliases:
        . `gpt2`
        . `gpt2-medium`
        . `gpt2-large`

    """

    _cls = pt.GPT2Model
