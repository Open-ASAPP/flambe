from typing import Optional

import torch
from transformers import AutoTokenizer

from flambe.field import Field


class PretrainedTransformerField(Field):
    """Field intergation of the transformers library.

    Instantiate this object using any alias available in the
    `transformers` library. More information can be found here:

    https://huggingface.co/transformers/

    """

    def __init__(self,
                 alias: str,
                 cache_dir: Optional[str] = None,
                 max_len_truncate: Optional[int] = None,
                 add_special_tokens: bool = True, **kwargs) -> None:
        """Initialize a pretrained tokenizer.

        Parameters
        ----------
        alias: str
            Alias of a pretrained tokenizer.
        cache_dir: str, optional
            A directory where to cache the downloaded vocabularies.
        max_len_truncate: int, optional
            If given, truncates the length of the tokenized sequence.
        add_special_tokens: bool, optional
            Add the special tokens to the inputs. Default ``True``.

        """
        self._tokenizer = AutoTokenizer.from_pretrained(alias, cache_dir=cache_dir, **kwargs)
        self.max_len_truncate = max_len_truncate
        self.add_special_tokens = add_special_tokens

    @property
    def padding_idx(self) -> int:
        """Get the padding index.

        Returns
        -------
        int
            The padding index in the vocabulary

        """
        pad_token = self._tokenizer.pad_token
        return self._tokenizer.convert_tokens_to_ids(pad_token)

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary length.

        Returns
        -------
        int
            The length of the vocabulary

        """
        return len(self._tokenizer)

    def process(self, example: str) -> torch.Tensor:  # type: ignore
        """Process an example, and create a Tensor.

        Parameters
        ----------
        example: str
            The example to process, as a single string

        Returns
        -------
        torch.Tensor
            The processed example, tokenized and numericalized

        """
        tokens = self._tokenizer.encode(example, add_special_tokens=self.add_special_tokens)

        if self.max_len_truncate is not None:
            tokens = tokens[:self.max_len_truncate]

        return torch.tensor(tokens)
