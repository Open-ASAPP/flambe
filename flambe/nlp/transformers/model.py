# type: ignore[override]

from typing import Optional, Any

import torch
from transformers import AutoModel

from flambe.nn import Module


class PretrainedTransformerEmbedder(Module):
    """Embedder intergation of the transformers library.

    Instantiate this object using any alias available in the
    `transformers` library. More information can be found here:

    https://huggingface.co/transformers/

    """

    def __init__(self,
                 alias: str,
                 cache_dir: Optional[str] = None,
                 padding_idx: Optional[int] = None,
                 pool: bool = False, **kwargs) -> None:
        """Initialize from a pretrained model.

        Parameters
        ----------
        alias: str
            Alias of a pretrained model.
        cache_dir: str, optional
            A directory where to cache the downloaded vocabularies.
        padding_idx: int, optional
            The padding index used to compute the attention mask.
        pool: optional, optional
            Whether to return the pooled output or the full sequence
            encoding. Default ``False``.

        """
        super().__init__()

        if 'gpt2' in alias and pool:
            raise ValueError('GPT2 does not support pooling.')

        embedder = AutoModel.from_pretrained(alias, cache_dir=cache_dir, **kwargs)
        self.config = embedder.config
        self._embedder = embedder
        self.padding_idx = padding_idx
        self.pool = pool

    def forward(self,
                data: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform a forward pass through the network.

        If pool was provided, will only return the pooled output
        of shape [B x H]. Otherwise, returns the full sequence encoding
        of shape [S x B x H].

        Parameters
        ----------
        data : torch.Tensor
            The input data of shape [B x S]
        token_type_ids : Optional[torch.Tensor], optional
            Segment token indices to indicate first and second portions
            of the inputs. Indices are selected in ``[0, 1]``: ``0``
            corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token. Has shape [B x S]
        attention_mask : Optional[torch.Tensor], optional
            FloatTensor of shape [B x S]. Masked values should
            be 0 for padding tokens, 1 otherwise.
        position_ids : Optional[torch.Tensor], optional
            Indices of positions of each input sequence tokens
            in the position embedding. Defaults to the order given
            in the input. Has shape [B x S].
        head_mask : Optional[torch.Tensor], optional
            Mask to nullify selected heads of the self-attention
            modules. Should be 0 for heads to mask, 1 otherwise.
            Has shape [num_layers x num_heads]

        Returns
        -------
        torch.Tensor
            If pool is True, returns a tneosr of shape [B x H],
            else returns an encoding for each token in the sequence
            of shape [B x S x H].

        """
        if attention_mask is None and self.padding_idx is not None:
            attention_mask = (data != self.padding_idx).float()

        outputs = self._embedder(data,
                                 token_type_ids=token_type_ids,
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 head_mask=head_mask)

        output = outputs[0] if not self.pool else outputs[1]
        return output

    def __getattr__(self, name: str) -> Any:
        """Override getattr to inspect config.

        Parameters
        ----------
        name : str
            The attribute to fetch

        Returns
        -------
        Any
            The attribute

        """
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            config = self.__dict__['config']
            if hasattr(config, name):
                return getattr(config, name)
            else:
                raise e
