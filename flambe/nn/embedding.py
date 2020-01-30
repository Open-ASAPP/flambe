# type: ignore[override]

import math
from typing import Tuple, Union, Optional

import torch
from torch import nn
from torch import Tensor

from flambe.compile import registrable_factory
from flambe.nn.module import Module


class Embeddings(Module):
    """Implement an Embeddings module.

    This object replicates the usage of nn.Embedding but
    registers the from_pretrained classmethod to be used inside
    a Flambé configuration, as this does not happen automatically
    during the registration of PyTorch objects.

    The module also adds optional positional encoding, which can
    either be sinusoidal or learned during training. For the
    non-learned positional embeddings, we use sine and cosine
    functions of different frequencies.

    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)

    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = 0,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 positional_encoding: bool = False,
                 positional_learned: bool = False,
                 positonal_max_length: int = 5000) -> None:
        """Initialize an Embeddings module.

        Parameters
        ----------
        num_embeddings : int
            Size of the dictionary of embeddings.
        embedding_dim : int
            The size of each embedding vector.
        padding_idx : int, optional
            Pads the output with the embedding vector at
            :attr:`padding_idx` (initialized to zeros) whenever it
            encounters the index, by default 0
        max_norm : Optional[float], optional
            If given, each embedding vector with norm larger than
            :attr:`max_norm` is normalized to have norm :attr:`max_norm`
        norm_type : float, optional
            The p of the p-norm to compute for the :attr:`max_norm`
            option. Default ``2``.
        scale_grad_by_freq : bool, optional
            If given, this will scale gradients by the inverse of
            frequency of the words in the mini-batch. Default ``False``.
        sparse : bool, optional
            If ``True``, gradient w.r.t. :attr:`weight` matrix will
            be a sparse tensor. See Notes for more details.
        positional_encoding : bool, optional
            If True, adds positonal encoding to the token embeddings.
            By default, the embeddings are frozen sinusodial embeddings.
            To learn these during training, set positional_learned.
            Default ``False``.
        positional_learned : bool, optional
            Learns the positional embeddings during training instead
            of using frozen sinusodial ones. Default ``False``.
        positonal_max_length : int, optional
            The maximum length of a sequence used for the positonal
            embedding matrix. Default ``5000``.

        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_positions = positonal_max_length

        self.token_embedding = nn.Embedding(num_embeddings,
                                            embedding_dim,
                                            padding_idx,
                                            max_norm,
                                            norm_type,
                                            scale_grad_by_freq,
                                            sparse)

        self.pos_embedding = None
        if positional_learned and not positional_encoding:
            raise ValueError("postional_encoding is False, but positonal_learned is True")

        elif positional_encoding and positional_learned:
            self.pos_embedding = nn.Embedding(positonal_max_length, embedding_dim)

        elif positional_encoding and not positional_learned:
            # Use sinusodial encoding
            position = torch.arange(0, positonal_max_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.arange(0, embedding_dim, 2).float()
            div_term = torch.exp(div_term * (-math.log(10000.0) / embedding_dim))

            pos_embedding = torch.zeros(positonal_max_length, embedding_dim)
            pos_embedding[:, 0::2] = torch.sin(position * div_term)
            pos_embedding[:, 1::2] = torch.cos(position * div_term)

            self.pos_embedding = nn.Embedding.from_pretrained(pos_embedding, freeze=True)

    @registrable_factory
    @classmethod
    def from_pretrained(cls,
                        embeddings: Tensor,
                        freeze: bool = True,
                        padding_idx: int = 0,
                        max_norm: Optional[float] = None,
                        norm_type: float = 2.0,
                        scale_grad_by_freq: bool = False,
                        sparse: bool = False,
                        positional_encoding: bool = False,
                        positional_learned: bool = False,
                        positonal_max_length: int = 5000,
                        positonal_embeddings: Optional[Tensor] = None,
                        positonal_freeze: bool = True):
        """Create an Embeddings instance from pretrained embeddings.

        Parameters
        ----------
        embeddings: torch.Tensor
            FloatTensor containing weights for the Embedding.
            First dimension is being passed to Embedding as
            num_embeddings, second as embedding_dim.
        freeze: bool
            If True, the tensor does not get updated in the learning
            process. Default: True
        padding_idx : int, optional
            Pads the output with the embedding vector at
            :attr:`padding_idx` (initialized to zeros) whenever it
            encounters the index, by default 0
        max_norm : Optional[float], optional
            If given, each embedding vector with norm larger than
            :attr:`max_norm` is normalized to have norm :attr:`max_norm`
        norm_type : float, optional
            The p of the p-norm to compute for the :attr:`max_norm`
            option. Default ``2``.
        scale_grad_by_freq : bool, optional
            If given, this will scale gradients by the inverse of
            frequency of the words in the mini-batch. Default ``False``.
        sparse : bool, optional
            If ``True``, gradient w.r.t. :attr:`weight` matrix will
            be a sparse tensor. See Notes for more details.
        positional_encoding : bool, optional
            If True, adds positonal encoding to the token embeddings.
            By default, the embeddings are frozen sinusodial embeddings.
            To learn these during training, set positional_learned.
            Default ``False``.
        positional_learned : bool, optional
            Learns the positional embeddings during training instead
            of using frozen sinusodial ones. Default ``False``.
        positonal_embeddings: torch.Tensor, optional
            If given, also replaces the positonal embeddings with
            this matrix. The max length will be ignored and replaced
            by the dimension of this matrix.
        positonal_freeze: bool, optional
            Whether the positonal embeddings should be frozen

        """
        if embeddings.dim() != 2:
            raise ValueError('Embeddings parameter is expected to be 2-dimensional')
        if positonal_embeddings is not None:
            if positonal_embeddings.dim() != 2:
                raise ValueError('Positonal embeddings parameter is expected to be 2-dimensional')
            if positonal_embeddings.size() != embeddings.size():
                raise ValueError('Both pretrained matrices must have the same dimensions')

        rows, cols = embeddings.shape
        positional_encoding = positional_encoding or (positonal_embeddings is not None)

        embedding = cls(num_embeddings=rows,
                        embedding_dim=cols,
                        padding_idx=padding_idx,
                        max_norm=max_norm,
                        norm_type=norm_type,
                        scale_grad_by_freq=scale_grad_by_freq,
                        sparse=sparse,
                        positional_encoding=positional_encoding,
                        positional_learned=positional_learned,
                        positonal_max_length=positonal_max_length)

        embedding.token_embedding.weight.data = embeddings
        embedding.token_embedding.weight.requires_grad = not freeze

        if positonal_embeddings is not None:
            embedding.pos_embedding.weight.data = positonal_embeddings  # type: ignore
            embedding.pos_embedding.weight.requires_grad = not positonal_freeze  # type: ignore

        return embedding

    def forward(self, data: Tensor) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        data : Tensor
            The input tensor of shape [S x B]

        Returns
        -------
        Tensor
            The output tensor of shape [S x B x E]

        """
        out = self.token_embedding(data)

        if self.pos_embedding is not None:
            column = torch.arange(data.size(0)).unsqueeze(1)
            positions = column.repeat(1, data.size(1)).to(data)
            out = out + self.pos_embedding(positions)

        return out


class Embedder(Module):
    """Implements an Embedder module.

    An Embedder takes as input a sequence of index tokens,
    and computes the corresponding embedded representations, and
    padding mask. The encoder may be initialized using a pretrained
    embedding matrix.

    Attributes
    ----------
    embeddings: Module
        The embedding module
    encoder: Module
        The sub-encoder that this object is wrapping
    pooling: Module
        An optional pooling module
    drop: nn.Dropout
        The dropout layer

    """

    def __init__(self,
                 embedding: Module,
                 encoder: Module,
                 pooling: Optional[Module] = None,
                 embedding_dropout: float = 0,
                 padding_idx: Optional[int] = 0) -> None:
        """Initializes the TextEncoder module.

        Extra arguments are passed to the nn.Embedding module.

        Parameters
        ----------
        embedding: nn.Embedding
            The embedding layer
        encoder: Module
            The encoder
        pooling: Module, optional
            An optioonal pooling module, takes a sequence of Tensor and
            reduces them to a single Tensor.
        embedding_dropout: float, optional
            Amount of dropout between the embeddings and the encoder
        padding_idx: int, optional
            Passed the nn.Embedding object. See pytorch documentation.

        """
        super().__init__()

        self.embedding = embedding
        self.dropout = nn.Dropout(embedding_dropout)
        self.encoder = encoder
        self.pooling = pooling
        self.padding_idx = padding_idx

    def forward(self, data: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor of shape [S x B]

        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor]]
            The encoded output, as a float tensor. May return a state
            if the encoder is an RNN and no pooling is provided.

        """
        embedded = self.embedding(data)
        embedded = self.dropout(embedded)

        padding_mask: Optional[Tensor]
        if self.padding_idx is not None:
            padding_mask = (data != self.padding_idx)
            encoding = self.encoder(embedded, padding_mask=padding_mask)
        else:
            padding_mask = None
            encoding = self.encoder(embedded)

        if self.pooling is not None:
            # Ignore states from encoders such as RNN or TransformerSRU
            encoding = encoding[0] if isinstance(encoding, tuple) else encoding
            encoding = self.pooling(encoding, padding_mask)

        return encoding
