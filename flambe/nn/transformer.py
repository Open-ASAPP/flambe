# type: ignore[override]

"""
Code taken from the PyTorch source code. Slightly modified to improve
the interface to the TransformerEncoder, and TransformerDecoder modules.

"""
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flambe.nn import Module


class Transformer(Module):
    """A Transformer model

    User is able to modify the attributes as needed. The architechture
    is based on the paper "Attention Is All You Need". Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
    Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    """

    def __init__(self,
                 input_size,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1) -> None:
        """Initialize the Transformer Model.

        Parameters
        ----------
        input_size : int, optional
            dimension of embeddings. If different from
            d_model, then a linear layer is added to project from
            input_size to d_model.
        d_model : int, optional
            the number of expected features in the
            encoder/decoder inputs (default=512).
        nhead : int, optional
            the number of heads in the multiheadattention
            models (default=8).
        num_encoder_layers : int, optional
            the number of sub-encoder-layers in the encoder
            (default=6).
        num_decoder_layers : int, optional
            the number of sub-decoder-layers in the decoder
            (default=6).
        dim_feedforward : int, optional
            the dimension of the feedforward network model
            (default=2048).
        dropout : float, optional
            the dropout value (default=0.1).

        """
        super().__init__()

        self.encoder = TransformerEncoder(input_size,
                                          d_model,
                                          nhead,
                                          dim_feedforward,
                                          num_encoder_layers,
                                          dropout)

        self.decoder = TransformerDecoder(input_size,
                                          d_model,
                                          nhead,
                                          dim_feedforward,
                                          num_encoder_layers,
                                          dropout)

    def forward(self,  # type: ignore
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Take in and process masked source/target sequences.

        Parameters
        ----------
        src: torch.Tensor
            the sequence to the encoder (required).
            shape: :math:`(N, S, E)`.
        tgt: torch.Tensor
            the sequence to the decoder (required).
            shape: :math:`(N, T, E)`.
        src_mask: torch.Tensor, optional
            the additive mask for the src sequence (optional).
            shape: :math:`(S, S)`.
        tgt_mask: torch.Tensor, optional
            the additive mask for the tgt sequence (optional).
            shape: :math:`(T, T)`.
        memory_mask: torch.Tensor, optional
            the additive mask for the encoder output (optional).
            shape: :math:`(T, S)`.
        src_key_padding_mask: torch.Tensor, optional
            the ByteTensor mask for src keys per batch (optional).
            shape: :math:`(N, S)`
        tgt_key_padding_mask: torch.Tensor, optional
            the ByteTensor mask for tgt keys per batch (optional).
            shape: :math:`(N, T)`.
        memory_key_padding_mask: torch.Tensor, optional
            the ByteTensor mask for memory keys per batch (optional).
            shape" :math:`(N, S)`.

        Returns
        -------
        output: torch.Tensor
            The output sequence, shape: :math:`(N, T, E)`.

        Note: [src/tgt/memory]_mask should be filled with
            float('-inf') for the masked positions and float(0.0) else.
            These masks ensure that predictions for position i depend
            only on the unmasked positions j and are applied identically
            for each sequence in a batch.
            [src/tgt/memory]_key_padding_mask should be a ByteTensor
            where False values are positions that should be masked with
            float('-inf') and True values will be unchanged.
            This mask ensures that no information will be taken from
            position i if it is masked, and has a separate mask for each
            sequence in a batch.
        Note: Due to the multi-head attention architecture in the
            transformer model, the output sequence length of a
            transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target
            sequence length, N is the batchsize, E is the feature number

        """
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src,
                              mask=src_mask,
                              padding_mask=src_key_padding_mask)
        output = self.decoder(tgt,
                              memory,
                              tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


class TransformerEncoder(Module):
    """TransformerEncoder is a stack of N encoder layers."""

    def __init__(self,
                 input_size: int = 512,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1) -> None:
        """Initialize the TransformerEncoder.

        Parameters
        ---------
        input_size : int
            The embedding dimension of the model.  If different from
            d_model, a linear projection layer is added.
        d_model : int
            the number of expected features in encoder/decoder inputs.
            Default ``512``.
        nhead : int, optional
            the number of heads in the multiheadattention
            Default ``8``.
        num_layers : int
            the number of sub-encoder-layers in the encoder (required).
            Default ``6``.
        dim_feedforward : int, optional
            the inner feedforard dimension. Default ``2048``.
        dropout : float, optional
            the dropout percentage. Default ``0.1``.

        """
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model

        if input_size != d_model:
            self.proj = nn.Linear(input_size, d_model)

        layer = TransformerEncoderLayer(d_model,
                                        nhead,
                                        dim_feedforward,
                                        dropout)

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers

        self._reset_parameters()

    def forward(self,  # type: ignore
                src: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pass the input through the endocder layers in turn.

        Parameters
        ----------
        src: torch.Tensor
            The sequence to the encoder (required).
        memory: torch.Tensor, optional
            Optional memory, unused by default.
        mask: torch.Tensor, optional
            The mask for the src sequence (optional).
        padding_mask: torch.Tensor, optional
            The mask for the src keys per batch (optional).
            Should be True for tokens to leave untouched, and False
            for padding tokens.

        """
        output = src.transpose(0, 1)

        if self.input_size != self.d_model:
            output = self.proj(output)

        for i in range(self.num_layers):
            output = self.layers[i](output,
                                    memory=memory,
                                    src_mask=mask,
                                    padding_mask=padding_mask)

        return output.transpose(0, 1)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerDecoder(Module):
    """TransformerDecoder is a stack of N decoder layers"""

    def __init__(self,
                 input_size: int,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1) -> None:
        """Initialize the TransformerDecoder.

        Parameters
        ---------
        input_size : int
            The embedding dimension of the model.  If different from
            d_model, a linear projection layer is added.
        d_model : int
            The number of expected features in encoder/decoder inputs.
        nhead : int, optional
            The number of heads in the multiheadattention.
        num_layers : int
            The number of sub-encoder-layers in the encoder (required).
        dim_feedforward : int, optional
            The inner feedforard dimension, by default 2048.
        dropout : float, optional
            The dropout percentage, by default 0.1.

        """
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model

        if input_size != d_model:
            self.proj = nn.Linear(input_size, d_model)

        layer = TransformerDecoderLayer(d_model,
                                        nhead,
                                        dim_feedforward,
                                        dropout)

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers

        self._reset_parameters()

    def forward(self,  # type: ignore
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer in turn.

        Parameters
        ----------
        tgt: torch.Tensor
            The sequence to the decoder (required).
        memory: torch.Tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask: torch.Tensor, optional
            The mask for the tgt sequence (optional).
        memory_mask: torch.Tensor, optional
            The mask for the memory sequence (optional).
        padding_mask: torch.Tensor, optional
            The mask for the tgt keys per batch (optional).
            Should be True for tokens to leave untouched, and False
            for padding tokens.
        memory_key_padding_mask: torch.Tensor, optional
            The mask for the memory keys per batch (optional).

        Returns
        -------
        torch.Tensor

        """
        output = tgt.transpose(0, 1)

        if self.input_size != self.d_model:
            output = self.proj(output)

        for i in range(self.num_layers):
            output = self.layers[i](output,
                                    memory,
                                    tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    padding_mask=padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)

        return output.transpose(0, 1)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerEncoderLayer(Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward.

    This standard encoder layer is based on the paper "Attention Is
    All You Need". Ashish Vaswani, Noam Shazeer, Niki Parmar,
    Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may
    modify or implement in a different way during application.

    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1) -> None:
        """Initialize a TransformerEncoderLayer.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input.
        n_head : int
            The number of heads in the multiheadattention models.
        dim_feedforward : int, optional
            The dimension of the feedforward network (default=2048).
        dropout : float, optional
            The dropout value (default=0.1).

        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,  # type: ignore
                src: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pass the input through the endocder layer.

        Parameters
        ----------
        src: torch.Tensor
            The seqeunce to the encoder layer (required).
        memory: torch.Tensor, optional
            Optional memory from previous sequence, unused by default.
        src_mask: torch.Tensor, optional
            The mask for the src sequence (optional).
        padding_mask: torch.Tensor, optional
            The mask for the src keys per batch (optional).
            Should be True for tokens to leave untouched, and False
            for padding tokens.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B x S x H]

        """
        # Transpose and reverse
        if padding_mask is not None:
            padding_mask = ~padding_mask.bool()

        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(Module):
    """A TransformerDecoderLayer.

    A TransformerDecoderLayer is made up of self-attn, multi-head-attn
    and feedforward network. This standard decoder layer is based on the
    paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz
    Kaiser, and Illia Polosukhin. 2017. Attention is all you need.
    In Advances in Neural Information Processing Systems,
    pages 6000-6010. Users may modify or implement in a different way
    during application.

    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1) -> None:
        """Initialize a TransformerDecoder.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input.
        n_head : int
            The number of heads in the multiheadattention models.
        dim_feedforward : int, optional
            The dimension of the feedforward network (default=2048).
        dropout : float, optional
            The dropout value (default=0.1).

        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,  # type: ignore
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Parameters
        ----------
        tgt: torch.Tensor
            The sequence to the decoder layer (required).
        memory: torch.Tensor
            The sequnce from the last layer of the encoder (required).
        tgt_mask: torch.Tensor, optional
            The mask for the tgt sequence (optional).
        memory_mask: torch.Tensor, optional
            the mask for the memory sequence (optional).
        padding_mask: torch.Tensor, optional
            the mask for the tgt keys per batch (optional).
            Should be True for tokens to leave untouched, and False
            for padding tokens.
        memory_key_padding_mask: torch.Tensor, optional
            the mask for the memory keys per batch (optional).

        Returns
        -------
        torch.Tensor
            Output tensor of shape [T x B x H]

        """
        # Transpose anr reverse
        if padding_mask is not None:
            padding_mask = ~padding_mask

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


def generate_square_subsequent_mask(self, sz):
    r"""Generate a square mask for the sequence.

    The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).t()
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
