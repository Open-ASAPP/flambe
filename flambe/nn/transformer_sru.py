# type: ignore[override]

import copy
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from sru import SRUCell

from flambe.nn import Module


class TransformerSRU(Module):
    """A Transformer with an SRU replacing the FFN."""

    def __init__(self,
                 input_size: int = 512,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 sru_dropout: Optional[float] = None,
                 bidrectional: bool = False,
                 **kwargs: Dict[str, Any]) -> None:
        """Initialize the TransformerSRU Model.

        Parameters
        ----------
        input_size : int, optional
            dimension of embeddings (default=512). if different from
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
        sru_dropout: float, optional
            Dropout for the SRU cell. If not given, uses the same
            dropout value as the rest of the transformer.
        bidrectional: bool, optional
            Whether the SRU Encoder module should be bidrectional.
            Defaul ``False``.

        Extra keyword arguments are passed to the SRUCell.

        """
        super().__init__()

        self.encoder = TransformerSRUEncoder(input_size,
                                             d_model,
                                             nhead,
                                             dim_feedforward,
                                             num_encoder_layers,
                                             dropout,
                                             sru_dropout,
                                             bidrectional,
                                             **kwargs)

        self.decoder = TransformerSRUDecoder(input_size,
                                             d_model,
                                             nhead,
                                             dim_feedforward,
                                             num_encoder_layers,
                                             dropout,
                                             sru_dropout,
                                             **kwargs)

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
            shape: :math:`(N, S)`.
        tgt_key_padding_mask: torch.Tensor, optional
            the ByteTensor mask for tgt keys per batch (optional).
            shape: :math:`(N, T)`.
        memory_key_padding_mask: torch.Tensor, optional
            the ByteTensor mask for memory keys per batch (optional).
            shape" :math:`(N, S)`.

        Returns
        -------
        output: torch.Tensor
            The output sequence, shape: :math:`(T, N, E)`.

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

        memory, state = self.encoder(src,
                                     mask=src_mask,
                                     padding_mask=src_key_padding_mask)
        output = self.decoder(tgt,
                              memory,
                              state=state,
                              tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


class TransformerSRUEncoder(Module):
    """A TransformerSRUEncoder with an SRU replacing the FFN."""

    def __init__(self,
                 input_size: int = 512,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 sru_dropout: Optional[float] = None,
                 bidirectional: bool = False,
                 **kwargs: Dict[str, Any]) -> None:
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
        sru_dropout: float, optional
            Dropout for the SRU cell. If not given, uses the same
            dropout value as the rest of the transformer.
        bidirectional: bool
            Whether the SRU module should be bidrectional.
            Defaul ``False``.

        Extra keyword arguments are passed to the SRUCell.

        """
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model

        if input_size != d_model:
            self.proj = nn.Linear(input_size, d_model)

        layer = TransformerSRUEncoderLayer(d_model,
                                           nhead,
                                           dim_feedforward,
                                           dropout,
                                           sru_dropout,
                                           bidirectional)

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers

        self._reset_parameters()

    def forward(self,  # type: ignore
                src: torch.Tensor,
                state: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass the input through the endocder layers in turn.

        Parameters
        ----------
        src: torch.Tensor
            The sequnce to the encoder (required).
        state: Optional[torch.Tensor]
            Optional state from previous sequence encoding.
            Only passed to the SRU (not used to perform multihead
            attention).
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

        new_states = []
        for i in range(self.num_layers):
            input_state = state[i] if state is not None else None
            output, new_state = self.layers[i](output,
                                               state=input_state,
                                               src_mask=mask,
                                               padding_mask=padding_mask)
            new_states.append(new_state)

        new_states = torch.stack(new_states, dim=0)
        return output.transpose(0, 1), new_states

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerSRUDecoder(Module):
    """A TransformerSRUDecoderwith an SRU replacing the FFN."""

    def __init__(self,
                 input_size: int = 512,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 sru_dropout: Optional[float] = None,
                 **kwargs: Dict[str, Any]) -> None:
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
        sru_dropout: float, optional
            Dropout for the SRU cell. If not given, uses the same
            dropout value as the rest of the transformer.

        Extra keyword arguments are passed to the SRUCell.

        """
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model

        if input_size != d_model:
            self.proj = nn.Linear(input_size, d_model)

        layer = TransformerSRUDecoderLayer(d_model,
                                           nhead,
                                           dim_feedforward,
                                           dropout,
                                           sru_dropout)

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers

        self._reset_parameters()

    def forward(self,  # type: ignore
                tgt: torch.Tensor,
                memory: torch.Tensor,
                state: Optional[torch.Tensor] = None,
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
        state: Optional[torch.Tensor]
            Optional state from previous sequence encoding.
            Only passed to the SRU (not used to perform multihead
            attention).
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
        state = state or [None] * self.num_layers

        if self.input_size != self.d_model:
            output = self.proj(output)

        for i in range(self.num_layers):
            output = self.layers[i](output,
                                    memory,
                                    state=state[i],
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


class TransformerSRUEncoderLayer(Module):
    """A TransformerSRUEncoderLayer with an SRU replacing the FFN."""

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 sru_dropout: Optional[float] = None,
                 bidirectional: bool = False,
                 **kwargs: Dict[str, Any]) -> None:
        """Initialize a TransformerSRUEncoderLayer.

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
        sru_dropout: float, optional
            Dropout for the SRU cell. If not given, uses the same
            dropout value as the rest of the transformer.
        bidirectional: bool
            Whether the SRU module should be bidrectional.
            Defaul ``False``.

        Extra keyword arguments are passed to the SRUCell.

        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.sru = SRUCell(d_model,
                           dim_feedforward,
                           dropout,
                           sru_dropout or dropout,
                           bidirectional=bidirectional,
                           has_skip_term=False, **kwargs)

        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,  # type: ignore
                src: torch.Tensor,
                state: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass the input through the endocder layer.

        Parameters
        ----------
        src: torch.Tensor
            The sequence to the encoder layer (required).
        state: Optional[torch.Tensor]
            Optional state from previous sequence encoding.
            Only passed to the SRU (not used to perform multihead
            attention).
        src_mask: torch.Tensor, optional
            The mask for the src sequence (optional).
        padding_mask: torch.Tensor, optional
            The mask for the src keys per batch (optional).
            Should be True for tokens to leave untouched, and False
            for padding tokens.

        Returns
        -------
        torch.Tensor
            Output Tensor of shape [S x B x H]
        torch.Tensor
            Output state of the SRU of shape [N x B x H]

        """
        # Transpose and reverse
        reversed_mask = None
        if padding_mask is not None:
            reversed_mask = ~padding_mask

        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=reversed_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2, state = self.sru(src, state, mask_pad=padding_mask)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, state


class TransformerSRUDecoderLayer(Module):
    """A TransformerSRUDecoderLayer with an SRU replacing the FFN."""

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 sru_dropout: Optional[float] = None,
                 **kwargs: Dict[str, Any]) -> None:
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
        sru_dropout: float, optional
            Dropout for the SRU cell. If not given, uses the same
            dropout value as the rest of the transformer.

        Extra keyword arguments are passed to the SRUCell.

        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.sru = SRUCell(d_model,
                           dim_feedforward,
                           dropout,
                           sru_dropout or dropout,
                           bidirectional=False,
                           has_skip_term=False, **kwargs)

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
                state: Optional[torch.Tensor] = None,
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
            The sequence from the last layer of the encoder (required).
        state: Optional[torch.Tensor]
            Optional state from previous sequence encoding.
            Only passed to the SRU (not used to perform multihead
            attention).
        tgt_mask: torch.Tensor, optional
            The mask for the tgt sequence (optional).
        memory_mask: torch.Tensor, optional
            the mask for the memory sequence (optional).
        padding_mask: torch.Tensor, optional
            the mask for the tgt keys per batch (optional).
        memory_key_padding_mask: torch.Tensor, optional
            the mask for the memory keys per batch (optional).

        Returns
        -------
        torch.Tensor
            Output Tensor of shape [S x B x H]

        """
        # Transpose and reverse
        reversed_mask = None
        if padding_mask is not None:
            reversed_mask = ~padding_mask

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=reversed_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2, _ = self.sru(tgt, state, mask_pad=padding_mask)
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
