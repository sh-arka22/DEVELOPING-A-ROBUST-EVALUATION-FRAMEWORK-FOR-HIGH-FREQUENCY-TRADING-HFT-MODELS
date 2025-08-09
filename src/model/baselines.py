"""
Baselines: LSTM/GRU and Transformer encoders for HFT sequence modelling
-----------------------------------------------------------------------
PyTorch implementation (≈300 LOC, no other dependencies except torch>=2.0).
If you prefer TensorFlow, porting is ~1‑hour task.

Author: Week‑3 implementation
"""

# src/models/baselines.py  – LSTM/GRU & Transformer

from __future__ import annotations
import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# 1.  Recurrent baseline (LSTM/GRU) -----------------------------------
# ---------------------------------------------------------------------
class RNNBaseline(nn.Module):
    """
    Generic stacked RNN with optional attention head.

    Args
    ----
    input_dim     : number of input features per tick
    hidden_dim    : hidden size per layer
    num_layers    : number of stacked recurrent layers
    rnn_type      : "lstm" or "gru"
    dropout       : between‑layer dropout
    out_dim       : number of output units (e.g. 1 for regression, 2‑3 for class.)
    attention     : bool, if True add additive attention on top of last hidden seq
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.1,
        out_dim: int = 1,
        attention: bool = False,
    ):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.rnn = rnn_cls(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = attention
        if attention:
            self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v = nn.Parameter(torch.rand(hidden_dim))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _apply_attention(self, h):  # h: [B, T, H]
        u = torch.tanh(self.W(h))               # [B, T, H]
        scores = torch.matmul(u, self.v)        # [B, T]
        alpha = torch.softmax(scores, dim=1)    # [B, T]
        context = torch.sum(alpha.unsqueeze(-1) * h, dim=1)  # [B, H]
        return context

    def forward(self, x):        # x [B, T, F]
        h, _ = self.rnn(x)       # h [B, T, H]
        if self.attention:
            h_last = self._apply_attention(h)
        else:
            h_last = h[:, -1]    # last timestep
        return self.head(h_last)


# ---------------------------------------------------------------------
# 2.  Transformer encoder baseline ------------------------------------
# ---------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerBaseline(nn.Module):
    """
    Lightweight Transformer encoder with global average pooling.

    Args
    ----
    input_dim     : feature dimension
    d_model       : embedding size (will project input -> d_model)
    n_heads       : multi‑head attention heads
    num_layers    : encoder blocks
    dim_feedforward: FFN hidden dimension
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        out_dim: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x):             # [B, T, F]
        z = self.input_proj(x)        # [B, T, D]
        z = self.pos_enc(z)           # pos + dropout
        z = self.encoder(z)           # [B, T, D]
        z = z.mean(dim=1)             # global average
        return self.head(z)
