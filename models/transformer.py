"""
Transformer builder - Transformer encoder פשוט עם pooling ו-FC head.
"""

import math

import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    """Positional encoding סטנדרטי (sinusoidal)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class _DropPath(nn.Module):
    """Stochastic depth: randomly drop entire residual paths during training.

    Used to wrap each transformer encoder layer. When the path is "dropped",
    the input passes through unchanged (identity skip).
    """

    def __init__(self, layer: nn.Module, drop_prob: float = 0.0):
        super().__init__()
        self.layer = layer
        self.drop_prob = float(drop_prob)

    def forward(self, x, *args, **kwargs):
        if self.training and self.drop_prob > 0:
            if torch.rand(1).item() < self.drop_prob:
                # Drop the path: return input unchanged
                return x
        return self.layer(x, *args, **kwargs)


class _StochasticDepthEncoder(nn.Module):
    """Stack of transformer encoder layers, each wrapped in DropPath.
    Replacement for nn.TransformerEncoder when stochastic depth is active.
    """

    def __init__(self, wrapped_layers):
        super().__init__()
        self.layers = nn.ModuleList(wrapped_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _TransformerClassifier(nn.Module):
    """Transformer encoder + mean pooling + linear output head."""

    def __init__(
        self,
        input_projection: nn.Module,
        encoder: nn.TransformerEncoder,
        pos_encoding: _PositionalEncoding,
        d_model: int,
        output_dim: int,
        dropout: float,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_projection = input_projection
        self.embedding_dropout = (
            nn.Dropout(embedding_dropout) if embedding_dropout > 0 else nn.Identity()
        )
        self.pos_encoding = pos_encoding
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = self.input_projection(x)
        else:
            x = self.input_projection(x)
        x = self.embedding_dropout(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(self.dropout(x))


def build_transformer(hp: dict, data_info: dict) -> nn.Module:
    output_dim = data_info["output_dim"]
    vocab_size = data_info.get("vocab_size")

    d_model = int(hp.get("d_model", 128))
    nhead = int(hp.get("nhead", 4))
    num_layers = int(hp.get("num_encoder_layers", 3))
    dim_ff = int(hp.get("dim_feedforward", 512))
    dropout_p = float(hp.get("dropout", 0.1))
    attn_dropout = float(hp.get("attention_dropout", 0.0))
    activation = hp.get("activation", "relu")

    # חשוב: d_model חייב להתחלק ב-nhead. אם ה-trial הגריל צירוף לא חוקי -
    # מתקנים כאן (במקום להיכשל). זו הגנה רכה; הקונפיג מנסה ממילא לא להציע
    # אפשרויות גרועות.
    if d_model % nhead != 0:
        # בוחרים את ה-nhead הגדול ביותר שמחלק את d_model
        candidates = [h for h in (1, 2, 4, 8, 16) if d_model % h == 0]
        nhead = max(candidates) if candidates else 1

    # projection של הקלט ל-d_model
    if vocab_size is not None:
        input_projection = nn.Embedding(vocab_size, d_model, padding_idx=0)
    else:
        input_dim = data_info.get("input_dim", 1)
        # wrapper לקלט 2D -> נבצע projection לינארי ברמת כל timestep
        input_projection = nn.Linear(input_dim, d_model)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_ff,
        dropout=dropout_p,
        activation=activation,
        batch_first=True,
    )
    _ = attn_dropout  # not exposed separately by torch's encoder layer

    # Stochastic depth: only meaningful for deep transformers (>= 4 layers).
    # Wrap each encoder layer with a DropPath that randomly skips it.
    stoch_depth = float(hp.get("stochastic_depth", 0.0))
    if stoch_depth > 0 and num_layers >= 4:
        # Build encoder layers manually so we can wrap them
        from copy import deepcopy
        layers = []
        for i in range(num_layers):
            # Linear scaling: deeper layers get higher drop prob
            layer_drop = stoch_depth * (i / max(1, num_layers - 1))
            wrapped = _DropPath(deepcopy(encoder_layer), drop_prob=layer_drop)
            layers.append(wrapped)
        encoder = _StochasticDepthEncoder(layers)
    else:
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    pos = _PositionalEncoding(d_model)

    return _TransformerClassifier(
        input_projection, encoder, pos, d_model, output_dim, dropout_p,
        embedding_dropout=float(hp.get("embedding_dropout", 0.0))
    )
