"""
RNN builder - vanilla RNN לסיווג/רגרסיה של סדרות.
"""

import torch
import torch.nn as nn


class _SequenceClassifier(nn.Module):
    """עטיפה משותפת ל-RNN/LSTM - לוקחת את hidden state האחרון לסיווג."""

    def __init__(self, rnn: nn.Module, embedding: nn.Module | None,
                 hidden_size: int, output_dim: int, dropout: float,
                 bidirectional: bool):
        super().__init__()
        self.embedding = embedding
        self.rnn = rnn
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        direction_mult = 2 if bidirectional else 1
        self.head = nn.Linear(hidden_size * direction_mult, output_dim)

    def forward(self, x):
        # x: (B, T) לטקסט (token ids) או (B, T, F) לסדרות numeric.
        if self.embedding is not None:
            x = self.embedding(x)  # (B, T, E)
        # הוודא מימד batch-first
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, T, 1) - סדרה של סקלרים
        output, _ = self.rnn(x)  # output: (B, T, H*dir)
        last = output[:, -1, :]  # לקיחת ה-timestep האחרון
        return self.head(self.dropout(last))


def build_rnn(hp: dict, data_info: dict) -> nn.Module:
    """בונה RNN פשוט עם ראש סיווג."""
    output_dim = data_info["output_dim"]
    vocab_size = data_info.get("vocab_size")  # None למשימות נומריות

    hidden_size = int(hp.get("hidden_size", 128))
    num_layers = int(hp.get("num_layers", 1))
    bidirectional = bool(hp.get("bidirectional", False))
    dropout_p = float(hp.get("dropout", 0.0))
    embedding_dim = int(hp.get("embedding_dim", 128))

    # אם יש אוצר מילים (NLP) - שכבת embedding. אחרת input הוא numeric.
    if vocab_size is not None:
        embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        input_size = embedding_dim
    else:
        embedding = None
        input_size = data_info.get("input_dim", 1)

    # dropout ב-RNN של torch רלוונטי רק אם num_layers>1
    rnn_dropout = dropout_p if num_layers > 1 else 0.0
    rnn = nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=bidirectional,
        dropout=rnn_dropout,
        nonlinearity="tanh",
    )
    return _SequenceClassifier(
        rnn, embedding, hidden_size, output_dim, dropout_p, bidirectional
    )
