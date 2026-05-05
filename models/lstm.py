"""
LSTM builder - דומה ל-RNN אבל עם nn.LSTM.
"""

import torch.nn as nn

from models.rnn import _SequenceClassifier


def build_lstm(hp: dict, data_info: dict) -> nn.Module:
    output_dim = data_info["output_dim"]
    vocab_size = data_info.get("vocab_size")

    hidden_size = int(hp.get("hidden_size", 128))
    num_layers = int(hp.get("num_layers", 1))
    bidirectional = bool(hp.get("bidirectional", False))
    dropout_p = float(hp.get("dropout", 0.0))
    embedding_dim = int(hp.get("embedding_dim", 128))

    if vocab_size is not None:
        embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        input_size = embedding_dim
    else:
        embedding = None
        input_size = data_info.get("input_dim", 1)

    rnn_dropout = dropout_p if num_layers > 1 else 0.0
    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=bidirectional,
        dropout=rnn_dropout,
    )
    return _SequenceClassifier(
        lstm, embedding, hidden_size, output_dim, dropout_p, bidirectional,
        embedding_dropout=float(hp.get("embedding_dropout", 0.0))
    )
