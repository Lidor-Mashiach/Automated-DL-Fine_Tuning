"""
LSTM builder - either classification (default) or language modeling.

Language modeling mode adds:
  - Pretrained Word2Vec embeddings (loaded from data_info["embedding_matrix"])
  - Optional MIDI feature fusion (concatenate / project)
  - Per-timestep output (not just last hidden state)
"""

import torch
import torch.nn as nn

from models.rnn import _SequenceClassifier


def build_lstm(hp: dict, data_info: dict) -> nn.Module:
    task_type = data_info.get("task_type", "classification")
    if task_type == "language_modeling":
        return _build_lstm_lm(hp, data_info)
    return _build_lstm_classifier(hp, data_info)


def _build_lstm_classifier(hp: dict, data_info: dict) -> nn.Module:
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


def _build_lstm_lm(hp: dict, data_info: dict) -> nn.Module:
    """LSTM language model: Word2Vec + (optional MIDI) -> next-word logits."""
    vocab_size = int(data_info["vocab_size"])
    embedding_dim = int(data_info["embedding_dim"])
    midi_dim = int(data_info["midi_dim"])
    embedding_matrix = data_info.get("embedding_matrix")  # numpy or None

    hidden_size = int(hp.get("hidden_size", 256))
    num_layers = int(hp.get("num_layers", 1))
    dropout_p = float(hp.get("dropout", 0.3))
    fusion_method = hp.get("fusion_method", "concatenate")
    fusion_proj_dim = int(hp.get("fusion_proj_dim", 128))
    freeze_embeddings = bool(hp.get("freeze_embeddings", True))
    tie_weights = bool(hp.get("tie_weights", False))

    return _LSTMLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        midi_dim=midi_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout_p,
        fusion_method=fusion_method,
        fusion_proj_dim=fusion_proj_dim,
        embedding_matrix=embedding_matrix,
        freeze_embeddings=freeze_embeddings,
        embedding_dropout=float(hp.get("embedding_dropout", 0.0)),
        tie_weights=tie_weights,
    )


class _LSTMLanguageModel(nn.Module):
    """
    LSTM-based language model with optional MIDI feature fusion.

    Inputs:
      - word_ids: (B, T) LongTensor of vocab indices
      - midi_feats: (B, T, midi_dim) FloatTensor (or empty when midi_dim==0)

    Output:
      - logits: (B, T, vocab_size) - next-word logits at every timestep

    Fusion strategies:
      - "concatenate": LSTM input = [word_emb; midi_feats]
      - "project":     LSTM input = [word_emb_proj; midi_feats_proj]
                       Each is passed through a linear -> ReLU before concat.
    """

    def __init__(self, vocab_size, embedding_dim, midi_dim,
                 hidden_size, num_layers, dropout,
                 fusion_method, fusion_proj_dim,
                 embedding_matrix=None, freeze_embeddings=True,
                 embedding_dropout=0.0, tie_weights=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.midi_dim = midi_dim
        self.fusion_method = fusion_method

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.embedding_dropout = (
            nn.Dropout(embedding_dropout) if embedding_dropout > 0 else nn.Identity()
        )

        # Determine LSTM input size based on fusion
        if midi_dim == 0:
            lstm_input_dim = embedding_dim
        elif fusion_method == "concatenate":
            lstm_input_dim = embedding_dim + midi_dim
            self.word_proj = nn.Identity()
            self.midi_proj = nn.Identity()
        elif fusion_method == "project":
            self.word_proj = nn.Sequential(
                nn.Linear(embedding_dim, fusion_proj_dim), nn.ReLU()
            )
            self.midi_proj = nn.Sequential(
                nn.Linear(midi_dim, fusion_proj_dim), nn.ReLU()
            )
            lstm_input_dim = 2 * fusion_proj_dim
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_size, vocab_size)

        if tie_weights and hidden_size == embedding_dim:
            self.head.weight = self.embedding.weight

    def forward(self, word_ids, midi_feats=None):
        """
        word_ids: (B, T) Long
        midi_feats: (B, T, midi_dim) Float or None
        Returns: logits (B, T, vocab_size), hidden_state
        """
        emb = self.embedding(word_ids)
        emb = self.embedding_dropout(emb)

        if self.midi_dim == 0 or midi_feats is None or midi_feats.size(-1) == 0:
            lstm_in = emb
        elif self.fusion_method == "concatenate":
            lstm_in = torch.cat([emb, midi_feats], dim=-1)
        else:  # project
            w = self.word_proj(emb)
            m = self.midi_proj(midi_feats)
            lstm_in = torch.cat([w, m], dim=-1)

        out, hidden = self.lstm(lstm_in)
        out = self.dropout(out)
        logits = self.head(out)
        return logits, hidden

    def step(self, word_id, midi_feat, hidden=None):
        """
        Single-timestep forward for generation.
        word_id: (B,) Long
        midi_feat: (B, midi_dim) Float or None
        hidden: previous LSTM state
        Returns: logits (B, vocab_size), new_hidden
        """
        word_ids = word_id.unsqueeze(1)  # (B, 1)
        if midi_feat is not None and self.midi_dim > 0:
            midi_feats = midi_feat.unsqueeze(1)  # (B, 1, midi_dim)
        else:
            midi_feats = None

        emb = self.embedding(word_ids)
        emb = self.embedding_dropout(emb)

        if self.midi_dim == 0 or midi_feats is None:
            lstm_in = emb
        elif self.fusion_method == "concatenate":
            lstm_in = torch.cat([emb, midi_feats], dim=-1)
        else:
            w = self.word_proj(emb)
            m = self.midi_proj(midi_feats)
            lstm_in = torch.cat([w, m], dim=-1)

        out, new_hidden = self.lstm(lstm_in, hidden)
        out = self.dropout(out)
        logits = self.head(out[:, -1, :])  # (B, vocab_size)
        return logits, new_hidden

