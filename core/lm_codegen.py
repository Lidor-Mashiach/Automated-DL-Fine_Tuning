"""
lm_codegen.py
-------------
Generator for a standalone, runnable model.py file for language-modeling runs.

Unlike the other architecture builders (which inline tiny hyperparameter sets
and small models), the LM model.py is built around external file dependencies
(CSV lyrics, Word2Vec embeddings, MIDI files). To keep it usable:

  - All paths live as global constants at the top of the generated file.
  - A `data/` subfolder is created next to model.py with copies of the inputs.
  - The user can run as-is, swap files in `data/`, or edit a path constant.

The generated model.py:
  1. Builds a vocabulary from the CSV.
  2. Loads Word2Vec embeddings.
  3. Extracts MIDI features (variant-driven).
  4. Defines the LSTM language model with fusion (concatenate / project).
  5. Trains using the best hyperparameters found by AutoTune-NN.
  6. Generates lyrics on the test set with the chosen sampling strategy.

It does NOT depend on the autotune_nn package - it is fully self-contained.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def _populate_lm_data_dir(data_dir: Path, cfg) -> None:
    """
    Copy the source data files next to model.py so the generated script is
    self-contained.

    Copies (or symlinks if too large):
      - The lyrics CSV (always).
      - The MIDI directory (always, even if empty - baseline runs use it).
      - The Word2Vec file (only if < 500 MB to avoid blowing up the run dir).
        For larger files, leaves a NOTE.txt with the original path.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Lyrics CSV
    if cfg.local_dataset_path:
        src = Path(cfg.local_dataset_path)
        if src.exists() and src.is_file():
            try:
                shutil.copy2(src, data_dir / src.name)
            except Exception as e:
                _write_note(data_dir, f"lyrics CSV at {src} (copy failed: {e})")

    # MIDI directory
    if cfg.midi_dir:
        src = Path(cfg.midi_dir)
        if src.exists() and src.is_dir():
            dest = data_dir / "midi_files"
            try:
                if not dest.exists():
                    shutil.copytree(src, dest)
            except Exception as e:
                _write_note(data_dir, f"MIDI directory at {src} (copy failed: {e})")

    # Word2Vec - only copy if small enough
    if cfg.word2vec_path:
        src = Path(cfg.word2vec_path)
        if src.exists() and src.is_file():
            size_mb = src.stat().st_size / (1024 * 1024)
            if size_mb < 500:
                try:
                    shutil.copy2(src, data_dir / src.name)
                except Exception as e:
                    _write_note(data_dir, f"Word2Vec at {src} (copy failed: {e})")
            else:
                _write_note(
                    data_dir,
                    f"Word2Vec file too large to copy ({size_mb:.0f} MB): {src}\n"
                    f"Edit the WORD2VEC_PATH constant in model.py to point to it,\n"
                    f"or place a copy at {data_dir}/{src.name}",
                )


def _write_note(data_dir: Path, msg: str) -> None:
    note = data_dir / "NOTE.txt"
    with open(note, "a", encoding="utf-8") as f:
        f.write(msg + "\n\n")


def _hp_value(hp: dict, key: str, default):
    """Safely fetch a hyperparameter, returning a Python literal."""
    return hp.get(key, default)


def _generate_lm_model_py(*, cfg, hp, data_info,
                           total_trials, best_quality,
                           best_metric_raw, best_metric_smoothed,
                           test_metric, test_loss,
                           smoothing_window, seed) -> str:
    """Return the full Python source of a standalone LM model.py."""

    # Resolve relative filenames so the generated model.py looks for them
    # under `./data/...` (next to itself). User can override the global vars
    # to point elsewhere.
    lyrics_csv_name = Path(cfg.local_dataset_path).name if cfg.local_dataset_path else "lyrics.csv"
    midi_dir_name = "midi_files"  # always this name inside data/
    w2v_name = ""
    if cfg.word2vec_path:
        w2v_name = Path(cfg.word2vec_path).name

    # Hyperparameters that the generated model.py needs
    hp_str = (
        f"HIDDEN_SIZE          = {int(_hp_value(hp, 'hidden_size', 256))}\n"
        f"NUM_LAYERS           = {int(_hp_value(hp, 'num_layers', 2))}\n"
        f"DROPOUT              = {float(_hp_value(hp, 'dropout', 0.3))}\n"
        f"EMBEDDING_DROPOUT    = {float(_hp_value(hp, 'embedding_dropout', 0.0))}\n"
        f"FUSION_METHOD        = {repr(_hp_value(hp, 'fusion_method', 'concatenate'))}\n"
        f"FUSION_PROJ_DIM      = {int(_hp_value(hp, 'fusion_proj_dim', 128))}\n"
        f"FREEZE_EMBEDDINGS    = {bool(_hp_value(hp, 'freeze_embeddings', True))}\n"
        f"TIE_WEIGHTS          = {bool(_hp_value(hp, 'tie_weights', False))}\n"
        f"LEARNING_RATE        = {float(_hp_value(hp, 'learning_rate', 1e-3))}\n"
        f"WEIGHT_DECAY         = {float(_hp_value(hp, 'weight_decay', 0.0))}\n"
        f"BATCH_SIZE           = {int(_hp_value(hp, 'batch_size', 64))}\n"
        f"EPOCHS               = {int(_hp_value(hp, 'epochs', 30))}\n"
        f"SEQ_LEN              = {int(_hp_value(hp, 'sequence_length', 32))}\n"
        f"GRADIENT_CLIPPING    = {float(_hp_value(hp, 'gradient_clipping', 1.0))}\n"
        f"OPTIMIZER_NAME       = {repr(_hp_value(hp, 'optimizer_name', 'adam'))}\n"
        f"LABEL_SMOOTHING      = {float(_hp_value(hp, 'label_smoothing', 0.0))}\n"
    )

    # Sampling settings
    sampling_str = (
        f"SAMPLING_STRATEGY    = {repr(cfg.sampling_strategy)}\n"
        f"SAMPLING_TEMPERATURE = {float(cfg.sampling_temperature)}\n"
        f"SAMPLING_TOP_K       = {int(cfg.sampling_top_k)}\n"
        f"SAMPLING_TOP_P       = {float(cfg.sampling_top_p)}\n"
        f"MAX_GENERATED_WORDS  = {int(cfg.max_generated_words)}\n"
        f"INITIAL_WORDS        = {cfg.initial_words or ['love', 'the', 'i']!r}\n"
    )

    return _LM_TEMPLATE.format(
        architecture=cfg.architecture.upper(),
        run_name=cfg.run_name,
        total_trials=total_trials,
        best_quality=best_quality,
        best_metric_smoothed=best_metric_smoothed,
        best_metric_raw=best_metric_raw,
        test_metric=test_metric,
        test_loss=test_loss,
        smoothing_window=smoothing_window,
        seed=seed,
        embedding_dim=int(data_info.get("embedding_dim", 300)),
        midi_variant=cfg.midi_variant,
        line_separator_token=cfg.line_separator_token,
        min_word_count=cfg.min_word_count,
        lyrics_csv_name=lyrics_csv_name,
        midi_dir_name=midi_dir_name,
        w2v_name=w2v_name,
        hp_block=hp_str,
        sampling_block=sampling_str,
        melody_probe=bool(cfg.melody_probe),
    )


# ============================================================================
# The full LM model.py template - this is what gets written to disk.
# ============================================================================
_LM_TEMPLATE = '''"""
================================================================================
 AutoTune-NN  |  Best Trial Deliverable  |  Language Modeling ({architecture})
================================================================================

Run name              : {run_name}
Total trials explored : {total_trials}
Best quality score    : {best_quality:.4f}
Best val metric       : {best_metric_smoothed:.4f}  (smoothed over {smoothing_window} epochs)
Best val metric (raw) : {best_metric_raw:.4f}
Final test loss       : {test_loss:.4f}

HOW TO USE
----------
This file is a self-contained training + generation script.

  1. The `data/` subfolder (next to this file) contains your inputs:
       - {lyrics_csv_name}    : lyrics CSV
       - {midi_dir_name}/     : MIDI files (optional)
       - {w2v_name}           : Word2Vec embeddings (optional - falls back to
                                random init if absent or too large to copy)

  2. To re-train with the best hyperparameters:
         python model.py

  3. To swap data, place new files under data/ (same names) OR edit the
     path constants in the GLOBAL CONFIGURATION section below.

  4. If model_checkpoint.pt is present, training is skipped and the script
     proceeds straight to generation. Delete it to force retraining.

  5. The generated lyrics are written to generated_lyrics.txt (this folder).
================================================================================
"""

from __future__ import annotations

import csv
import os
import re
import sys
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# GLOBAL CONFIGURATION  -  edit these to point at different files
# =============================================================================

# Paths are resolved relative to this file unless absolute.
_HERE = Path(__file__).resolve().parent
_DATA = _HERE / "data"

LYRICS_CSV_PATH = _DATA / "{lyrics_csv_name}"
MIDI_DIR        = _DATA / "{midi_dir_name}"
WORD2VEC_PATH   = _DATA / "{w2v_name}" if "{w2v_name}" else None

CHECKPOINT_PATH = _HERE / "model_checkpoint.pt"
OUTPUT_LYRICS_PATH = _HERE / "generated_lyrics.txt"
PROBE_PATH = _HERE / "melody_probe.json"

# Data layout
MIDI_VARIANT          = "{midi_variant}"     # "none" | "simple" | "per_word"
LINE_SEPARATOR_TOKEN  = "{line_separator_token}"
MIN_WORD_COUNT        = {min_word_count}
EMBEDDING_DIM         = {embedding_dim}

# Train/Val split (test = the songs reserved by AutoTune-NN)
VALIDATION_SPLIT = 0.2

# Best hyperparameters from AutoTune-NN
{hp_block}

# Generation settings
{sampling_block}

# Whether to run the melody-influence probe at generation time
MELODY_PROBE = {melody_probe}

SEED   = {seed}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]

# MIDI feature dims (must match training-time choices)
MIDI_SIMPLE_DIM = 8
MIDI_PER_WORD_DIM = 8


# =============================================================================
# VOCABULARY
# =============================================================================

class Vocabulary:
    def __init__(self, tokens, min_count=2, line_separator_token="&"):
        self.line_separator_token = line_separator_token
        counts = Counter(tokens)
        keep = [t for t, c in counts.most_common() if c >= min_count]
        if line_separator_token in counts and line_separator_token not in keep:
            keep.append(line_separator_token)
        self.itos = list(SPECIAL_TOKENS) + [t for t in keep if t not in SPECIAL_TOKENS]
        self.stoi = {{t: i for i, t in enumerate(self.itos)}}
        self.pad_idx = self.stoi[PAD_TOKEN]
        self.unk_idx = self.stoi[UNK_TOKEN]
        self.eos_idx = self.stoi[EOS_TOKEN]
        self.line_idx = self.stoi.get(line_separator_token, self.unk_idx)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, self.unk_idx) for t in tokens]


# =============================================================================
# WORD2VEC
# =============================================================================

def load_word2vec(path, vocab, dim):
    """Load pretrained embeddings; fall back to random init if missing."""
    matrix = np.random.uniform(-0.05, 0.05, (len(vocab), dim)).astype("float32")
    matrix[vocab.pad_idx] = 0.0

    if path is None or not Path(path).exists():
        print(f"[model] No Word2Vec found at {{path}}; using random init.")
        return matrix

    suffix = Path(path).suffix.lower()
    matched = 0
    if suffix == ".bin":
        try:
            from gensim.models import KeyedVectors
            kv = KeyedVectors.load_word2vec_format(str(path), binary=True)
            for tok, idx in vocab.stoi.items():
                if tok in kv:
                    matrix[idx] = kv[tok]; matched += 1
        except ImportError:
            print("[model] gensim not installed; skipping .bin load.")
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < 2: continue
                tok = parts[0]
                if tok in vocab.stoi:
                    try:
                        vec = np.asarray(parts[1:1+dim], dtype="float32")
                        if len(vec) == dim:
                            matrix[vocab.stoi[tok]] = vec; matched += 1
                    except ValueError: continue
    print(f"[model] Word2Vec: matched {{matched}}/{{len(vocab)}} tokens.")
    return matrix


# =============================================================================
# MIDI FEATURES
# =============================================================================

def _safe_filename(artist, song):
    a = re.sub(r"\\s+", "_", artist.strip())
    s = re.sub(r"\\s+", "_", song.strip())
    return f"{{a}}_-_{{s}}.mid"


def _load_midi_safely(midi_path):
    if not midi_path.exists(): return None
    try:
        import pretty_midi
        return pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"[model] Failed to load {{midi_path.name}}: {{e}}")
        return None


def _midi_global_features(midi):
    try: tempo = float(midi.estimate_tempo()) if midi.instruments else 120.0
    except Exception: tempo = 120.0
    try: duration = float(midi.get_end_time())
    except Exception: duration = 0.0
    num_instruments = len(midi.instruments)
    num_notes = sum(len(i.notes) for i in midi.instruments)
    num_drum = sum(1 for i in midi.instruments if i.is_drum)
    avg_pitch = 0.0
    if num_notes > 0:
        pitches = [n.pitch for i in midi.instruments for n in i.notes]
        avg_pitch = float(sum(pitches) / len(pitches))
    feats = [tempo, duration, float(num_instruments), float(num_notes),
             float(num_drum), avg_pitch, 0.0, 0.0]
    return np.asarray(feats[:MIDI_SIMPLE_DIM], dtype="float32")


def _midi_per_word_features(midi, num_words):
    out = np.zeros((num_words, MIDI_PER_WORD_DIM), dtype="float32")
    if midi is None or num_words <= 0: return out
    try: total = float(midi.get_end_time())
    except Exception: total = 0.0
    if total <= 0: return out
    slot_dur = total / num_words
    all_notes = []
    for inst in midi.instruments:
        for n in inst.notes:
            all_notes.append((n.start, n.end, n.pitch, n.velocity,
                              inst.is_drum, id(inst)))
    if not all_notes: return out
    for w in range(num_words):
        t_start, t_end = w * slot_dur, (w + 1) * slot_dur
        active = [n for n in all_notes if n[0] < t_end and n[1] > t_start]
        if not active: continue
        pitches = [n[2] for n in active]
        velocities = [n[3] for n in active]
        out[w, 0] = float(len(active))
        out[w, 1] = float(np.mean(pitches))
        out[w, 2] = float(np.mean(velocities))
        out[w, 3] = float(len({{n[5] for n in active}}))
        out[w, 4] = 1.0 if any(n[4] for n in active) else 0.0
        out[w, 5] = float(len(active))
    return out


def get_midi_feature_dim(variant):
    return {{"none": 0, "simple": MIDI_SIMPLE_DIM,
            "per_word": MIDI_PER_WORD_DIM}}.get(variant, 0)


def tokenize_lyrics(text, line_separator_token="&"):
    text = text.lower().strip()
    text = re.sub(r"\\s+", " ", text)
    return [t for t in text.split(" ") if t]


# =============================================================================
# DATASET
# =============================================================================

class LyricsDataset(Dataset):
    def __init__(self, songs, vocab, midi_variant, seq_len):
        self.seq_len = int(seq_len)
        self.midi_dim = get_midi_feature_dim(midi_variant)
        self._stream_tokens = []
        self._stream_midi = []
        for song in songs:
            token_indices = vocab.encode(song["tokens"]) + [vocab.eos_idx]
            self._stream_tokens.extend(token_indices)
            if midi_variant == "none":
                self._stream_midi.extend([np.zeros(0, dtype="float32")] * len(token_indices))
            elif midi_variant == "simple":
                vec = song.get("midi_simple", np.zeros(MIDI_SIMPLE_DIM, dtype="float32"))
                self._stream_midi.extend([vec] * len(token_indices))
            elif midi_variant == "per_word":
                pw = song.get("midi_per_word")
                if pw is None:
                    pw = np.zeros((len(token_indices), MIDI_PER_WORD_DIM), dtype="float32")
                if len(pw) < len(token_indices):
                    pad = np.zeros((len(token_indices) - len(pw),
                                     MIDI_PER_WORD_DIM), dtype="float32")
                    pw = np.vstack([pw, pad])
                self._stream_midi.extend(pw[i] for i in range(len(token_indices)))

    def __len__(self):
        return max(0, len(self._stream_tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self._stream_tokens[idx:idx + self.seq_len]
        y = self._stream_tokens[idx + 1:idx + 1 + self.seq_len]
        if self.midi_dim > 0:
            midi = np.stack(self._stream_midi[idx:idx + self.seq_len])
            midi = torch.from_numpy(midi).float()
        else:
            midi = torch.zeros((self.seq_len, 0), dtype=torch.float32)
        return (torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long), midi)


# =============================================================================
# MODEL
# =============================================================================

class LSTMLanguageModel(nn.Module):
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
        self.embedding_dropout = (nn.Dropout(embedding_dropout)
                                    if embedding_dropout > 0 else nn.Identity())
        if midi_dim == 0:
            lstm_input_dim = embedding_dim
        elif fusion_method == "concatenate":
            lstm_input_dim = embedding_dim + midi_dim
            self.word_proj = nn.Identity()
            self.midi_proj = nn.Identity()
        elif fusion_method == "project":
            self.word_proj = nn.Sequential(
                nn.Linear(embedding_dim, fusion_proj_dim), nn.ReLU())
            self.midi_proj = nn.Sequential(
                nn.Linear(midi_dim, fusion_proj_dim), nn.ReLU())
            lstm_input_dim = 2 * fusion_proj_dim
        else:
            raise ValueError(f"Unknown fusion_method: {{fusion_method}}")
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              dropout=rnn_dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_size, vocab_size)
        if tie_weights and hidden_size == embedding_dim:
            self.head.weight = self.embedding.weight

    def forward(self, word_ids, midi_feats=None):
        emb = self.embedding_dropout(self.embedding(word_ids))
        if self.midi_dim == 0 or midi_feats is None or midi_feats.size(-1) == 0:
            lstm_in = emb
        elif self.fusion_method == "concatenate":
            lstm_in = torch.cat([emb, midi_feats], dim=-1)
        else:
            lstm_in = torch.cat([self.word_proj(emb),
                                   self.midi_proj(midi_feats)], dim=-1)
        out, hidden = self.lstm(lstm_in)
        return self.head(self.dropout(out)), hidden

    def step(self, word_id, midi_feat, hidden=None):
        word_ids = word_id.unsqueeze(1)
        midi_feats = midi_feat.unsqueeze(1) if midi_feat is not None and self.midi_dim > 0 else None
        emb = self.embedding_dropout(self.embedding(word_ids))
        if self.midi_dim == 0 or midi_feats is None:
            lstm_in = emb
        elif self.fusion_method == "concatenate":
            lstm_in = torch.cat([emb, midi_feats], dim=-1)
        else:
            lstm_in = torch.cat([self.word_proj(emb),
                                   self.midi_proj(midi_feats)], dim=-1)
        out, new_hidden = self.lstm(lstm_in, hidden)
        return self.head(self.dropout(out[:, -1, :])), new_hidden


# =============================================================================
# SAMPLING
# =============================================================================

def sample_next(strategy, logits, temperature=1.0, k=40, p=0.9):
    if strategy == "proportional":
        return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
    if strategy == "temperature":
        return torch.multinomial(F.softmax(logits / max(temperature, 1e-8), dim=-1), 1).squeeze(-1)
    if strategy == "top_k":
        k = min(max(k, 1), logits.size(-1))
        top_vals, _ = torch.topk(logits, k)
        cutoff = top_vals[-1]
        masked = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
        return torch.multinomial(F.softmax(masked / max(temperature, 1e-8), dim=-1), 1).squeeze(-1)
    if strategy == "nucleus":
        probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > p
        mask[1:] = mask[:-1].clone(); mask[0] = False
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()
        pos = torch.multinomial(sorted_probs, 1)
        return sorted_idx[pos].squeeze(-1)
    raise ValueError(f"Unknown sampling strategy: {{strategy}}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_lyrics_data():
    print(f"[model] Loading {{LYRICS_CSV_PATH}}...")
    songs = []
    all_tokens = []
    with open(LYRICS_CSV_PATH, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or len(row) < 3: continue
            artist, song, lyrics = row[0], row[1], row[2]
            if not lyrics.strip(): continue
            tokens = tokenize_lyrics(lyrics, LINE_SEPARATOR_TOKEN)
            if not tokens: continue
            songs.append({{"artist": artist.strip(), "song": song.strip(),
                           "tokens": tokens}})
            all_tokens.extend(tokens)
    print(f"[model] Loaded {{len(songs)}} songs, {{len(all_tokens)}} tokens.")
    return songs, all_tokens


def attach_midi_features(songs):
    if MIDI_VARIANT == "none" or not MIDI_DIR.exists():
        return
    loaded = 0
    for song in songs:
        fname = _safe_filename(song["artist"], song["song"])
        midi = _load_midi_safely(MIDI_DIR / fname)
        if midi is None: continue
        loaded += 1
        if MIDI_VARIANT == "simple":
            song["midi_simple"] = _midi_global_features(midi)
        elif MIDI_VARIANT == "per_word":
            song["midi_per_word"] = _midi_per_word_features(midi, len(song["tokens"]) + 1)
    print(f"[model] Loaded MIDI for {{loaded}}/{{len(songs)}} songs.")


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_loader, val_loader, vocab):
    pad_idx = vocab.pad_idx
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=LABEL_SMOOTHING)
    opt_cls = {{"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
                  "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}}
    optimizer = opt_cls.get(OPTIMIZER_NAME, torch.optim.Adam)(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        total_train = 0.0; n_train = 0
        for xb, yb, midi in train_loader:
            xb, yb, midi = xb.to(DEVICE), yb.to(DEVICE), midi.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(xb, midi)
            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B*T, V), yb.reshape(B*T))
            loss.backward()
            if GRADIENT_CLIPPING > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
            optimizer.step()
            total_train += loss.item(); n_train += 1

        model.eval()
        total_val = 0.0; n_val = 0
        with torch.no_grad():
            for xb, yb, midi in val_loader:
                xb, yb, midi = xb.to(DEVICE), yb.to(DEVICE), midi.to(DEVICE)
                logits, _ = model(xb, midi)
                B, T, V = logits.shape
                loss = loss_fn(logits.reshape(B*T, V), yb.reshape(B*T))
                total_val += loss.item(); n_val += 1

        avg_train = total_train / max(1, n_train)
        avg_val = total_val / max(1, n_val)
        ppl = float(np.exp(avg_val)) if avg_val < 20 else float("inf")
        print(f"[model] Epoch {{epoch+1}}/{{EPOCHS}}: "
              f"train_loss={{avg_train:.4f}}  val_loss={{avg_val:.4f}}  val_ppl={{ppl:.2f}}")
        if avg_val < best_val:
            best_val = avg_val
    return best_val


# =============================================================================
# GENERATION
# =============================================================================

def generate(model, vocab, initial_word, midi_features, max_words):
    model.eval()
    idx = vocab.stoi.get(initial_word.lower(), vocab.unk_idx)
    current = torch.tensor([idx], dtype=torch.long, device=DEVICE)
    hidden = None
    generated = []
    midi_dim = get_midi_feature_dim(MIDI_VARIANT)

    def midi_for_step(step):
        if midi_dim == 0 or midi_features is None: return None
        if midi_features.ndim == 1:
            return torch.from_numpy(midi_features).float().unsqueeze(0).to(DEVICE)
        s = min(step, midi_features.shape[0] - 1)
        return torch.from_numpy(midi_features[s]).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        for step in range(max_words):
            logits, hidden = model.step(current, midi_for_step(step), hidden)
            next_id = sample_next(SAMPLING_STRATEGY, logits.squeeze(0),
                                    SAMPLING_TEMPERATURE, SAMPLING_TOP_K, SAMPLING_TOP_P)
            ni = int(next_id.item())
            if ni == vocab.eos_idx: break
            generated.append(vocab.itos[ni])
            current = torch.tensor([ni], dtype=torch.long, device=DEVICE)
    return generated


def format_lyrics(tokens):
    lines = []; current = []
    for t in tokens:
        if t == LINE_SEPARATOR_TOKEN:
            if current: lines.append(" ".join(current)); current = []
        else:
            current.append(t)
    if current: lines.append(" ".join(current))
    return "\\n".join(lines)


def run_generation(model, vocab, test_songs):
    sections = []
    probe_results = []
    for song in test_songs:
        midi = song.get("midi_per_word") if MIDI_VARIANT == "per_word" else song.get("midi_simple")
        section = [f"="*70, f"Song: {{song['artist']}} - {{song['song']}}", f"="*70]
        for init in INITIAL_WORDS:
            section.append(f"\\n--- Initial: '{{init}}' ---")
            tokens = generate(model, vocab, init, midi, MAX_GENERATED_WORDS)
            section.append(f"{{init}} " + format_lyrics(tokens))
        sections.append("\\n".join(section))

        if MELODY_PROBE and midi is not None:
            torch.manual_seed(SEED)
            real = generate(model, vocab, INITIAL_WORDS[0], midi, MAX_GENERATED_WORDS // 2)
            shuffled = midi.copy()
            rng = np.random.default_rng(SEED)
            if shuffled.ndim == 2: rng.shuffle(shuffled, axis=0)
            else: rng.shuffle(shuffled)
            torch.manual_seed(SEED)
            corrupt = generate(model, vocab, INITIAL_WORDS[0], shuffled, MAX_GENERATED_WORDS // 2)
            set_r, set_c = set(real), set(corrupt)
            union = set_r | set_c
            jaccard = len(set_r & set_c) / max(1, len(union))
            L = min(len(real), len(corrupt))
            seq_ov = sum(1 for i in range(L) if real[i] == corrupt[i]) / max(1, L)
            probe_results.append({{
                "song": f"{{song['artist']}} - {{song['song']}}",
                "jaccard_similarity": jaccard,
                "sequence_overlap": seq_ov,
                "length_difference": abs(len(real) - len(corrupt)),
            }})

    OUTPUT_LYRICS_PATH.write_text("\\n\\n".join(sections), encoding="utf-8")
    print(f"[model] Wrote {{OUTPUT_LYRICS_PATH.name}}")
    if probe_results:
        PROBE_PATH.write_text(json.dumps(probe_results, indent=2), encoding="utf-8")
        print(f"[model] Wrote {{PROBE_PATH.name}}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    songs, all_tokens = load_lyrics_data()
    if not songs:
        sys.exit(f"No songs in {{LYRICS_CSV_PATH}}.")
    vocab = Vocabulary(all_tokens, MIN_WORD_COUNT, LINE_SEPARATOR_TOKEN)
    print(f"[model] Vocabulary size: {{len(vocab)}}")
    emb_matrix = load_word2vec(WORD2VEC_PATH, vocab, EMBEDDING_DIM)
    attach_midi_features(songs)

    # Train/val split (test set = last 10%)
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(songs))
    n = len(songs)
    n_train = int(n * 0.7); n_val = int(n * 0.2)
    train_songs = [songs[i] for i in indices[:n_train]]
    val_songs   = [songs[i] for i in indices[n_train:n_train+n_val]]
    test_songs  = [songs[i] for i in indices[n_train+n_val:]]

    midi_dim = get_midi_feature_dim(MIDI_VARIANT)
    model = LSTMLanguageModel(
        vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM, midi_dim=midi_dim,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
        fusion_method=FUSION_METHOD, fusion_proj_dim=FUSION_PROJ_DIM,
        embedding_matrix=emb_matrix, freeze_embeddings=FREEZE_EMBEDDINGS,
        embedding_dropout=EMBEDDING_DROPOUT, tie_weights=TIE_WEIGHTS,
    ).to(DEVICE)

    if CHECKPOINT_PATH.exists():
        print(f"[model] Loading existing checkpoint {{CHECKPOINT_PATH.name}}.")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if "state_dict" in ckpt: model.load_state_dict(ckpt["state_dict"])
        else: model.load_state_dict(ckpt)
    else:
        train_ds = LyricsDataset(train_songs, vocab, MIDI_VARIANT, SEQ_LEN)
        val_ds   = LyricsDataset(val_songs,   vocab, MIDI_VARIANT, SEQ_LEN)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        print(f"[model] Training for {{EPOCHS}} epochs (device={{DEVICE}})...")
        best_val = train_model(model, train_loader, val_loader, vocab)
        torch.save({{"state_dict": model.state_dict(), "best_val_loss": best_val}},
                   CHECKPOINT_PATH)
        print(f"[model] Saved {{CHECKPOINT_PATH.name}}")

    print(f"[model] Generating lyrics ({{len(test_songs)}} test songs x "
          f"{{len(INITIAL_WORDS)}} initial words, strategy={{SAMPLING_STRATEGY}})...")
    run_generation(model, vocab, test_songs)


if __name__ == "__main__":
    main()
'''
