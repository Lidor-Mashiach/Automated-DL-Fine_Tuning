"""
lyrics_loader.py
----------------
Data loader for language-modeling tasks on song lyrics + MIDI.

Inputs:
  - CSV with columns (artist, song, lyrics) - exact column names configurable.
  - Lyrics are sequences of words separated by spaces. Lines are separated by
    a configurable token (default '&').
  - MIDI files in a directory. Filename is derived from artist+song.
  - Pretrained Word2Vec embeddings (300-dim, configurable).

Output for each step in a sequence:
  - Input vector = Word2Vec(word_t) ++ MIDI_features(t)
  - Target = word_{t+1} (token index in vocabulary)

MIDI variants:
  - "none"     - no MIDI features (lyrics-only baseline). Zero-vector at the
                 model's MIDI port.
  - "simple"   - global features (tempo, key, time_signature, num_instruments,
                 total_duration). Same vector at every timestep.
  - "per_word" - per-timestep features aligned by time. The midi total duration
                 is divided evenly across the song's words. For each word, we
                 extract the active notes at the word's time slot.

If a song has no matching MIDI file, midi features fall back to zeros.

This loader handles vocab building, OOV (<unk>), padding (<pad>), end-of-line
(configurable token), and end-of-song (<eos>).
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]


# -----------------------------------------------------------------------------
# Vocabulary
# -----------------------------------------------------------------------------

class Vocabulary:
    """Token-to-index mapping, with special tokens and OOV handling."""

    def __init__(self, tokens: list[str], min_count: int = 2,
                 line_separator_token: str = "&"):
        self.line_separator_token = line_separator_token
        counts = Counter(tokens)
        # Keep frequent tokens, plus the line separator if it appears
        keep = [t for t, c in counts.most_common() if c >= min_count]
        # Ensure line separator and EOS are kept regardless of frequency
        if line_separator_token in counts and line_separator_token not in keep:
            keep.append(line_separator_token)
        # Build vocab: specials first
        self.itos = list(SPECIAL_TOKENS) + [t for t in keep if t not in SPECIAL_TOKENS]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_idx = self.stoi[PAD_TOKEN]
        self.unk_idx = self.stoi[UNK_TOKEN]
        self.eos_idx = self.stoi[EOS_TOKEN]
        self.line_idx = self.stoi.get(line_separator_token, self.unk_idx)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, self.unk_idx) for t in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.itos[i] for i in indices]


# -----------------------------------------------------------------------------
# Word2Vec loader
# -----------------------------------------------------------------------------

def load_word2vec(path: str, vocab: Vocabulary, embedding_dim: int = 300) -> np.ndarray:
    """
    Load pretrained Word2Vec embeddings, aligned to a vocabulary.

    Supports two formats:
      - .bin (Google News binary format) - requires gensim
      - .txt (GloVe/word2vec text format) - parsed directly

    Returns: (vocab_size, embedding_dim) numpy float32 matrix.
    Out-of-vocab words get a small random vector.
    """
    matrix = np.random.uniform(-0.05, 0.05, (len(vocab), embedding_dim)).astype("float32")
    # Pad token = zeros
    matrix[vocab.pad_idx] = 0.0

    if path is None or not Path(path).exists():
        print(f"[lyrics_loader] No Word2Vec file at {path}; using random init.")
        return matrix

    suffix = Path(path).suffix.lower()
    matched = 0

    if suffix == ".bin":
        try:
            from gensim.models import KeyedVectors
            kv = KeyedVectors.load_word2vec_format(path, binary=True)
            for token, idx in vocab.stoi.items():
                if token in kv:
                    matrix[idx] = kv[token]
                    matched += 1
        except ImportError:
            print("[lyrics_loader] gensim not installed; skipping .bin load.")
    else:
        # GloVe / word2vec text format: each line = "token v1 v2 ... vN"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < 2:
                    continue
                token = parts[0]
                if token in vocab.stoi:
                    try:
                        vec = np.asarray(parts[1:1 + embedding_dim], dtype="float32")
                        if len(vec) == embedding_dim:
                            matrix[vocab.stoi[token]] = vec
                            matched += 1
                    except ValueError:
                        continue
    print(f"[lyrics_loader] Word2Vec: matched {matched}/{len(vocab)} tokens.")
    return matrix


# -----------------------------------------------------------------------------
# MIDI features
# -----------------------------------------------------------------------------

def _safe_filename(artist: str, song: str) -> str:
    """Reproduce the assignment's MIDI filename convention."""
    a = re.sub(r"\s+", "_", artist.strip())
    s = re.sub(r"\s+", "_", song.strip())
    return f"{a}_-_{s}.mid"


def _midi_global_features(midi) -> np.ndarray:
    """Extract a small global feature vector from a PrettyMIDI object."""
    try:
        tempo = float(midi.estimate_tempo()) if midi.instruments else 120.0
    except Exception:
        tempo = 120.0
    try:
        duration = float(midi.get_end_time())
    except Exception:
        duration = 0.0
    num_instruments = len(midi.instruments)
    num_notes = sum(len(inst.notes) for inst in midi.instruments)
    # Drum vs pitched ratio
    num_drum = sum(1 for inst in midi.instruments if inst.is_drum)
    avg_pitch = 0.0
    if num_notes > 0:
        pitches = [n.pitch for inst in midi.instruments for n in inst.notes]
        avg_pitch = float(sum(pitches) / len(pitches))
    feats = [tempo, duration, float(num_instruments), float(num_notes),
             float(num_drum), avg_pitch]
    # Pad to MIDI_SIMPLE_DIM
    feats = feats + [0.0] * (MIDI_SIMPLE_DIM - len(feats))
    return np.asarray(feats[:MIDI_SIMPLE_DIM], dtype="float32")


def _midi_per_word_features(midi, num_words: int) -> np.ndarray:
    """
    Time-align MIDI to words. Returns (num_words, MIDI_PER_WORD_DIM).

    Strategy: divide total duration evenly by num_words. For each word slot,
    extract notes that are active in that slot's time window.

    Per-slot features:
      - num_active_notes
      - mean pitch
      - mean velocity
      - num distinct instruments
      - drum presence (0/1)
      - max simultaneous notes in the slot
    """
    out = np.zeros((num_words, MIDI_PER_WORD_DIM), dtype="float32")
    if midi is None or num_words <= 0:
        return out

    try:
        total = float(midi.get_end_time())
    except Exception:
        total = 0.0
    if total <= 0:
        return out

    slot_dur = total / num_words

    # Flatten all notes once
    all_notes = []
    for inst in midi.instruments:
        for n in inst.notes:
            all_notes.append((n.start, n.end, n.pitch, n.velocity, inst.is_drum, id(inst)))

    if not all_notes:
        return out

    for w in range(num_words):
        t_start = w * slot_dur
        t_end = t_start + slot_dur
        active = [n for n in all_notes if n[0] < t_end and n[1] > t_start]
        if not active:
            continue
        pitches = [n[2] for n in active]
        velocities = [n[3] for n in active]
        drum_present = 1.0 if any(n[4] for n in active) else 0.0
        distinct_instruments = len({n[5] for n in active})
        out[w, 0] = float(len(active))
        out[w, 1] = float(np.mean(pitches))
        out[w, 2] = float(np.mean(velocities))
        out[w, 3] = float(distinct_instruments)
        out[w, 4] = drum_present
        out[w, 5] = float(len(active))  # max simultaneous (approx via count)
    return out


MIDI_SIMPLE_DIM = 8
MIDI_PER_WORD_DIM = 8


def get_midi_feature_dim(variant: str) -> int:
    """Return the per-step MIDI feature dimension for a variant."""
    if variant == "none":
        return 0
    if variant == "simple":
        return MIDI_SIMPLE_DIM
    if variant == "per_word":
        return MIDI_PER_WORD_DIM
    raise ValueError(f"Unknown midi_variant: {variant}")


def _load_midi_safely(midi_path: Path):
    """Load a MIDI file, returning None on failure."""
    if not midi_path.exists():
        return None
    try:
        import pretty_midi
        return pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"[lyrics_loader] Failed to load {midi_path.name}: {e}")
        return None


# -----------------------------------------------------------------------------
# Lyrics tokenization
# -----------------------------------------------------------------------------

def tokenize_lyrics(text: str, line_separator_token: str = "&") -> list[str]:
    """
    Lowercase, strip, split on whitespace. Keep line_separator_token as a token.
    """
    text = text.lower().strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    tokens = text.split(" ")
    # Keep only non-empty
    tokens = [t for t in tokens if t]
    return tokens


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class LyricsDataset(Dataset):
    """
    Each item is a (input_indices, target_indices, midi_features) triple.

    input_indices: LongTensor (seq_len,) - word indices for input timesteps
    target_indices: LongTensor (seq_len,) - same indices shifted by 1 (next-word)
    midi_features: FloatTensor (seq_len, midi_dim) - MIDI features per step
                   For "simple" variant, the same global vector is broadcast.
                   For "none", an empty (seq_len, 0) tensor.
    """

    def __init__(self, songs: list[dict], vocab: Vocabulary,
                 midi_variant: str, seq_len: int):
        self.songs = songs
        self.vocab = vocab
        self.midi_variant = midi_variant
        self.seq_len = int(seq_len)
        self.midi_dim = get_midi_feature_dim(midi_variant)

        # Pre-tokenize all songs into one big stream of (token_idx, midi_feat_for_word)
        # so we can sample fixed-length windows.
        self._stream_tokens = []
        self._stream_midi = []

        for song in songs:
            tokens = song["tokens"]
            token_indices = vocab.encode(tokens) + [vocab.eos_idx]
            self._stream_tokens.extend(token_indices)

            if midi_variant == "none":
                # zero-dim, fill nothing
                self._stream_midi.extend([np.zeros(0, dtype="float32")] * len(token_indices))
            elif midi_variant == "simple":
                vec = song.get("midi_simple")
                if vec is None:
                    vec = np.zeros(MIDI_SIMPLE_DIM, dtype="float32")
                self._stream_midi.extend([vec] * len(token_indices))
            elif midi_variant == "per_word":
                per_word = song.get("midi_per_word")
                if per_word is None:
                    per_word = np.zeros((len(token_indices), MIDI_PER_WORD_DIM),
                                        dtype="float32")
                # If lengths mismatch, pad or truncate
                if len(per_word) < len(token_indices):
                    pad = np.zeros((len(token_indices) - len(per_word),
                                    MIDI_PER_WORD_DIM), dtype="float32")
                    per_word = np.vstack([per_word, pad])
                self._stream_midi.extend(per_word[i] for i in range(len(token_indices)))

    def __len__(self) -> int:
        # Number of valid windows (need seq_len + 1 tokens for input + target)
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
                torch.tensor(y, dtype=torch.long),
                midi)


# -----------------------------------------------------------------------------
# Entry point used by data_loaders/__init__.py
# -----------------------------------------------------------------------------

def load_lyrics(cfg):
    """
    Load lyrics + MIDI dataset for language modeling.

    Returns (make_loaders, data_info):
      - make_loaders(batch_size, val_split, ...) -> (train_loader, val_loader)
      - data_info: dict with vocab, embedding_matrix, midi_dim, output_dim, etc.
    """
    csv_path = Path(cfg.local_dataset_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Lyrics CSV not found: {csv_path}")

    midi_dir = Path(cfg.midi_dir) if cfg.midi_dir else None
    sep_token = cfg.line_separator_token

    # Detect column layout: the assignment CSV has no header (artist, song, lyrics)
    # so we read all rows and split.
    songs: list[dict] = []
    all_tokens: list[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            raise ValueError("Empty CSV.")
        # Heuristic: if the first row's third field looks like song lyrics
        # (long, contains many words), treat it as data. Otherwise treat as header.
        is_header = (len(first) >= 3
                     and len(first[2].split()) < 10
                     and first[2].strip().lower() in ("lyrics", "text"))
        rows = reader if is_header else [first] + list(reader)
        # `reader` is a generator - need to capture rows
        if not is_header:
            rest = list(reader)
            rows = [first] + rest

    # Re-read more robustly
    songs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue
            artist, song, lyrics = row[0], row[1], row[2]
            if not lyrics.strip():
                continue
            tokens = tokenize_lyrics(lyrics, line_separator_token=sep_token)
            if not tokens:
                continue
            songs.append({
                "artist": artist.strip(),
                "song": song.strip(),
                "tokens": tokens,
            })
            all_tokens.extend(tokens)

    print(f"[lyrics_loader] Loaded {len(songs)} songs, {len(all_tokens)} total tokens.")

    if not songs:
        raise ValueError(f"No songs loaded from {csv_path}.")

    # Build vocab
    vocab = Vocabulary(all_tokens, min_count=cfg.min_word_count,
                        line_separator_token=sep_token)
    print(f"[lyrics_loader] Vocabulary size: {len(vocab)}")

    # Load embedding matrix
    embedding_matrix = load_word2vec(cfg.word2vec_path, vocab, cfg.embedding_dim)

    # Load MIDI features per song
    if cfg.midi_variant != "none" and midi_dir is not None:
        loaded_midi = 0
        for song in songs:
            fname = _safe_filename(song["artist"], song["song"])
            midi = _load_midi_safely(midi_dir / fname)
            if midi is None:
                continue
            loaded_midi += 1
            if cfg.midi_variant == "simple":
                song["midi_simple"] = _midi_global_features(midi)
            elif cfg.midi_variant == "per_word":
                # Account for the <eos> token we append later
                num_words = len(song["tokens"]) + 1
                song["midi_per_word"] = _midi_per_word_features(midi, num_words)
        print(f"[lyrics_loader] Loaded MIDI for {loaded_midi}/{len(songs)} songs.")

    # Train/Val/Test split at the SONG level (not token level)
    np.random.seed(cfg.random_seed or 42)
    n = len(songs)
    indices = np.random.permutation(n)
    n_train = int(n * cfg.train_pct)
    n_val = int(n * cfg.val_pct)
    train_songs = [songs[i] for i in indices[:n_train]]
    val_songs = [songs[i] for i in indices[n_train:n_train + n_val]]
    test_songs = [songs[i] for i in indices[n_train + n_val:]]

    print(f"[lyrics_loader] Split: train={len(train_songs)}, "
          f"val={len(val_songs)}, test={len(test_songs)}")

    midi_dim = get_midi_feature_dim(cfg.midi_variant)

    def make_loaders(batch_size: int = 64, val_split: float = 0.2,
                     seed: int = 42, num_workers: int = 0,
                     seq_len: int = 32, **_):
        train_ds = LyricsDataset(train_songs, vocab, cfg.midi_variant, seq_len)
        val_ds = LyricsDataset(val_songs, vocab, cfg.midi_variant, seq_len)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, val_loader

    data_info = {
        "data_type": "lyrics",
        "task_type": "language_modeling",
        "vocab": vocab,
        "vocab_size": len(vocab),
        "embedding_matrix": embedding_matrix,
        "embedding_dim": cfg.embedding_dim,
        "midi_dim": midi_dim,
        "midi_variant": cfg.midi_variant,
        "output_dim": len(vocab),  # next-word prediction -> vocab_size
        "input_dim": cfg.embedding_dim + midi_dim,
        "test_songs": test_songs,  # used by generation pipeline
        "imbalance_ratio": 1.0,
    }
    return make_loaders, data_info
