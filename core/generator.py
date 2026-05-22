"""
generator.py
------------
Lyrics generation pipeline.

Given a trained LSTM language model, an initial word, and an optional MIDI
representation, generate a sequence of lyrics token-by-token.

Supports:
  - Multiple sampling strategies (see core/sampling.py).
  - End-of-line awareness (configurable token).
  - Maximum length cap.
  - Melody-influence probe: generate with real MIDI vs shuffled MIDI and
    measure how much the output diverged.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.sampling import sample as sample_token


def generate_lyrics(
    model,
    vocab,
    initial_word: str,
    midi_features: Optional[np.ndarray],
    midi_dim: int,
    device,
    max_words: int = 200,
    sampling_strategy: str = "proportional",
    sampling_kwargs: dict = None,
    line_separator_token: str = "&",
    stop_on_eos: bool = True,
    max_words_per_line: int = 100,
    min_words_per_line: int = 0,
) -> list[str]:
    """
    Generate a lyrics sequence starting from initial_word.

    Args:
        model: trained LSTM language model with .step(word_id, midi_feat, hidden) method
        vocab: Vocabulary object
        initial_word: starting word
        midi_features: (T, midi_dim) array for per-word MIDI, or (midi_dim,) for
                       simple/global, or None for no MIDI.
        midi_dim: model's MIDI input dimension
        device: torch device
        max_words: maximum number of words to generate
        sampling_strategy: see core/sampling.py
        sampling_kwargs: additional kwargs for the sampling strategy
        line_separator_token: token marking end-of-line (kept in output)
        stop_on_eos: stop generation if <eos> is sampled
        max_words_per_line: if a line reaches this length without a separator,
                            forcibly insert one.
        min_words_per_line: if a separator is sampled before this many words,
                            suppress it and resample.

    Returns: list of generated tokens (does not include initial_word).
    """
    model.eval()
    sampling_kwargs = sampling_kwargs or {}

    # Resolve initial word -> token id
    initial_idx = vocab.stoi.get(initial_word.lower(), vocab.unk_idx)
    current_id = torch.tensor([initial_idx], dtype=torch.long, device=device)
    hidden = None

    generated = []

    # Handle MIDI features layout.
    # CRITICAL: if midi_dim > 0, the model REQUIRES a (B, midi_dim) tensor at
    # every step - otherwise the LSTM input shape mismatches its expected
    # input_size (e.g. expects 308 = 300 word_emb + 8 midi, gets 300 alone
    # and crashes with "Expected 308, got 300"). When MIDI features aren't
    # available (no .mid file, "simple" with no MIDI dir, etc.), feed zeros
    # so the model can still run - it'll behave as if midi were silent.
    def get_midi_for_step(step: int):
        if midi_dim == 0:
            return None
        if midi_features is None:
            # Model expects midi but song has none - feed zeros so shapes match
            return torch.zeros((1, midi_dim), dtype=torch.float32, device=device)
        feats = midi_features
        if feats.ndim == 1:
            # global (simple): same vector at every step
            return torch.from_numpy(feats).float().unsqueeze(0).to(device)
        # per_word: (T, midi_dim) - clamp index to last available step
        step = min(step, feats.shape[0] - 1)
        return torch.from_numpy(feats[step]).float().unsqueeze(0).to(device)

    line_sep_idx = vocab.stoi.get(line_separator_token, None)
    words_since_separator = 0

    with torch.no_grad():
        for step in range(max_words):
            midi_feat = get_midi_for_step(step)
            logits, hidden = model.step(current_id, midi_feat, hidden)
            # logits: (1, V)

            # Line-length constraints
            if line_sep_idx is not None:
                # If we've reached max_words_per_line without a separator,
                # force the next token to be the separator.
                if words_since_separator >= max_words_per_line:
                    next_idx = line_sep_idx
                else:
                    # If we haven't reached min_words_per_line, suppress
                    # the separator by setting its logit to -inf.
                    if words_since_separator < min_words_per_line:
                        logits_modified = logits.clone()
                        logits_modified[0, line_sep_idx] = float("-inf")
                    else:
                        logits_modified = logits
                    next_id = sample_token(sampling_strategy,
                                            logits_modified.squeeze(0),
                                            **sampling_kwargs)
                    next_idx = int(next_id.item())
            else:
                next_id = sample_token(sampling_strategy, logits.squeeze(0),
                                         **sampling_kwargs)
                next_idx = int(next_id.item())

            if stop_on_eos and next_idx == vocab.eos_idx:
                break

            generated.append(vocab.itos[next_idx])

            # Track distance from last separator
            if next_idx == line_sep_idx:
                words_since_separator = 0
            else:
                words_since_separator += 1

            current_id = torch.tensor([next_idx], dtype=torch.long, device=device)

    return generated


def format_lyrics(tokens: list[str], line_separator_token: str = "&") -> str:
    """
    Convert a list of generated tokens into a multi-line lyrics string.
    The line_separator_token is rendered as a newline.
    """
    lines = []
    current = []
    for tok in tokens:
        if tok == line_separator_token:
            if current:
                lines.append(" ".join(current))
                current = []
        else:
            current.append(tok)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def melody_influence_probe(
    model,
    vocab,
    initial_word: str,
    midi_features: np.ndarray,
    midi_dim: int,
    device,
    max_words: int = 100,
    sampling_strategy: str = "proportional",
    sampling_kwargs: dict = None,
    line_separator_token: str = "&",
    seed: int = 42,
) -> dict:
    """
    Run the same model with real vs corrupted MIDI; measure output divergence.

    Quantitative measures used:
      1. Jaccard similarity of unique tokens (lower = more different).
      2. Sequence-level overlap (token-by-token match rate).
      3. Length difference (absolute words).

    Returns dict with both lyrics + the metrics.
    """
    sampling_kwargs = sampling_kwargs or {}

    # Seed for reproducibility of the comparison
    torch.manual_seed(seed)
    real_tokens = generate_lyrics(
        model, vocab, initial_word, midi_features, midi_dim, device,
        max_words=max_words, sampling_strategy=sampling_strategy,
        sampling_kwargs=sampling_kwargs,
        line_separator_token=line_separator_token,
    )

    # Corrupt the MIDI: shuffle along the time axis if 2D, else permute features
    corrupted = midi_features.copy() if midi_features is not None else None
    if corrupted is not None:
        rng = np.random.default_rng(seed)
        if corrupted.ndim == 2:
            rng.shuffle(corrupted, axis=0)
        else:
            rng.shuffle(corrupted)

    torch.manual_seed(seed)  # same RNG state -> any divergence is from MIDI alone
    corrupted_tokens = generate_lyrics(
        model, vocab, initial_word, corrupted, midi_dim, device,
        max_words=max_words, sampling_strategy=sampling_strategy,
        sampling_kwargs=sampling_kwargs,
        line_separator_token=line_separator_token,
    )

    # Metrics
    set_real, set_corr = set(real_tokens), set(corrupted_tokens)
    union = set_real | set_corr
    jaccard = len(set_real & set_corr) / max(1, len(union))

    seq_overlap = 0.0
    if real_tokens and corrupted_tokens:
        L = min(len(real_tokens), len(corrupted_tokens))
        if L > 0:
            matches = sum(1 for i in range(L) if real_tokens[i] == corrupted_tokens[i])
            seq_overlap = matches / L

    length_diff = abs(len(real_tokens) - len(corrupted_tokens))

    return {
        "real_lyrics": format_lyrics(real_tokens, line_separator_token),
        "corrupted_lyrics": format_lyrics(corrupted_tokens, line_separator_token),
        "real_tokens": real_tokens,
        "corrupted_tokens": corrupted_tokens,
        "jaccard_similarity": jaccard,         # 1=identical token sets, 0=disjoint
        "sequence_overlap": seq_overlap,        # 1=identical sequence, 0=disjoint
        "length_difference": length_diff,
        "interpretation": (
            f"Real and corrupted-MIDI outputs share {jaccard:.1%} of unique tokens "
            f"and match position-by-position {seq_overlap:.1%} of the time. "
            f"Length differs by {length_diff} words. "
            f"Lower values = melody influences generation more."
        ),
    }


def run_generation_for_test_set(
    model,
    vocab,
    test_songs: list[dict],
    initial_words: list[str],
    midi_variant: str,
    midi_dim: int,
    device,
    output_dir: Path,
    sampling_strategy: str = "proportional",
    sampling_kwargs: dict = None,
    max_words: int = 200,
    line_separator_token: str = "&",
    run_probe: bool = False,
):
    """
    Generate lyrics for every test song x every initial word.
    Optionally also run a melody-influence probe per test song.

    Writes to output_dir/generated_lyrics.txt.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "generated_lyrics.txt"
    probe_path = output_dir / "melody_probe.json"

    sampling_kwargs = sampling_kwargs or {}

    sections = []
    probe_results = []

    for song in test_songs:
        section = []
        section.append(f"=" * 70)
        section.append(f"Song: {song['artist']} - {song['song']}")
        section.append(f"=" * 70)

        midi_feats = song.get("midi_per_word") if midi_variant == "per_word" else song.get("midi_simple")

        for initial in initial_words:
            section.append(f"\n--- Initial word: '{initial}' ---")
            tokens = generate_lyrics(
                model, vocab, initial, midi_feats, midi_dim, device,
                max_words=max_words, sampling_strategy=sampling_strategy,
                sampling_kwargs=sampling_kwargs,
                line_separator_token=line_separator_token,
            )
            section.append(f"{initial} " + format_lyrics(tokens, line_separator_token))

        sections.append("\n".join(section))

        # Probe
        if run_probe and midi_feats is not None:
            probe = melody_influence_probe(
                model, vocab, initial_words[0], midi_feats, midi_dim, device,
                max_words=max_words // 2, sampling_strategy=sampling_strategy,
                sampling_kwargs=sampling_kwargs,
                line_separator_token=line_separator_token,
            )
            probe_results.append({
                "song": f"{song['artist']} - {song['song']}",
                "initial_word": initial_words[0],
                "jaccard_similarity": probe["jaccard_similarity"],
                "sequence_overlap": probe["sequence_overlap"],
                "length_difference": probe["length_difference"],
                "interpretation": probe["interpretation"],
            })

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections))
    print(f"[generator] Wrote generated lyrics -> {out_path}")

    if probe_results:
        with open(probe_path, "w", encoding="utf-8") as f:
            json.dump(probe_results, f, indent=2)
        print(f"[generator] Wrote melody probe -> {probe_path}")


def run_decoding_comparison(
    model,
    vocab,
    test_songs: list[dict],
    initial_word: str,
    midi_variant: str,
    midi_dim: int,
    device,
    output_dir: Path,
    strategies: list[dict] = None,
    max_words: int = 200,
    line_separator_token: str = "&",
    num_songs_to_compare: int = 2,
):
    """
    Compare decoding strategies on a small subset of test songs.

    Required by Assignment 3 sec. 13 ("Compare the decoding strategies you
    implemented. For a small fixed subset of the test cases (e.g., two
    melodies), generate lyrics using proportional, temperature-scaled, and
    top-k or nucleus sampling").

    Writes decoding_comparison.txt with the same (song, initial_word) generated
    via each strategy, so the user can directly compare diversity vs. coherence.

    Args:
        strategies: list of dicts like
            [{"name": "proportional"},
             {"name": "temperature", "temperature": 0.7},
             {"name": "nucleus", "p": 0.9}]
          Defaults to a reasonable comparison set.
        num_songs_to_compare: how many test songs to use (first N).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "decoding_comparison.txt"

    if strategies is None:
        strategies = [
            {"name": "proportional"},
            {"name": "temperature", "temperature": 0.7},
            {"name": "nucleus", "p": 0.9},
        ]

    songs_subset = test_songs[:num_songs_to_compare]

    sections = []
    sections.append("=" * 70)
    sections.append("DECODING STRATEGY COMPARISON")
    sections.append("=" * 70)
    sections.append(
        f"Comparing {len(strategies)} sampling strategies on "
        f"{len(songs_subset)} test songs (initial word: '{initial_word}').\n"
        f"Use this to assess diversity vs. coherence trade-offs."
    )
    sections.append("")

    for song in songs_subset:
        sections.append("=" * 70)
        sections.append(f"Song: {song['artist']} - {song['song']}")
        sections.append("=" * 70)

        midi_feats = (song.get("midi_per_word") if midi_variant == "per_word"
                       else song.get("midi_simple"))

        for strat in strategies:
            name = strat["name"]
            kwargs = {k: v for k, v in strat.items() if k != "name"}
            label_parts = [name]
            for k, v in kwargs.items():
                label_parts.append(f"{k}={v}")
            label = " | ".join(label_parts)
            sections.append(f"\n--- {label} ---")

            tokens = generate_lyrics(
                model, vocab, initial_word, midi_feats, midi_dim, device,
                max_words=max_words, sampling_strategy=name,
                sampling_kwargs=kwargs,
                line_separator_token=line_separator_token,
            )
            sections.append(f"{initial_word} " + format_lyrics(tokens, line_separator_token))

        sections.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))
    print(f"[generator] Wrote decoding comparison -> {out_path}")
