"""
================================================================================
 AutoTune-NN  |  Automated Fine-Tuning Framework for Neural Networks
================================================================================

This file contains the TOP-LEVEL SELECTIONS for a run. They can be overridden
from the command line - every constant below has a matching `--<lower_name>` arg.

  python main.py                                # use the defaults below
  python main.py --architecture lstm            # override one setting
  python main.py --run_name ex1 --task_type language_modeling \
                 --data_type lyrics --local_dataset_path ./Data/lyrics.csv \
                 --midi_dir ./midi_files --word2vec_path ./word2vec.bin

Generation mode (after a training run):

  python main.py --mode generate --checkpoint <path> \
                 --initial_words love the morning \
                 --sampling_strategy nucleus --sampling_top_p 0.9

All other settings (parameter ranges, stopping conditions, quality weights,
parallelism, etc.) are in YAML files under configs/.

For a step-by-step setup walkthrough, see SETUP_GUIDE.md.
================================================================================
"""

# ==============================================================================
#                              RUN IDENTITY
# ==============================================================================

# RUN_NAME
# --------
# A short label for this run. Used as a prefix in the output folder name so
# multiple runs (different datasets / tasks) can coexist without overwriting.
# Examples: "experiment1", "iris_class", "lyrics_baseline"
RUN_NAME = "experiment1"


# ==============================================================================
#                           TOP-LEVEL SELECTIONS
# ==============================================================================

# ARCHITECTURE
#   "mlp" | "cnn" | "rnn" | "lstm" | "transformer"
ARCHITECTURE = "mlp"

# SEARCH_STRATEGY
#   "ftts" (default) | "bayesian" | "grid"
SEARCH_STRATEGY = "ftts"

# TASK_TYPE
#   "classification"   : predicts a class label (accuracy metric).
#   "regression"       : predicts a continuous value (RMSE metric).
#   "language_modeling": next-word prediction (loss/perplexity metric).
#                        Requires data_type='lyrics' and a sequence architecture.
TASK_TYPE = "classification"


# ==============================================================================
#                                DATASET
# ==============================================================================

# DATASET_MODE
#   "local" | "imported"
DATASET_MODE = "local"

# --- Local mode ---
LOCAL_DATASET_PATH = "./Data/Data-Set.csv"

# DATA_TYPE
#   "tabular" | "image" | "text" | "lyrics"
DATA_TYPE = "tabular"

# Feature/label columns (tabular). None = auto.
FEATURE_COLUMNS = None
LABEL_COLUMN = None

# Data split percentages. Sum = 1.0.
TRAIN_PCT = 0.6
VAL_PCT = 0.2
TEST_PCT = 0.2

# --- Imported mode ---
#   image: "mnist" | "fashion_mnist" | "cifar10" | "cifar100"
#   tabular: "iris" | "wine" | "breast_cancer" | "digits"
IMPORTED_DATASET_NAME = "mnist"


# ==============================================================================
#                  LANGUAGE MODELING (lyrics + MIDI) SETTINGS
#                used only when TASK_TYPE = "language_modeling"
# ==============================================================================

# WORD2VEC_PATH
# Path to pretrained word embeddings.
#   .bin format = Google News word2vec (requires gensim).
#   .txt format = GloVe-style space-separated text file.
# If not found, falls back to random init.
WORD2VEC_PATH = None

# MIDI_DIR
# Directory containing .mid files. Filenames must match "<artist>_-_<song>.mid"
# (with spaces -> underscores). Missing files fall back to zero vectors.
MIDI_DIR = None

# MIDI_VARIANT
#   "none"     : lyrics-only baseline (no MIDI features).
#   "simple"   : 8-dim global features (tempo, duration, num_instruments, ...).
#                Same vector at every timestep.
#   "per_word" : 8-dim per-timestep features (aligned by time).
MIDI_VARIANT = "none"

# Word2Vec embedding dimension (300 for Google News, 100 for GloVe-100, ...).
EMBEDDING_DIM = 300

# LINE_SEPARATOR_TOKEN
# Token marking the end of a lyrics line in the input CSV.
# Default: "&" (Assignment 3 convention). Override per dataset.
LINE_SEPARATOR_TOKEN = "&"

# Minimum word frequency to keep in the vocabulary. Less frequent words
# become <unk>.
MIN_WORD_COUNT = 2

# CSV column names for lyrics datasets.
LYRICS_TEXT_COLUMN = "lyrics"
LYRICS_ARTIST_COLUMN = "artist"
LYRICS_SONG_COLUMN = "song"


# ==============================================================================
#                     GENERATION MODE (post-training)
# ==============================================================================

# MODE
#   "tune"     : default - run hyperparameter tuning + final phase.
#   "generate" : skip tuning, load a checkpoint and generate lyrics.
MODE = "tune"

# CHECKPOINT_PATH (mode=generate only): path to model_checkpoint.pt
CHECKPOINT_PATH = None

# WARM_START_CHECKPOINT (mode=tune): if provided, load weights from this
# checkpoint as the starting point for the first trial (T0001), instead of
# random initialization. Useful for continuing tuning from a previously
# trained model. The checkpoint must match the architecture being tuned.
WARM_START_CHECKPOINT = None

# INITIAL_WORDS (mode=generate, or override at end of tune):
# List of starting words. None = use ["love", "the", "i"] as defaults.
INITIAL_WORDS = None

# SAMPLING_STRATEGY
#   "proportional" | "temperature" | "top_k" | "nucleus"
SAMPLING_STRATEGY = "proportional"
SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOP_K = 40
SAMPLING_TOP_P = 0.9

# Max generated words per sample.
MAX_GENERATED_WORDS = 200

# MELODY_PROBE: if True, run the melody-influence probe on each test song.
MELODY_PROBE = False

# RUN_DECODING_COMPARISON: if True, also generate lyrics with proportional /
# temperature / nucleus sampling side-by-side on 2 test songs. Required by
# Assignment 3 sec. 13 (diversity vs coherence analysis).
RUN_DECODING_COMPARISON = False


# ==============================================================================
#                                HARDWARE
# ==============================================================================
DEVICE = "auto"           # "auto" | "gpu" | "cpu"
RANDOM_SEED = 42


# ==============================================================================
#                                OUTPUT
# ==============================================================================
EXPERIMENTS_ROOT = "./experiments"


# ==============================================================================
#                          END OF USER SETTINGS
# ==============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.device_utils import resolve_device
from core.orchestrator import Orchestrator
from core.run_config import RunConfig


def _str_or_none(s):
    """argparse type: convert 'none'/'null' string to Python None."""
    if s is None:
        return None
    if isinstance(s, str) and s.lower() in ("none", "null", ""):
        return None
    return s


def _bool_arg(s):
    """argparse type: parse a boolean from string."""
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in ("true", "1", "yes", "y", "t")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser. Every flag maps to a constant at the top of this file."""
    p = argparse.ArgumentParser(
        description="AutoTune-NN: automated NN fine-tuning + lyrics generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Identity
    p.add_argument("--run_name", default=RUN_NAME)

    # Top-level selections
    p.add_argument("--architecture", default=ARCHITECTURE,
                    choices=["mlp", "cnn", "rnn", "lstm", "transformer"])
    p.add_argument("--search_strategy", default=SEARCH_STRATEGY,
                    choices=["ftts", "bayesian", "grid"])
    p.add_argument("--task_type", default=TASK_TYPE,
                    choices=["classification", "regression", "language_modeling"])

    # Dataset
    p.add_argument("--dataset_mode", default=DATASET_MODE,
                    choices=["local", "imported"])
    p.add_argument("--local_dataset_path", default=LOCAL_DATASET_PATH)
    p.add_argument("--data_type", default=DATA_TYPE,
                    choices=["tabular", "image", "text", "lyrics"])
    p.add_argument("--feature_columns", nargs="+", default=FEATURE_COLUMNS,
                    help="Space-separated column names; omit for auto.")
    p.add_argument("--label_column", type=_str_or_none, default=LABEL_COLUMN)
    p.add_argument("--train_pct", type=float, default=TRAIN_PCT)
    p.add_argument("--val_pct", type=float, default=VAL_PCT)
    p.add_argument("--test_pct", type=float, default=TEST_PCT)
    p.add_argument("--imported_dataset_name", default=IMPORTED_DATASET_NAME)

    # Language modeling
    p.add_argument("--word2vec_path", type=_str_or_none, default=WORD2VEC_PATH)
    p.add_argument("--midi_dir", type=_str_or_none, default=MIDI_DIR)
    p.add_argument("--midi_variant", default=MIDI_VARIANT,
                    choices=["none", "simple", "per_word"])
    p.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM)
    p.add_argument("--line_separator_token", default=LINE_SEPARATOR_TOKEN)
    p.add_argument("--min_word_count", type=int, default=MIN_WORD_COUNT)
    p.add_argument("--lyrics_text_column", default=LYRICS_TEXT_COLUMN)
    p.add_argument("--lyrics_artist_column", default=LYRICS_ARTIST_COLUMN)
    p.add_argument("--lyrics_song_column", default=LYRICS_SONG_COLUMN)

    # Generation
    p.add_argument("--mode", default=MODE, choices=["tune", "generate"])
    p.add_argument("--checkpoint", "--checkpoint_path", dest="checkpoint_path",
                    type=_str_or_none, default=CHECKPOINT_PATH,
                    help="For mode=generate: path to model_checkpoint.pt")
    p.add_argument("--warm_start_checkpoint", type=_str_or_none,
                    default=WARM_START_CHECKPOINT,
                    help="For mode=tune: load weights from this checkpoint "
                         "before T0001 (continue tuning from a trained model).")
    p.add_argument("--initial_words", nargs="+", default=INITIAL_WORDS,
                    help="Space-separated starting words for generation.")
    p.add_argument("--sampling_strategy", default=SAMPLING_STRATEGY,
                    choices=["proportional", "temperature", "top_k", "nucleus"])
    p.add_argument("--sampling_temperature", type=float, default=SAMPLING_TEMPERATURE)
    p.add_argument("--sampling_top_k", type=int, default=SAMPLING_TOP_K)
    p.add_argument("--sampling_top_p", type=float, default=SAMPLING_TOP_P)
    p.add_argument("--max_generated_words", type=int, default=MAX_GENERATED_WORDS)
    p.add_argument("--melody_probe", type=_bool_arg, default=MELODY_PROBE)
    p.add_argument("--run_decoding_comparison", type=_bool_arg,
                    default=RUN_DECODING_COMPARISON,
                    help="Generate side-by-side comparison of "
                         "proportional/temperature/nucleus sampling on 2 test songs.")

    # Hardware / output
    p.add_argument("--device", default=DEVICE, choices=["auto", "gpu", "cpu"])
    p.add_argument("--random_seed", type=_str_or_none, default=RANDOM_SEED,
                    help="Integer seed, or 'none' for non-deterministic.")
    p.add_argument("--experiments_root", default=EXPERIMENTS_ROOT)

    return p


def build_run_config(args) -> RunConfig:
    seed = args.random_seed
    if isinstance(seed, str):
        seed = int(seed) if seed and seed.lower() not in ("none", "null") else None

    return RunConfig(
        run_name=args.run_name,
        architecture=args.architecture,
        search_strategy=args.search_strategy,
        task_type=args.task_type,

        dataset_mode=args.dataset_mode,
        local_dataset_path=args.local_dataset_path,
        imported_dataset_name=args.imported_dataset_name,
        data_type=args.data_type,
        feature_columns=args.feature_columns,
        label_column=args.label_column,
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,

        device=resolve_device(args.device),
        random_seed=seed,
        experiments_root=args.experiments_root,

        # Language modeling
        word2vec_path=args.word2vec_path,
        midi_dir=args.midi_dir,
        midi_variant=args.midi_variant,
        line_separator_token=args.line_separator_token,
        min_word_count=args.min_word_count,
        embedding_dim=args.embedding_dim,
        lyrics_text_column=args.lyrics_text_column,
        lyrics_artist_column=args.lyrics_artist_column,
        lyrics_song_column=args.lyrics_song_column,

        # Generation mode
        mode=args.mode,
        checkpoint_path=args.checkpoint_path,
        warm_start_checkpoint=args.warm_start_checkpoint,
        initial_words=args.initial_words,
        sampling_strategy=args.sampling_strategy,
        sampling_temperature=args.sampling_temperature,
        sampling_top_k=args.sampling_top_k,
        sampling_top_p=args.sampling_top_p,
        max_generated_words=args.max_generated_words,
        melody_probe=args.melody_probe,
        run_decoding_comparison=args.run_decoding_comparison,
    )


def run_tune(cfg):
    """Standard tune mode: run hyperparameter search + final phase."""
    orchestrator = Orchestrator(cfg)
    summary = orchestrator.run()

    print("\n" + "=" * 80)
    print(" AutoTune-NN finished ".center(80, "="))
    print("=" * 80)
    print(f"Total trials          : {summary['total_trials']}")
    print(f"Best trial id         : {summary['best_trial_id']}")
    print(f"Best quality score    : {summary['best_quality']:.4f}")
    print(f"Best metric (raw)     : {summary['best_metric_raw']:.4f}")
    print(f"Best metric (smoothed): {summary['best_metric_smoothed']:.4f}")
    print(f"Stop reason           : {summary['stop_reason']}")
    print(f"Results directory     : {summary['results_dir']}")
    print("=" * 80)


def run_generate(cfg):
    """
    Generate-only mode: load checkpoint, run generation pipeline.
    Used when you've already trained a model and want to try different
    sampling strategies or initial words without retraining.
    """
    import torch
    from core.generator import run_generation_for_test_set
    from data_loaders.lyrics_loader import Vocabulary, load_lyrics

    if not cfg.checkpoint_path:
        raise ValueError("--checkpoint_path is required for mode=generate")

    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device)

    # Reconstruct vocab from checkpoint
    vocab = Vocabulary.__new__(Vocabulary)
    vocab.itos = ckpt["vocab_itos"]
    vocab.stoi = ckpt["vocab_stoi"]
    vocab.line_separator_token = ckpt["line_separator_token"]
    vocab.pad_idx = vocab.stoi.get("<pad>", 0)
    vocab.unk_idx = vocab.stoi.get("<unk>", 1)
    vocab.eos_idx = vocab.stoi.get("<eos>", 2)
    vocab.line_idx = vocab.stoi.get(vocab.line_separator_token, vocab.unk_idx)

    # Reload dataset for test_songs (with MIDI features)
    _, data_info = load_lyrics(cfg)
    # Override the loaded vocab with the checkpoint's vocab (so token IDs match)
    data_info["vocab"] = vocab
    data_info["vocab_size"] = len(vocab)

    # Build the model from saved hyperparameters
    from models import build_model
    model = build_model(ckpt["architecture"], ckpt["hyperparameters"], data_info)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device(cfg.device)
    model = model.to(device)

    output_dir = Path(cfg.experiments_root) / f"{cfg.run_name}_generate"
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_words = cfg.initial_words or ["love", "the", "i"]
    sampling_kwargs = {
        "temperature": cfg.sampling_temperature,
        "k": cfg.sampling_top_k,
        "p": cfg.sampling_top_p,
    }
    run_generation_for_test_set(
        model=model, vocab=vocab,
        test_songs=data_info["test_songs"],
        initial_words=initial_words,
        midi_variant=cfg.midi_variant, midi_dim=data_info["midi_dim"],
        device=device, output_dir=output_dir,
        sampling_strategy=cfg.sampling_strategy,
        sampling_kwargs=sampling_kwargs,
        max_words=cfg.max_generated_words,
        line_separator_token=cfg.line_separator_token,
        run_probe=cfg.melody_probe,
    )
    print(f"\n[generate] Output: {output_dir}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = build_run_config(args)
    cfg.validate()

    if cfg.mode == "generate":
        run_generate(cfg)
    else:
        run_tune(cfg)


if __name__ == "__main__":
    main()
