"""
RunConfig
---------
Small dataclass that bundles the user's main.py choices.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class RunConfig:
    """User selections from main.py."""

    # Identity
    run_name: str

    # High-level choices
    architecture: str
    search_strategy: str
    task_type: str

    # Data source
    dataset_mode: str
    local_dataset_path: str
    imported_dataset_name: str
    data_type: str
    feature_columns: Optional[list]
    label_column: Optional[str]
    train_pct: float
    val_pct: float
    test_pct: float

    # Hardware
    device: str
    random_seed: Optional[int]

    # Output
    experiments_root: str

    # Language modeling fields (optional, used only when task_type == "language_modeling")
    word2vec_path: Optional[str] = None        # path to .bin or .txt embeddings
    midi_dir: Optional[str] = None             # directory with .mid files
    midi_variant: str = "none"                 # "none" | "simple" | "per_word"
    line_separator_token: str = "&"            # marks end-of-line in lyrics
    min_word_count: int = 2                    # min token freq to keep in vocab
    embedding_dim: int = 300                   # Word2Vec dim; 300 for Google News
    lyrics_text_column: str = "lyrics"         # column name for text
    lyrics_artist_column: str = "artist"       # used to map to MIDI filename
    lyrics_song_column: str = "song"

    # Generation mode fields
    mode: str = "tune"                         # "tune" | "generate"
    checkpoint_path: Optional[str] = None      # for generate mode
    initial_words: Optional[list] = None       # for generate mode (3 first words)
    sampling_strategy: str = "proportional"    # see core/sampling.py
    sampling_temperature: float = 1.0          # for temperature strategy
    sampling_top_k: int = 40                   # for top_k strategy
    sampling_top_p: float = 0.9                # for nucleus strategy
    max_generated_words: int = 200             # generation length cap
    melody_probe: bool = False                 # run melody-influence probe

    _VALID_ARCH = ("mlp", "cnn", "rnn", "lstm", "transformer")
    _VALID_STRATEGY = ("ftts", "bayesian", "grid")
    _VALID_DATA_TYPES = ("tabular", "image", "text", "lyrics")
    _VALID_TASK_TYPES = ("classification", "regression", "language_modeling")
    _VALID_MODES = ("local", "imported")

    def validate(self) -> None:
        if not self.run_name or not self.run_name.strip():
            raise ValueError("RUN_NAME must be a non-empty string.")
        if self.architecture not in self._VALID_ARCH:
            raise ValueError(
                f"architecture='{self.architecture}' invalid. "
                f"Expected one of {self._VALID_ARCH}."
            )
        if self.search_strategy not in self._VALID_STRATEGY:
            raise ValueError(
                f"search_strategy='{self.search_strategy}' invalid. "
                f"Expected one of {self._VALID_STRATEGY}."
            )
        if self.task_type not in self._VALID_TASK_TYPES:
            raise ValueError(f"task_type='{self.task_type}' invalid.")
        if self.dataset_mode not in self._VALID_MODES:
            raise ValueError(f"dataset_mode='{self.dataset_mode}' invalid.")
        if self.data_type not in self._VALID_DATA_TYPES:
            raise ValueError(f"data_type='{self.data_type}' invalid.")

        total = self.train_pct + self.val_pct + self.test_pct
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_pct + val_pct + test_pct must equal 1.0, got {total}."
            )

        # language_modeling requires specific setup
        if self.task_type == "language_modeling":
            if self.data_type != "lyrics":
                raise ValueError(
                    "task_type='language_modeling' requires data_type='lyrics'."
                )
            if self.architecture not in ("rnn", "lstm", "transformer"):
                raise ValueError(
                    f"task_type='language_modeling' requires a sequence "
                    f"architecture (rnn/lstm/transformer), got {self.architecture}."
                )
            if self.midi_variant not in ("none", "simple", "per_word"):
                raise ValueError(
                    f"midi_variant must be 'none', 'simple', or 'per_word', "
                    f"got {self.midi_variant}."
                )
            if self.mode == "generate":
                if not self.checkpoint_path:
                    raise ValueError("mode='generate' requires checkpoint_path.")
                if not self.initial_words:
                    raise ValueError("mode='generate' requires initial_words.")

    def dataset_label(self) -> str:
        """A short label identifying the dataset (used in folder name)."""
        if self.dataset_mode == "imported":
            return self.imported_dataset_name.lower()
        from pathlib import Path
        return Path(self.local_dataset_path).stem.lower()

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}
