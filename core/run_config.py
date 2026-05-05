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

    _VALID_ARCH = ("mlp", "cnn", "rnn", "lstm", "transformer")
    _VALID_STRATEGY = ("ftts", "bayesian", "grid")
    _VALID_DATA_TYPES = ("tabular", "image", "text")
    _VALID_TASK_TYPES = ("classification", "regression")
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

    def dataset_label(self) -> str:
        """A short label identifying the dataset (used in folder name)."""
        if self.dataset_mode == "imported":
            return self.imported_dataset_name.lower()
        from pathlib import Path
        return Path(self.local_dataset_path).stem.lower()

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}
