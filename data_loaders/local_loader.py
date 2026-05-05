"""
Local data loader
-----------------
Loads a user-provided dataset from disk, with a 3-way split saved as CSV files
for full transparency and reproducibility.

On first run:
  Data/Data-Set.csv -> Data/Train_set.csv + Data/Val_set.csv + Data/Test_set.csv
  (split proportions from RunConfig: train_pct / val_pct / test_pct)

On subsequent runs: reuses the existing split files. To re-split, delete them.

During training:
  - Train_set is used for parameter updates.
  - Val_set is used for early stopping and quality scoring.
  - Test_set is NOT touched during tuning (holdout for final evaluation).

Supports tabular (CSV), text (CSV with text+label), and image (ImageFolder).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from core.run_config import RunConfig


def load_local(cfg: RunConfig):
    if cfg.data_type == "tabular":
        return _load_tabular(cfg)
    if cfg.data_type == "text":
        return _load_text(cfg)
    if cfg.data_type == "image":
        return _load_image(cfg)
    raise ValueError(f"data_type='{cfg.data_type}' not supported in local mode.")


# ============================================================= TABULAR

def _load_tabular(cfg: RunConfig):
    source = Path(cfg.local_dataset_path)
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset file '{source}' not found. "
            f"Place your CSV/NPY/Parquet there or edit LOCAL_DATASET_PATH."
        )

    # Per-dataset split sub-folder to avoid collisions when running multiple
    # datasets in parallel.
    split_dir = source.parent / f"{source.stem}_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "Train_set.csv"
    val_path = split_dir / "Val_set.csv"
    test_path = split_dir / "Test_set.csv"

    if not all(p.exists() for p in (train_path, val_path, test_path)):
        df = _read_tabular_file(source)
        _split_3way_csv(df, cfg, train_path, val_path, test_path)
        print(f"[data] Created 3-way split for '{source.name}' in {split_dir.name}/:")
        print(f"[data]   Train: {int(cfg.train_pct*100)}%   "
              f"Val: {int(cfg.val_pct*100)}%   "
              f"Test: {int(cfg.test_pct*100)}%")
    else:
        print(f"[data] Using existing split in {split_dir.name}/")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    feature_cols, X_train, y_train, output_dim, classes = _prepare_xy(
        train_df, cfg, fit_stats=True
    )
    _, X_val, y_val, _, _ = _prepare_xy(
        val_df, cfg, fit_stats=False,
        precomputed_feature_cols=feature_cols, precomputed_classes=classes,
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train),
                                   torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val),
                                 torch.from_numpy(y_val))

    # Class-imbalance detection (classification only)
    imbalance_ratio = 1.0
    if cfg.task_type == "classification" and classes is not None:
        unique, counts = np.unique(y_train, return_counts=True)
        if len(counts) > 1 and counts.min() > 0:
            imbalance_ratio = float(counts.max()) / float(counts.min())
            counts_str = ", ".join(
                f"{int(unique[i])}={counts[i]}" for i in range(len(unique))
            )
            print(f"[data] Class distribution: {{{counts_str}}}")
            print(f"[data] Imbalance ratio: {imbalance_ratio:.1f}:1")
            if imbalance_ratio >= 10.0:
                print("[data] [INFO] High class imbalance - "
                      "auto loss_function will pick Focal Loss with gamma=2.5.")
            elif imbalance_ratio >= 3.0:
                print("[data] [INFO] Moderate class imbalance - "
                      "auto loss_function will pick Focal Loss with gamma=1.5.")

    data_info = {
        "input_dim": X_train.shape[1],
        "output_dim": output_dim,
        "n_samples": len(train_dataset),
        "task_type": cfg.task_type,
        "data_type": "tabular",
        "feature_columns": feature_cols,
        "classes": classes.tolist() if classes is not None else None,
        "test_set_path": str(test_path),
        "imbalance_ratio": imbalance_ratio,
    }

    def make_loaders(batch_size: int, val_split: float = 0.2,
                     seed: int = 42, num_workers: int = 0, **_ignored):
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers),
        )

    return make_loaders, data_info


def _read_tabular_file(source: Path) -> pd.DataFrame:
    """Read a tabular file based on its extension. Supports csv, npy, parquet."""
    suffix = source.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(source)

    if suffix == ".parquet":
        return pd.read_parquet(source)

    if suffix == ".npy":
        # NPY: assume the last column is the label, others are features.
        # User can override via FEATURE_COLUMNS / LABEL_COLUMN naming.
        arr = np.load(source)
        if arr.ndim != 2:
            raise ValueError(
                f"NPY file must be 2D (samples x features+label), got shape {arr.shape}."
            )
        n_cols = arr.shape[1]
        col_names = [f"feature_{i}" for i in range(n_cols - 1)] + ["label"]
        return pd.DataFrame(arr, columns=col_names)

    raise ValueError(
        f"Unsupported file extension: {suffix}. "
        f"Supported: .csv, .npy, .parquet"
    )


# ============================================================= TEXT

def _load_text(cfg: RunConfig):
    from collections import Counter

    source = Path(cfg.local_dataset_path)
    data_dir = source.parent
    train_path = data_dir / "Train_set.csv"
    val_path = data_dir / "Val_set.csv"
    test_path = data_dir / "Test_set.csv"

    if not all(p.exists() for p in (train_path, val_path, test_path)):
        if not source.exists():
            raise FileNotFoundError(f"Dataset '{source}' not found.")
        df = pd.read_csv(source)
        _split_3way_csv(df, cfg, train_path, val_path, test_path)
        print(f"[data] Created 3-way split for text data.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    label_col = cfg.label_column or "label"
    text_col = (cfg.feature_columns[0] if cfg.feature_columns else "text")

    for c in (label_col, text_col):
        if c not in train_df.columns:
            raise ValueError(f"Column '{c}' not found in CSV.")

    # Build vocab from train only
    texts_train = train_df[text_col].fillna("").astype(str).tolist()
    counter = Counter()
    for t in texts_train:
        counter.update(t.lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in counter.most_common(20000 - 2):
        vocab[w] = len(vocab)

    classes, y_train = np.unique(train_df[label_col].values, return_inverse=True)
    # Encode val labels with the same classes
    y_val_raw = val_df[label_col].values
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_val = np.array([class_to_idx.get(v, 0) for v in y_val_raw], dtype=np.int64)

    output_dim = int(len(classes))
    texts_val = val_df[text_col].fillna("").astype(str).tolist()

    data_info = {
        "vocab_size": len(vocab),
        "output_dim": output_dim,
        "n_samples": len(texts_train),
        "task_type": "classification",
        "data_type": "text",
        "default_seq_len": 128,
        "test_set_path": str(test_path),
    }

    def _encode(t, seq_len):
        ids = [vocab.get(w, 1) for w in t.lower().split()]
        return ids[:seq_len] + [0] * max(0, seq_len - len(ids))


    class _AugmentedTextDataset(torch.utils.data.Dataset):
        """Wraps a tensor dataset with on-the-fly text augmentation (training only)."""

        def __init__(self, X, y, augmentation: str = "none", prob: float = 0.0):
            self.X = X
            self.y = y
            self.augmentation = augmentation
            self.prob = float(prob)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            x = self.X[idx].clone()
            y = self.y[idx]
            if self.augmentation == "none" or self.prob <= 0:
                return x, y

            if self.augmentation == "token_dropout":
                # Mask tokens (set to 0 = pad token) with given probability
                mask = torch.rand(x.shape) < self.prob
                # Don't mask actual padding (already 0)
                non_pad = x != 0
                x = torch.where(mask & non_pad, torch.zeros_like(x), x)

            elif self.augmentation == "word_shuffle":
                # Shuffle tokens within small windows of size 3
                if len(x) >= 3 and torch.rand(1).item() < self.prob:
                    n = len(x)
                    win = 3
                    for i in range(0, n - win + 1, win):
                        if torch.rand(1).item() < 0.5:
                            perm = torch.randperm(win)
                            x[i:i+win] = x[i:i+win][perm]

            elif self.augmentation == "ngram_shuffle":
                # Permute n-grams (size 2-3) across the sequence.
                # More aggressive than word_shuffle: breaks long-range order
                # while preserving local n-gram structure.
                if len(x) >= 6 and torch.rand(1).item() < self.prob:
                    ngram_size = 2 if torch.rand(1).item() < 0.5 else 3
                    n = len(x)
                    n_chunks = n // ngram_size
                    if n_chunks >= 2:
                        chunks = [x[i*ngram_size:(i+1)*ngram_size]
                                  for i in range(n_chunks)]
                        perm = torch.randperm(n_chunks)
                        chunks = [chunks[i] for i in perm]
                        shuffled = torch.cat(chunks)
                        # Keep any tail that didn't fit into a full chunk
                        tail = x[n_chunks*ngram_size:]
                        x = torch.cat([shuffled, tail])
            return x, y


    def make_loaders(batch_size: int, val_split: float = 0.2,
                     seed: int = 42, seq_len: int = 128,
                     num_workers: int = 0,
                     text_augmentation: str = "none",
                     text_augmentation_prob: float = 0.0, **_ignored):
        X_tr = np.stack([_encode(t, seq_len) for t in texts_train])
        X_va = np.stack([_encode(t, seq_len) for t in texts_val])

        X_tr_t = torch.from_numpy(X_tr).long()
        X_va_t = torch.from_numpy(X_va).long()
        y_tr_t = torch.from_numpy(y_train.astype(np.int64))
        y_va_t = torch.from_numpy(y_val)

        # Train set uses augmentation; val set never augmented
        tr_ds = _AugmentedTextDataset(X_tr_t, y_tr_t,
                                        augmentation=text_augmentation,
                                        prob=text_augmentation_prob)
        va_ds = TensorDataset(X_va_t, y_va_t)
        return (
            DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers),
            DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers),
        )

    return make_loaders, data_info


# ============================================================= IMAGE

def _load_image(cfg: RunConfig):
    """
    Image folder: Data/Data-Set/class_a/..., class_b/...

    For images we do NOT physically duplicate files into Train/Val/Test
    folders (would be expensive). Instead we deterministically partition the
    indices using random_seed, and this partition is identical on every run.
    """
    from torchvision import datasets, transforms

    source = Path(cfg.local_dataset_path)
    if not source.is_dir():
        raise FileNotFoundError(
            f"For local image mode, LOCAL_DATASET_PATH must be a folder: {source}"
        )

    probe = datasets.ImageFolder(
        str(source),
        transform=transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor()
        ])
    )
    num_classes = len(probe.classes)
    data_info = {
        "input_channels": 3,
        "output_dim": num_classes,
        "n_samples": len(probe),
        "task_type": "classification",
        "data_type": "image",
        "image_size": 64,
        "classes": probe.classes,
    }

    def make_loaders(batch_size: int, val_split: float = 0.2,
                     seed: int = 42, image_size: int = 64,
                     augmentation: str = "none", **_ignored):
        base_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        aug_tf = _image_augmentation(image_size, augmentation)

        full_train = datasets.ImageFolder(str(source), transform=aug_tf)
        full_val = datasets.ImageFolder(str(source), transform=base_tf)

        n = len(full_train)
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=gen).tolist()

        n_test = int(n * cfg.test_pct)
        n_val = int(n * cfg.val_pct)
        n_train = n - n_test - n_val

        train_idx = idx[:n_train]
        val_idx = idx[n_train: n_train + n_val]
        # test_idx = idx[n_train + n_val:]  # held out

        train_ds = torch.utils.data.Subset(full_train, train_idx)
        val_ds = torch.utils.data.Subset(full_val, val_idx)
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        )

    return make_loaders, data_info


def _image_augmentation(image_size: int, augmentation: str):
    from torchvision import transforms
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    if augmentation == "none":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), norm,
        ])
    if augmentation == "light":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ToTensor(), norm,
        ])
    # medium
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(), norm,
    ])


# ============================================================= helpers

def _split_3way_csv(df: pd.DataFrame, cfg: RunConfig,
                    train_path: Path, val_path: Path, test_path: Path):
    label_col = _resolve_label_column(df, cfg)
    if cfg.feature_columns:
        missing = [c for c in cfg.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"FEATURE_COLUMNS missing: {missing}")

    seed = cfg.random_seed if cfg.random_seed is not None else 42
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * cfg.train_pct)
    n_val = int(n * cfg.val_pct)

    shuffled.iloc[:n_train].to_csv(train_path, index=False)
    shuffled.iloc[n_train: n_train + n_val].to_csv(val_path, index=False)
    shuffled.iloc[n_train + n_val:].to_csv(test_path, index=False)


def _resolve_label_column(df: pd.DataFrame, cfg: RunConfig) -> str:
    if cfg.label_column is None:
        return df.columns[-1]
    if cfg.label_column not in df.columns:
        raise ValueError(
            f"LABEL_COLUMN='{cfg.label_column}' not in: {list(df.columns)}"
        )
    return cfg.label_column


# Persistent normalization statistics (mean/std from train set)
_tabular_stats: dict = {}


def _prepare_xy(df: pd.DataFrame, cfg: RunConfig, fit_stats: bool = True,
                precomputed_feature_cols=None, precomputed_classes=None):
    """Convert a DataFrame into (feature_cols, X, y, output_dim, classes).

    When fit_stats=True (on train set), compute and store mean/std + feature_cols.
    When fit_stats=False (on val/test), reuse them. This prevents data leakage.
    """
    label_col = _resolve_label_column(df, cfg)

    if precomputed_feature_cols is not None:
        feature_cols = precomputed_feature_cols
    elif cfg.feature_columns:
        feature_cols = list(cfg.feature_columns)
    else:
        feature_cols = [c for c in df.columns if c != label_col]

    X_df = df[feature_cols]
    X_df = pd.get_dummies(X_df, drop_first=False)

    # Align columns with train set when applying to val
    if fit_stats:
        _tabular_stats["dummy_cols"] = list(X_df.columns)
    else:
        expected_cols = _tabular_stats.get("dummy_cols", list(X_df.columns))
        for c in expected_cols:
            if c not in X_df.columns:
                X_df[c] = 0
        X_df = X_df[expected_cols]

    X = X_df.values.astype(np.float32)

    # Determine normalization method from config (default: standardize)
    norm_method = "standardize"
    # Look at any config_manager attached to cfg via attribute? We don't have it here.
    # Use the field on cfg if present, else default.
    if hasattr(cfg, "_normalization_method") and cfg._normalization_method:
        norm_method = cfg._normalization_method

    if fit_stats:
        if norm_method == "standardize":
            _tabular_stats["mean"] = X.mean(axis=0)
            _tabular_stats["std"] = X.std(axis=0) + 1e-8
        elif norm_method == "min_max":
            _tabular_stats["min"] = X.min(axis=0)
            _tabular_stats["max"] = X.max(axis=0)
        elif norm_method == "max_abs":
            _tabular_stats["max_abs"] = np.abs(X).max(axis=0) + 1e-8
        _tabular_stats["norm_method"] = norm_method

    method = _tabular_stats.get("norm_method", "standardize")
    if method == "standardize":
        mean = _tabular_stats.get("mean", X.mean(axis=0))
        std = _tabular_stats.get("std", X.std(axis=0) + 1e-8)
        X = (X - mean) / std
    elif method == "min_max":
        x_min = _tabular_stats.get("min", X.min(axis=0))
        x_max = _tabular_stats.get("max", X.max(axis=0))
        denom = (x_max - x_min)
        denom[denom == 0] = 1.0
        X = (X - x_min) / denom
    elif method == "max_abs":
        x_max_abs = _tabular_stats.get("max_abs", np.abs(X).max(axis=0) + 1e-8)
        X = X / x_max_abs
    # method == "none": leave X as is

    y_raw = df[label_col].values
    if cfg.task_type == "classification":
        if precomputed_classes is not None:
            classes = precomputed_classes
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y = np.array([class_to_idx.get(v, 0) for v in y_raw], dtype=np.int64)
        else:
            classes, y = np.unique(y_raw, return_inverse=True)
            y = y.astype(np.int64)
        output_dim = int(len(classes))
    else:
        y = y_raw.astype(np.float32).reshape(-1, 1)
        output_dim = 1
        classes = None

    return feature_cols, X, y, output_dim, classes
