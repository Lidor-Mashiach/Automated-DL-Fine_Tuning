# 📦 Data Loaders

Loading, preprocessing, and batching of the dataset. The entry point is `load_data(cfg)`, which dispatches to either the local or imported loader based on `DATASET_MODE` in `main.py`.

---

## 📂 Files

| File | Role |
|---|---|
| `__init__.py` | `load_data()` dispatcher |
| `local_loader.py` | Loads from `Data/` (tabular, text, image) with a saved 3-way split |
| `imported_loader.py` | Loads torchvision / sklearn datasets from `~/.cache/autotune_nn/` |

---

## 🔀 Local vs Imported

### Local mode

User provides their own data at `Data/Data-Set.csv` (or `Data/Data-Set/` for images).

On the first run:
- CSV data is shuffled using `RANDOM_SEED`, then split into `Train_set.csv`, `Val_set.csv`, `Test_set.csv` according to `TRAIN_PCT / VAL_PCT / TEST_PCT` from `main.py`.
- The split files are persistent — subsequent runs reuse them.
- Image data uses deterministic index partitioning (no file duplication).

**Data leakage prevention:** normalization statistics (mean/std) are computed on the **train set only**, then applied to val and test.

### Imported mode

Well-known datasets are downloaded and cached in `~/.cache/autotune_nn/`. The user's `Data/` folder is never used.

Supported:
- **Images (torchvision):** `mnist`, `fashion_mnist`, `cifar10`, `cifar100`
- **Tabular (sklearn):** `iris`, `wine`, `breast_cancer`, `digits`

---

## 📤 What Each Loader Returns

Both local and imported loaders return the same interface:

```python
make_loaders, data_info = load_data(cfg)
```

- **`make_loaders(batch_size, val_split, seed, **kwargs)`** — called once per trial, returns `(train_loader, val_loader)` tuples of `torch.utils.data.DataLoader`.
- **`data_info`** — dict consumed by the model builder. Contains:
  - `input_dim` or `input_channels` / `image_size` (depending on data type)
  - `output_dim` (number of classes or `1` for regression)
  - `task_type` (`classification` / `regression`)
  - `data_type` (`tabular` / `image` / `text`)
  - Metadata like `classes`, `feature_columns`, `vocab_size`, etc.

---

## 🔍 Data Types

### Tabular (`DATA_TYPE="tabular"`)

- Input: single CSV file
- Features: numeric columns + one-hot encoded categoricals
- Target: `LABEL_COLUMN` (or last column if `None`)
- Normalization: standardization (mean=0, std=1) fit on train only

### Text (`DATA_TYPE="text"`)

- Input: CSV with `text` and `label` columns (customizable via `FEATURE_COLUMNS` / `LABEL_COLUMN`)
- Tokenization: simple lowercase word-level split
- Vocabulary: built from train set only, capped at 20,000 tokens
- Encoding: sequences padded / truncated to `sequence_length`

### Image (`DATA_TYPE="image"`)

- Input: `ImageFolder` directory (one subfolder per class)
- Normalization: ImageNet statistics
- Augmentation: controlled by `data_augmentation` in the architecture YAML (`none` / `light` / `medium`)

---

## ➕ Adding a New Data Source

To support a new imported dataset:

1. Add a `_load_X(cfg)` function in `imported_loader.py`.
2. Register it in `_REGISTRY` with its type and loader function name.
3. Document it in `main.py`'s `IMPORTED_DATASET_NAME` docstring.

To support a new local format, extend `local_loader.py` with a new `_load_<type>` branch.

---

## 🗂️ Local Tabular - Supported Formats

Local tabular data can be supplied as:

| Format | Extension | Notes |
|---|---|---|
| CSV | `.csv` | First row = headers. Last column = label by default. |
| NumPy | `.npy` | 2-D array. Last column = label. Auto-named columns. |
| Parquet | `.parquet` | Column-oriented. Pandas-readable. |

The first run creates `Train_set.csv`, `Val_set.csv`, `Test_set.csv` inside a per-dataset sub-folder named `Data/<filename>_split/` to avoid collisions when running multiple datasets in parallel.

---

## ⚖️ Class Imbalance Detection

For classification tasks, the loader computes `imbalance_ratio = max(class_counts) / min(class_counts)` and prints it to the console. This ratio is also passed to the trainer, which uses it (along with `loss_function: "auto"`) to pick Focal Loss or Cross-Entropy automatically. See [`core/README.md`](../core/README.md) for the loss-selection table.

---

## 🔢 Tabular Normalization Methods

Available via the `normalization` parameter in `configs/architectures/mlp.yaml`:

| Method | Formula | Best for | Sensitive to outliers? |
|---|---|---|---|
| `none` | x (no change) | Pre-normalized data | — |
| `standardize` | `(x - mean) / std` | Most tabular data | Moderately |
| `min_max` | `(x - min) / (max - min)` → `[0, 1]` | Bounded features | Very |
| `max_abs` | `x / max(\|x\|)` → `[-1, 1]` | Sparse data (preserves zeros) | Moderately |

Statistics are fit on the **train set only** to avoid data leakage.

---

## 🌐 Imported Datasets - Split Handling

Different imported datasets have different built-in splits:

| Dataset | Built-in split | How TRAIN_PCT/VAL_PCT/TEST_PCT are used |
|---|---|---|
| MNIST, FashionMNIST, CIFAR-10, CIFAR-100 | Train + Test | Test = original test set. Train+Val = original train set, split using TRAIN_PCT:VAL_PCT ratio. TEST_PCT ignored. |
| Iris, Wine, Breast Cancer, Digits | None | All three percentages used; must sum to 1.0. |

The console output explains exactly what was done in each run.
