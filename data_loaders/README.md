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

---

## 🎵 lyrics_loader.py (language modeling)

Specialized loader for `task_type="language_modeling"` + `data_type="lyrics"`. Auto-selected when either condition is met.

### Input format
- **CSV**: 3 columns - `artist`, `song`, `lyrics`. No header required. Lines within a song separated by a configurable token (default `&`, set via `--line_separator_token`).
- **MIDI dir**: `.mid` files named `<Artist>_-_<Song>.mid` (underscores for spaces). Optional - missing files get zero feature vectors.
- **Word2Vec**: pretrained embeddings (`.bin` via gensim, or `.txt` GloVe-style). Optional - falls back to random init.

### Output
`load_lyrics(cfg)` returns `(make_loaders, data_info)` where:
- `make_loaders(batch_size, seq_len, ...)` produces `(train_loader, val_loader)` over fixed-length token windows.
- `data_info` contains `vocab`, `embedding_matrix`, `midi_dim`, `midi_variant`, `test_songs` (for generation).

### MIDI variants
| Variant | Dim per timestep | Description |
|---|---|---|
| `none` | 0 | Lyrics-only baseline (no MIDI). |
| `simple` | 8 | Global features (tempo, duration, num_instruments, ...) - same vector at every step. |
| `per_word` | 8 | Per-timestep features (notes active at each word's time slot). Time-aligned. |

### Vocabulary
Built from training tokens; minimum frequency configurable via `--min_word_count`. Special tokens: `<pad>` (0), `<unk>` (1), `<eos>` (2). The line separator becomes a regular vocab token so the model learns when to break lines.

### Dataset class
`LyricsDataset` returns `(input_ids, target_ids, midi_features)` per item. The trainer detects 3-tuple batches and routes them to the LM forward path.

---

## 🎛️ Teacher Forcing & Line-Length Controls

Two LM-specific hyperparameters affect how lyrics are produced — one at training time, two at generation time.

### `teacher_forcing_ratio` (training-time)

Pure parallel-LSTM training feeds the ground-truth previous token at every step (`tf_ratio=1.0`). This is fast but creates a train/test mismatch: at generation, the model sees its own outputs, which may diverge from clean ground-truth.

The trainer implements a pragmatic approximation: with probability `(1 - tf_ratio)`, individual input tokens are replaced by `<unk>`, forcing the model to learn how to recover from noisy context. Lower values = more robust generation but slower convergence.

| Value | Behavior |
|---|---|
| `1.0` | Always feed ground truth (classic LM training). |
| `0.7` | 30% of input tokens dropped to `<unk>`. |
| `0.5` | 50% — model relies more on its own representations. |
| `0.3` | Aggressive — exposes model to substantial noise. |

Applied only during training (`is_train=True`); validation always uses `tf_ratio=1.0`.

### `max_words_per_line` and `min_words_per_line` (generation-time)

Hard constraints applied in `generate_lyrics()`:
- **`max_words_per_line`**: if the current line reaches this length without sampling `<line_sep>`, the generator forces the next token to be `<line_sep>`.
- **`min_words_per_line`**: if the model samples `<line_sep>` before the line has this many words, the separator's logit is set to `-inf` and a different token is sampled.

These are deterministic constraints, not loss-based — they only influence generation, not training, and don't require retraining when changed.

The defaults (`max=12`, `min=2`) match typical pop-song line lengths. The Analyzer doesn't propose actions for these since they're generation hyperparameters, not training ones.
