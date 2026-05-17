# 🛠️ Setup Guide

Complete walkthrough for new users. Read top-to-bottom for first-time setup, or jump to your architecture using the table of contents.

> 📖 For project overview, see [`README.md`](README.md). This guide covers the **practical** side.

---

## 📚 Table of Contents

### Part 1: Getting Ready
1. [Install Dependencies](#1-install-dependencies)
2. [Prepare Your Data](#2-prepare-your-data)
3. [Configure main.py](#3-configure-mainpy)
4. [Choose Your Architecture](#4-choose-your-architecture)

### Part 2: How to Read the Architecture YAMLs
5. [Parameter Format - read this first](#5-parameter-format-read-this-first)

### Part 3: Architecture Configuration
6. [MLP — for tabular data](#6-mlp-configuration)
7. [CNN — for images](#7-cnn-configuration)
8. [RNN / LSTM — for sequences](#8-rnn--lstm-configuration)
9. [Transformer — for advanced sequences/NLP](#9-transformer-configuration)

### Part 4: Sizing & Run
10. [Sizing your network](#10-sizing-your-network)
11. [Strategy Configuration](#11-strategy-configuration)
12. [Run It](#12-run-it)

---

# Part 1: Getting Ready

## 1. Install Dependencies

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> 💡 GPU users: install CUDA-enabled PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) **before** running `pip install -r requirements.txt`.

---

## 2. Prepare Your Data

Choose **one** of two modes:

### Option A — Local Data

Place your file in `Data/`. Supported formats:

| Type | Extension | Notes |
|---|---|---|
| Tabular | `.csv` | first row = headers, last column = label by default |
| Tabular | `.npy` | 2-D NumPy array, last column = label |
| Tabular | `.parquet` | columnar, pandas-readable |
| Image | folder | ImageFolder layout (one subfolder per class) |
| Text | `.csv` | columns: `text`, `label` |

On the first run, AutoTune-NN auto-creates a sub-folder `<your_file>_split/` with `Train_set.csv`, `Val_set.csv`, `Test_set.csv`. Subsequent runs reuse them.

> 💡 Multiple datasets in parallel = each gets its own split sub-folder; no collisions.

### Option B — Imported Dataset

Skip data prep — auto-downloads to `~/.cache/autotune_nn/`. Choose one:

- **Image**: `mnist`, `fashion_mnist`, `cifar10`, `cifar100`
- **Tabular**: `iris`, `wine`, `breast_cancer`, `digits`

---

## 3. Configure main.py

Open `main.py`. These are the **top-level** choices (everything else is in the YAMLs):

| Setting | What | Example |
|---|---|---|
| `RUN_NAME` | Short label, used in output folder name | `"experiment1"` |
| `ARCHITECTURE` | One of: `mlp`, `cnn`, `rnn`, `lstm`, `transformer` | `"mlp"` |
| `SEARCH_STRATEGY` | `ftts` (default), `bayesian`, `grid` | `"ftts"` |
| `TASK_TYPE` | `classification` or `regression` | `"classification"` |
| `DATASET_MODE` | `local` or `imported` | `"local"` |
| `LOCAL_DATASET_PATH` | Path to local file (used only when `local`) | `"./Data/Data-Set.csv"` |
| `DATA_TYPE` | `tabular`, `image`, `text` (used only when `local`) | `"tabular"` |
| `FEATURE_COLUMNS` | Column names list, or `None` for all-but-label | `None` |
| `LABEL_COLUMN` | Column name, or `None` for last column | `None` |
| `TRAIN_PCT` / `VAL_PCT` / `TEST_PCT` | Split ratios; sum = 1.0 | `0.6 / 0.2 / 0.2` |
| `IMPORTED_DATASET_NAME` | Imported dataset name (used only when `imported`) | `"mnist"` |
| `DEVICE` | `auto`, `gpu`, `cpu` | `"auto"` |
| `RANDOM_SEED` | Reproducibility seed, or `None` | `42` |
| `EXPERIMENTS_ROOT` | Output root | `"./experiments"` |

### FAQ on main.py

- **`FEATURE_COLUMNS=None` - does it skip the label column automatically?** Yes. `None` means "all columns except the label".
- **What about imported datasets?** `FEATURE_COLUMNS` and `LABEL_COLUMN` are ignored - imported datasets have a fixed structure. Leave both as `None`.
- **In `local` mode, do I need `IMPORTED_DATASET_NAME`?** No - ignored. Leave the default value.
- **In `imported` mode, do I need `LOCAL_DATASET_PATH`?** No - ignored.
- **Built-in test split (e.g. MNIST has its own test set) - what happens to `TEST_PCT`?** The original test set is kept, and `TRAIN_PCT:VAL_PCT` becomes the ratio for splitting the original train into train+val. `TEST_PCT` is ignored. The console will tell you what was done.

---

## 4. Choose Your Architecture

| Your data... | → use | Why |
|---|---|---|
| **Tabular** (CSV/spreadsheet) | **MLP** | Dense layers - the standard for structured data |
| **Images** | **CNN** | Captures spatial patterns via convolutions |
| **Short sequences** (<50 tokens) | **RNN** | Simple, fast |
| **Medium-long sequences**, **basic NLP** | **LSTM** | Solves vanishing gradients of RNN |
| **Long-range NLP**, **complex tasks** | **Transformer** | Attention captures long-range dependencies |

```
What does each row of your data represent?
│
├── A vector of features        → MLP
├── An image                    → CNN
└── A sequence of tokens
    ├── < 50 tokens             → RNN
    ├── 50-500 tokens            → LSTM
    └── > 500 tokens or NLP     → Transformer
```

---

# Part 2: How to Read the Architecture YAMLs

## 5. Parameter Format (read this first)

Each parameter in a YAML follows the same shape:

```yaml
parameter_name:
  enabled: true              # turn the parameter on (true) or off (false)
  initial_value: 0.2         # the starting value for the search
  range: [0.0, 0.5]          # for continuous params (use either range OR choices)
  # OR
  choices: ["a", "b", "c"]   # for discrete params
```

### What you can do with any parameter

| Goal | How to set it |
|---|---|
| Use it, with a specific starting value | `enabled: true` + set `initial_value` |
| Use it, let the system pick a starting value | `enabled: true` + `initial_value: null` |
| Disable it completely (won't be used at all) | `enabled: false` |
| Force one value forever (no search) | Continuous: `range: [x, x]`. Discrete: `choices: ["x"]` |

### Important: Values and search behavior

- **`initial_value` is the starting point of the search, not a fixed value.** The Analyzer can move parameters across `range` or `choices` during the run.
- **Discrete `choices`**: the Analyzer can pick any element. To force one value, set `choices: ["only_this_one"]`.
- **Continuous `range`**: the Analyzer adjusts the value step-by-step within `[min, max]`.

> ⚠️ **Don't enable everything.** Each enabled parameter expands the search space. Defaults are sensible. Turn things on selectively, based on your problem and budget.

### How the table columns work in this guide

Each architecture has one big table covering everything:

| Column | Meaning |
|---|---|
| **Parameter** | YAML key |
| **What & Effect** | What it controls + what increasing/changing it does |
| **Type** | `range` (continuous), `choices` (discrete), `bool` |
| **Default** | `enabled / initial_value / range or choices` |
| **When to change** | Concrete advice |

---

# Part 3: Architecture Configuration

## 6. MLP Configuration

**File**: `configs/architectures/mlp.yaml`. **For tabular data** - each row is a vector of features.

| Parameter | What & Effect | Type | Default (enabled / initial / range\|choices) | When to change |
|---|---|---|---|---|
| `num_hidden_layers` | Number of hidden layers. More = more expressive but harder to train. | range | `true / 2 / [1, 5]` | Larger datasets (1M+ rows): widen to `[1, 8]`. |
| `hidden_size` | **Base** width of layers (the actual width of each layer is derived by `layer_shape`). Wider = more capacity. | choices | `true / 128 / {32, 64, 128, 256, 512}` | Tiny tasks (<1K rows): start at `32` or `64`. Huge datasets: add `1024`. |
| `layer_shape` | Pattern that derives the width of each layer from `hidden_size`. **The Analyzer can switch patterns during search**. See "Layer shape patterns" below. | choices | `true / uniform / {uniform, funnel, pyramid, hourglass, bottleneck}` | Narrow `choices` to fix a single pattern; `bottleneck` for autoencoder-like. |
| `activation` | Nonlinearity in every layer. `relu` safe; `gelu` smoother; `tanh` for bounded. | choices | `true / relu / {relu, gelu, leaky_relu, tanh, elu, selu, silu}` | Narrow `choices` if you want to avoid an alternative. |
| `dropout` | Random unit zeroing. Higher = stronger regularization. | range | `true / 0.2 / [0.0, 0.5]` | Very small datasets: widen up to `0.7`. |
| `batch_norm` | Normalizes activations. `true` accelerates training. | bool | `true / false / {true, false}` | Set initial=`true` for deeper nets (4+ layers). |
| `weight_decay` | L2 regularization (log scale). Higher = stronger. | range | `true / 1e-4 / [1e-6, 1e-2]` | Overfitting: try `1e-3`. Underfitting: try `0`. |
| `optimizer_name` | Optimizer choice. | choices | `true / adam / {adam, adamw, sgd, rmsprop}` | Narrow to `["adamw"]` for L2-aware decay. |
| `momentum` | Momentum for SGD/RMSprop (ignored for Adam). | range | `true / 0.9 / [0.0, 0.99]` | Default works well; rarely tuned. |
| `learning_rate` | Step size (log scale). Too high = unstable; too low = slow. | range | `true / 1e-3 / [1e-5, 1e-1]` | Narrow to `[1e-4, 1e-3]` if you know the right magnitude. |
| `lr_scheduler` | LR decay strategy. | choices | `true / none / {none, cosine, step, reduce_on_plateau}` | Long runs (50+ epochs): set initial=`cosine`. |
| `gradient_clipping` | Caps gradient norm. `0.0` disables. | range | `true / 0.0 / [0.0, 5.0]` | Set non-zero (e.g. `1.0`) if you see exploding gradients (NaN). |
| `batch_size` | Samples per gradient update. Larger = stable, more RAM. | choices | `true / 64 / {16, 32, 64, 128, 256}` | Limited GPU RAM: narrow to `{16, 32, 64}`. |
| `epochs` | Full data passes. | range | `true / 50 / [10, 200]` | Long training: widen to `[20, 500]`. |
| `validation_split` | In-trial val for early stopping (different from main.py's VAL_PCT). | range | `true / 0.2 / [0.1, 0.4]` | Rarely tuned; default fine. |
| `loss_function` | Training objective. `auto` picks per task & class balance. | choices | `true / auto / {auto, cross_entropy, focal, mse}` | Force `cross_entropy` if you don't trust auto detection. |
| `focal_gamma` | Focusing strength for Focal Loss (used only when `loss_function=focal`). γ=2 paper default. | range | `true / 2.0 / [0.5, 5.0]` | Heavy class imbalance (>10:1): widen to `[2.0, 5.0]`. |
| `normalization` | Feature scaling. | choices | `true / standardize / {none, standardize, min_max, max_abs}` | Bounded features (percentages): `min_max`. Sparse data: `max_abs`. |
| `num_workers` | DataLoader parallel workers (system setting; Analyzer doesn't tune this). | range | `true / 0 / [0, 8]` | Very large CSV files: try `2-4`. |
| `gradient_accumulation_steps` | Effective batch = `batch_size × steps`. | range | `true / 1 / [1, 8]` | Limited GPU but want effective batch ≥ 256. |
| `mixed_precision` | fp16 on GPU = ~2x speedup, halved VRAM. | bool | `true / false / {true, false}` | GPU users: set initial=`true`. |
| `adam_beta1` | Adam β1 (PyTorch always uses 0.9 internally). | range | **`false`** / 0.9 / [0.85, 0.95] | Enable only for advanced tuning of huge runs. |
| `adam_beta2` | Adam β2 (PyTorch always uses 0.999 internally). | range | **`false`** / 0.999 / [0.9, 0.9999] | Enable only for advanced tuning of huge runs. |
| `early_stopping` | Stops a single trial when val loss plateaus. | bool | `true / true / {true, false}` | Disable to always train full epochs. |
| `label_smoothing` | Softens hard labels. | range | `true / 0.0 / [0.0, 0.2]` | Many-class classification (10+): try `0.1`. |

### Layer shape patterns (with `hidden_size=128, num_hidden_layers=4`)

| Shape | Layer widths | Use case |
|---|---|---|
| `uniform` | 128 → 128 → 128 → 128 | Default; equal capacity per layer |
| `funnel` | 128 → 64 → 32 → 16 | Gradual compression; good for high-dim → low-dim |
| `pyramid` | 16 → 32 → 64 → 128 | Gradual expansion; useful from low-dim inputs |
| `hourglass` | 64 → 128 → 128 → 64 | Rich middle representation |
| `bottleneck` | 128 → 64 → 64 → 128 | Autoencoder-like; forces compression at the middle |

> 💡 **Context-aware proposals**: The Analyzer doesn't just cycle through shapes — it picks specific patterns based on the verdict:
> - **Overfit** → suggests `bottleneck` (0.55) or `funnel` (0.40) — both compress information.
> - **Failed to learn / underfit** → suggests `hourglass` (0.55) — extra middle capacity. Or `uniform` (0.40) — safest fallback.
> - **Slow training** → suggests `funnel` (0.45) — fewer parameters in deep layers.
> - **Healthy** → low-priority exploration: `bottleneck` (0.30) or `pyramid` (0.25).
>
> Suggestions matching the current shape are skipped (no-op). The system also avoids re-evaluating the same hyperparameter combination via different action paths (DAG-dedup).

---

## 7. CNN Configuration

**File**: `configs/architectures/cnn.yaml`. **For images** (ImageFolder layout or imported image datasets).

| Parameter | What & Effect | Type | Default (enabled / initial / range\|choices) | When to change |
|---|---|---|---|---|
| `num_conv_blocks` | Conv+Pool blocks. Each halves the spatial size. More = deeper. | range | `true / 3 / [2, 5]` | High-res images (128px+): widen to `[3, 7]`. |
| `base_filters` | Filters in 1st block; 2× per block (32→64→128). Higher = more capacity. | choices | `true / 32 / {16, 32, 64}` | Tiny images (28×28): start at `16`. |
| `kernel_size` | Conv kernel NxN. | choices | `true / 3 / {3, 5}` | Larger spatial patterns: include `5`. |
| `pooling` | Spatial reduction. | choices | `true / max / {max, avg}` | `avg` for smoother feature reduction. |
| `num_fc_layers` | FC layers in head. | range | `true / 1 / [1, 3]` | Usually 1 is enough; rarely change. |
| `fc_size` | Width of FC head. | choices | `true / 256 / {128, 256, 512}` | Large datasets: include `1024`. |
| `activation` | Conv & FC nonlinearity. | choices | `true / relu / {relu, gelu, leaky_relu}` | `gelu` for very deep nets. |
| `dropout` | Dropout in FC head only (not conv). | range | `true / 0.3 / [0.0, 0.5]` | Strong overfit: widen to `[0.0, 0.7]`. |
| `batch_norm` | Normalizes conv activations. Almost always true for CNNs. | bool | `true / true / {true, false}` | Very small batches (< 16): set false. |
| `data_augmentation` | Image augmentation strength. `light` = flip+crop; `medium` = adds rotation. | choices | `true / light / {none, light, medium}` | Small datasets: set initial=`medium`. |
| `mixup` | Blends 2 samples (with mixed labels). 0=disabled. | range | `true / 0.0 / [0.0, 0.5]` | Image classification with limited data: enable, try 0.2. |
| `cutout` | Masks random patch (mask = `cutout × image_size`). 0=disabled. | range | `true / 0.0 / [0.0, 0.4]` | Image overfit: enable, try 0.25. |
| `cutmix` | Cuts a patch from one image, pastes into another, mixes labels. 0=disabled. | range | `true / 0.0 / [0.0, 1.0]` | Strong alternative to mixup. Paper default α=1.0. |
| `weight_decay` | L2 regularization. | range | `true / 1e-4 / [1e-6, 1e-2]` | Overfit: try `1e-3`. |
| `optimizer_name` | Optimizer. | choices | `true / adamw / {adam, adamw, sgd}` | `sgd` traditional for image classification fine-tuning. |
| `learning_rate` | Step size (log). | range | `true / 1e-3 / [1e-4, 1e-2]` | Narrow if you know your magnitude. |
| `lr_scheduler` | LR decay. Strongly recommended for images. | choices | `true / cosine / {cosine, step, reduce_on_plateau}` | Use `cosine` (default) for long training. |
| `batch_size` | Samples per update. | choices | `true / 64 / {32, 64, 128}` | Large GPU: include `256`. |
| `epochs` | Full data passes. | range | `true / 50 / [10, 200]` | |
| `image_size` | Resize images to N×N. | range | `true / 64 / [16, 224]` | Match your dataset's natural resolution. |
| `loss_function` | Training objective. | choices | `true / auto / {auto, cross_entropy, focal, mse}` | Force `cross_entropy` if needed. |
| `focal_gamma` | γ for Focal Loss. | range | `true / 2.0 / [0.5, 5.0]` | Class imbalance >10:1: widen high. |
| `num_workers` | Parallel batch loaders (system setting). | range | `true / 2 / [0, 8]` | Large datasets: try 4-8. |
| `gradient_accumulation_steps` | Effective batch = `batch_size × steps`. | range | `true / 1 / [1, 8]` | Limited GPU but want big effective batch. |
| `mixed_precision` | fp16 on GPU = ~2x speedup. | bool | `true / false / {true, false}` | GPU users: set initial=`true`. |
| `adam_beta1` | Adam β1 (default 0.9 used internally). | range | **`false`** / 0.9 / [0.85, 0.95] | Advanced tuning only. |
| `adam_beta2` | Adam β2 (default 0.999 used internally). | range | **`false`** / 0.999 / [0.9, 0.9999] | Advanced tuning only. |
| `label_smoothing` | Softens hard labels. | range | `true / 0.1 / [0.0, 0.2]` | Many-class CIFAR-100, ImageNet. |
| `early_stopping` | Per-trial early stop. | bool | `true / true / {true, false}` | |

> 💡 For CNN, `num_workers=2-4` is recommended (data loading is the bottleneck). On GPU, `mixed_precision=true` typically gives ~2x speedup.

---

## 8. RNN / LSTM Configuration

**Files**: `configs/architectures/rnn.yaml`, `configs/architectures/lstm.yaml`. **For sequences/text**.

| Parameter | What & Effect | Type | RNN default | LSTM default | When to change |
|---|---|---|---|---|---|
| `hidden_size` | Width of recurrent layers (uniform across stack). | choices | 128 | 128 | Larger vocab/seq: include `512`. |
| `num_layers` | Stacked recurrent layers. | range | `[1, 3]` initial 1 | initial 2 | Rarely useful beyond 3. |
| `bidirectional` | Read sequence both directions. | bool | initial false | initial false | Sentence classification: set true. Streaming: keep false. |
| `embedding_dim` | Embedding vector size. | choices | 128 / `{64, 128, 256}` | same | Larger vocab: include `512`. |
| `dropout` | Between stacked recurrent layers. Only effective if `num_layers > 1`. | range | initial 0.2 | initial 0.3 | Both `[0.0, 0.5]`. Overfit: widen high. |
| `embedding_dropout` | Dropout on embedding output. Strong NLP regularizer. | range | 0.0 | 0.0 | Small datasets: try 0.2. |
| `optimizer_name` | Optimizer. | choices | `adam / {adam, adamw, rmsprop}` | same | RNNs: `rmsprop` traditional. |
| `learning_rate` | Step size (log). RNNs are LR-sensitive. | range | `1e-3 / [1e-4, 1e-2]` | same | |
| `lr_scheduler` | LR decay. | choices | initial `none` | initial `cosine` | Both have `{none, cosine, reduce_on_plateau}`. |
| `batch_size` | Samples per update. | choices | 64 / `{32, 64, 128}` | same | Long sequences: smaller batches. |
| `epochs` | Full data passes. | range | initial 40 | initial 50 | Both `[10, 200]`. |
| `sequence_length` | Fixed length (truncate/pad). | range | 128 | 128 | Match data: NLP often uses 256-512. |
| `validation_split` | In-trial val. | range | 0.2 | 0.2 | |
| `text_augmentation` | Sequence augmentation. See "Text augmentation options" below. | choices | initial `none` | initial `none` | Small NLP datasets: try `token_dropout`. |
| `text_augmentation_prob` | Probability of perturbation when active. | range | 0.1 | 0.1 | Both `[0.0, 0.3]`. |
| `loss_function` | Training objective. | choices | `auto / {auto, cross_entropy, focal, mse}` | same | |
| `focal_gamma` | γ for Focal Loss. | range | 2.0 / `[0.5, 5.0]` | same | |
| `num_workers` | Parallel batch loaders. | range | 2 / `[0, 8]` | same | |
| `gradient_accumulation_steps` | Effective batch multiplier. | range | 1 / `[1, 8]` | same | |
| `mixed_precision` | fp16 on GPU. | bool | false / `{true, false}` | same | GPU users: set true. |
| `adam_beta1` | Adam β1. | range | **`false`** / 0.9 / [0.85, 0.95] | same | |
| `adam_beta2` | Adam β2. | range | **`false`** / 0.999 / [0.9, 0.9999] | same | |

### Text augmentation options

| Choice | What it does | Use case |
|---|---|---|
| `none` | No augmentation | Default |
| `token_dropout` | Randomly mask tokens (replace with pad token) at rate `text_augmentation_prob` | Strongest regularizer; like word-level dropout |
| `word_shuffle` | Shuffle tokens within tiny windows of size 3 | Mild perturbation; preserves locality |
| `ngram_shuffle` | Shuffle 2-3-grams across the entire sequence | Aggressive structural shuffle; breaks long-range order |

---

## 9. Transformer Configuration

**File**: `configs/architectures/transformer.yaml`. **For complex NLP**, long-range dependencies, state-of-the-art language tasks.

| Parameter | What & Effect | Type | Default (enabled / initial / range\|choices) | When to change |
|---|---|---|---|---|
| `d_model` | Hidden dimension - vector size representing each token. Must be divisible by `nhead`. | choices | `true / 128 / {128, 256, 512}` | More data: include `768` or `1024`. |
| `nhead` | Number of attention heads. | choices | `true / 4 / {2, 4, 8}` | Pair with `d_model` such that d_model % nhead == 0. |
| `num_encoder_layers` | Stacked encoder layers. Deeper = more abstract features. | range | `true / 4 / [2, 6]` | Lots of data: widen to `[4, 12]` or even `[4, 24]`. |
| `dim_feedforward` | FFN hidden size (usually 4× d_model). | choices | `true / 512 / {256, 512, 1024}` | Match d_model scaling. |
| `embedding_dim` | Token embedding size (often = d_model). | choices | `true / 128 / {64, 128, 256, 512}` | Keep equal to or below d_model. |
| `activation` | FFN activation. | choices | `true / gelu / {relu, gelu}` | gelu is the modern default. |
| `dropout` | General dropout (FFN + outputs). | range | `true / 0.1 / [0.0, 0.5]` | Larger nets: widen to 0.2-0.3. |
| `attention_dropout` | Dropout on attention weights. | range | `true / 0.1 / [0.0, 0.3]` | Reduce overfit on attention. |
| `embedding_dropout` | Dropout on embeddings. | range | `true / 0.0 / [0.0, 0.4]` | Small NLP datasets: try 0.2. |
| `stochastic_depth` | Randomly drops entire encoder layers. **Only effective when `num_encoder_layers >= 4`**. | range | `true / 0.0 / [0.0, 0.3]` | Deep transformers (12+ layers): try 0.1-0.2. |
| `weight_decay` | L2 regularization. | range | `true / 1e-4 / [1e-6, 1e-2]` | Use with adamw. |
| `label_smoothing` | Softens hard labels. | range | `true / 0.0 / [0.0, 0.2]` | Many-class classification (10+). |
| `optimizer_name` | Optimizer. | choices | `true / adamw / {adam, adamw}` | adamw strongly preferred. |
| `learning_rate` | Step size (log). Transformers very LR-sensitive. | range | `true / 1e-4 / [1e-5, 1e-3]` | Lower than other architectures. |
| `lr_scheduler` | LR decay. | choices | `true / cosine / {cosine, reduce_on_plateau}` | Cosine strongly preferred. |
| `lr_warmup` | Gradual LR ramp at start. ESSENTIAL for transformers. | bool | `true / true / {true, false}` | Almost never disable. |
| `batch_size` | Samples per update. | choices | `true / 32 / {16, 32, 64}` | Limited GPU: narrow to `{16, 32}`. |
| `epochs` | Full data passes. | range | `true / 40 / [10, 200]` | |
| `sequence_length` | Fixed length. | range | `true / 128 / [32, 1024]` | NLP often uses 256-512. |
| `text_augmentation` | Sequence augmentation. | choices | `true / none / {none, token_dropout, word_shuffle, ngram_shuffle}` | Small NLP datasets: try `token_dropout`. |
| `text_augmentation_prob` | Probability when active. | range | `true / 0.1 / [0.0, 0.3]` | |
| `loss_function` | Training objective. | choices | `true / auto / {auto, cross_entropy, focal, mse}` | |
| `focal_gamma` | γ for Focal Loss. | range | `true / 2.0 / [0.5, 5.0]` | |
| `num_workers` | Parallel batch loaders. | range | `true / 2 / [0, 8]` | |
| `gradient_accumulation_steps` | Effective batch multiplier. | range | `true / 1 / [1, 8]` | Limited GPU but big batch needed. |
| `mixed_precision` | fp16 on GPU. | bool | `true / false / {true, false}` | **Highly recommended for transformers on GPU.** |
| `adam_beta1` | Adam β1 (default 0.9 used). | range | **`false`** / 0.9 / [0.85, 0.95] | Advanced. |
| `adam_beta2` | Adam β2 (default 0.999 used). | range | **`false`** / 0.999 / [0.9, 0.9999] | Advanced; GPT-style runs use 0.95. |

### Safe `d_model` / `nhead` combinations

| d_model | nhead options |
|---|---|
| 128 | 2, 4, 8 |
| 256 | 2, 4, 8 |
| 512 | 2, 4, 8 |
| 768 | 2, 4, 8 |
| 1024 | 2, 4, 8 |

The system auto-corrects if a bad combination is sampled (picks the largest divisor of d_model).

---

# Part 4: Sizing & Run

## 10. Sizing Your Network

How deep, how wide? Practical guide.

### Rule of thumb by problem type

| Problem | Architecture | Layers | Width |
|---|---|---|---|
| Tiny tabular (Iris, <1K rows) | MLP | 1-2 | 32-64 |
| Small tabular (1K-10K rows) | MLP | 2-3 | 64-128 |
| Medium tabular (10K-1M rows) | MLP | 3-4 | 128-256 |
| Large tabular (1M+ rows) | MLP | 4-5 | 256-512 |
| MNIST-like (28x28 grayscale) | CNN | 2-3 conv blocks | base_filters=32 |
| CIFAR-like (32x32 color) | CNN | 3-4 conv blocks | base_filters=32-64 |
| Larger images (64-128 px) | CNN | 4-5 conv blocks | base_filters=64 |
| Short sequences (<50 tokens) | RNN | 1-2 | hidden_size=64-128 |
| Medium sequences (50-200 tokens) | LSTM | 1-2 | hidden_size=128-256 |
| Long text / NLP | Transformer | 2-4 | d_model=128-256 |
| Deep NLP / lots of data | Transformer | 4-6 | d_model=256-512 |
| Production-grade NLP | Transformer | 6-12 | d_model=256-512 |

### Want FTTS to explore deeper?

By default, ranges are conservative. Edit the YAML:

```yaml
num_encoder_layers:
  enabled: true
  initial_value: 4
  range: [2, 24]      # was [2, 6] - now FTTS can go up to 24 layers
```

> 💡 **128 GB RAM users**: feel free to raise `range` widely. The FTTS tree itself is tiny (~50 MB even after 25,000 trials). Memory cost is dominated by the actively-training model.

### When to raise a range vs. wait

- If FTTS keeps suggesting `add_depth` until hitting the cap and quality keeps improving, **raise the cap** and rerun.
- If quality plateaus before hitting the cap, the range is wide enough.

---

## 11. Strategy Configuration

**File**: `configs/strategies/<strategy>.yaml`.

### Profile comparison

| Profile | best_metric | stability | speed | gap | When to use |
|---|---|---|---|---|---|
| `performance` | 70% | 10% | 10% | 10% | Just want top accuracy |
| `balanced` | 50% | 20% | 15% | 15% | Default; good tradeoff |
| `robust` | 35% | 30% | 15% | 20% | Production; reliability |

### Ranking metric (which trial picks as final best)

| Value | Behavior | When to use |
|---|---|---|
| `quality_score` | Composite score (default) | **Recommended**; picks stable, well-generalizing models |
| `smoothed_accuracy` | Raw smoothed val accuracy | Benchmark scenarios |
| `raw_accuracy` | Peak single-epoch | Not recommended; sensitive to spikes |

> ℹ️ FTTS always uses `quality_score` internally during search; `ranking_metric` only changes which trial is reported as final best.

### Stopping conditions

The run stops as soon as **any** of these conditions fires. **At least one must be set** (non-`null`); the system enforces this at startup. Set unwanted conditions to `null` to disable them.

```yaml
stopping:
  max_trials: 50              # stop after this many trials
  time_limit_minutes: null    # stop after this many wall-clock minutes
  target_accuracy: null       # stop when smoothed val accuracy reaches this
  convergence_patience: 40    # stop after N trials with no quality improvement
```

| Condition | Type | Range / format | What it does | When to use | How to disable |
|---|---|---|---|---|---|
| `max_trials` | int | 1 - 10000+ | Hard cap on total trials. | Always recommended as a safety net (prevents runaway runs). | Set to `null`. |
| `time_limit_minutes` | int / float | 1 - any (minutes) | Wall-clock time budget. Counts time across all trials. | Time-boxed runs (overnight, lunch break). | Set to `null`. |
| `target_accuracy` | float | 0.0 - 1.0 | Stops when smoothed val accuracy ≥ this value. Compared on smoothed metric (not raw, to avoid stopping on a spike). | When you have a clear "good enough" threshold (e.g. 0.95 for MNIST). | Set to `null`. |
| `convergence_patience` | int | 5 - 200 | Stops after N consecutive trials with no quality_score improvement. | Always useful - prevents infinite tuning when FTTS has plateaued. | Set to `null`. |

> ⚠️ All four set to `null` is **not allowed** - the run would never stop. The orchestrator validates this at startup.

#### Recommended combinations

| Goal | max_trials | time_limit_minutes | target_accuracy | convergence_patience |
|---|---|---|---|---|
| Quick prototype (15 min) | 30 | 15 | null | 10 |
| Full tune (overnight) | 200 | 480 | null | 40 |
| Stop at known target | 100 | null | 0.95 | 30 |
| Production tune | 500 | null | null | 80 |

### Execution

```yaml
execution:
  max_parallel_experiments: 1
```

Keep at 1 for GPU. Scale up only on CPU with enough RAM.

---

## 12. Run It

### Activate the virtual environment first

```bash
# Linux / macOS
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat
```

### Run

```bash
python main.py
```

### Run in the background (long tunes)

```bash
# Linux / macOS
nohup python main.py > run.log 2>&1 &
echo "PID: $!"

# Windows (PowerShell)
Start-Process powershell -ArgumentList "python main.py" -RedirectStandardOutput "run.log" -RedirectStandardError "err.log"
```

### Stopping early

`Ctrl+C` is safe. The `tuning/report.txt` is updated after every trial.

### What you'll see during the run

**Run setup (once at start):**
```
[orchestrator] Starting run: experiment1_ftts_iris_qualityscore
[orchestrator] Architecture: mlp
[orchestrator] Strategy:     ftts
[orchestrator] Device:       cuda:0
[data] Class distribution: {0: 50, 1: 50, 2: 50}
[data] Imbalance ratio: 1.0:1
```

**Per-trial progress (every 5 epochs + last + diagnosis + child preview):**
```
[T0001]   epoch   5/50  loss=0.2341  metric=0.8234
[T0001]   epoch  10/50  loss=0.1856  metric=0.8567
[T0001] completed     raw=0.9203 smooth=0.9145 quality=0.847 verdict=overfit children=4 (root) [NEW BEST]
[T0001]   -> proposed: increase_dropout              priority=0.85 expected=0.720
[T0001]   -> proposed: try_layer_shape_bottleneck    priority=0.55 expected=0.466
[T0001]   -> proposed: increase_weight_decay         priority=0.80 expected=0.678

[T0002]   epoch   5/50  loss=0.2102  metric=0.8512
[T0002] completed     raw=0.9301 smooth=0.9223 quality=0.871 verdict=healthy children=5 <- T0001 [NEW BEST]
[T0002]   -> proposed: decrease_lr                   priority=0.60 expected=0.523
...
[ftts] dedup: skipping action 'try_layer_shape_uniform' from T0008 - target HP already explored.
```

**Final Phase (after all trials):**
```
[orchestrator] === Final Phase ===
[orchestrator] Best trial: T0042
[final]   refit epoch 10/50: loss=0.18 metric=0.89
[final] Test results: loss=0.21 metric=0.87
[final] Generated standalone code -> model.py
```

### Output

```
experiments/<RUN_NAME>_<strategy>_<dataset>_<ranking>/
├── tuning/
│   ├── report.txt          ← every trial logged
│   └── best_trial.png      ← curves of the best tuning trial
└── final/
    ├── model.py            ← standalone code of the best architecture
    ├── final_training.png  ← final refit training curves
    └── test_evaluation.txt ← Test_set evaluation summary
```

> 📦 `final/model.py` is the actual deliverable - runs end-to-end on its own.
> ℹ️ The output folder is created at the **start** of the run and updated in-place. There's no proliferation of folders even when FTTS finds a new "best" trial mid-run.

---

# Part 5: Language Modeling Mode (lyrics generation)

## 13. Language Modeling

A specialized mode for next-word prediction tasks (e.g. Assignment 3 lyrics generation). Activated by `task_type="language_modeling"` + `data_type="lyrics"`. Only `lstm` is fully supported as the architecture (rnn/transformer can be extended similarly).

### When to use

- Predict the next word in a sequence from previous words.
- Loss metric (lower = better) instead of accuracy.
- Optionally combine text input with an auxiliary modality (MIDI features).

### Quick start

```bash
# Tune (training + generation in one command)
python main.py \
  --architecture lstm \
  --run_name lyrics_baseline \
  --task_type language_modeling \
  --data_type lyrics \
  --local_dataset_path ./Data/lyrics_train_set.csv \
  --midi_dir ./midi_files \
  --midi_variant simple \
  --word2vec_path ./GoogleNews-vectors-negative300.bin \
  --sampling_strategy nucleus \
  --sampling_top_p 0.9 \
  --melody_probe true

# Generate-only (skip training, use checkpoint)
python main.py \
  --mode generate \
  --checkpoint ./experiments/lyrics_baseline_*/final/model_checkpoint.pt \
  --initial_words love the morning \
  --sampling_strategy temperature \
  --sampling_temperature 0.8
```

### MIDI variants

| `--midi_variant` | Behavior | Use case |
|---|---|---|
| `none` | Lyrics-only baseline (no MIDI features). | Required by the assignment as a control. |
| `simple` | 8-dim global features (tempo, duration, # instruments, ...). Same vector at every timestep. | Tests whether song-level structure helps. |
| `per_word` | 8-dim per-timestep features (notes active at each word's time slot). | Tests whether moment-by-moment melody helps. |

Run all three variants (`none` / `simple` / `per_word`) to satisfy the assignment's "two melody-conditioned variants + baseline" requirement.

### Sampling strategies (generation-time only)

| `--sampling_strategy` | What it does | Relevant flags |
|---|---|---|
| `proportional` | Sample from softmax probabilities directly. | (none) |
| `temperature` | Divide logits by T before softmax (T<1 sharpens, T>1 flattens). | `--sampling_temperature` |
| `top_k` | Keep only the top K logits, sample from those. | `--sampling_top_k`, optional `--sampling_temperature` |
| `nucleus` (top-p) | Smallest set whose cumulative prob exceeds p. | `--sampling_top_p`, optional `--sampling_temperature` |

Strategies do **not** require retraining — they're applied at generation time only.

### Melody-influence probe

`--melody_probe true` runs each generation twice per test song:
1. With real MIDI.
2. With shuffled (corrupted) MIDI.

Reports:
- **Jaccard similarity** of unique tokens (lower = melody matters more).
- **Sequence overlap** (token-by-token match rate).
- **Length difference** (absolute word count).

Output: `final/melody_probe.json`.

### Decoding strategy comparison

`--run_decoding_comparison true` generates the same prompt with **proportional / temperature=0.7 / nucleus p=0.9** on the first 2 test songs, side-by-side. Required by Assignment 3 sec. 13 (diversity vs coherence analysis).

Output: `final/decoding_comparison.txt` - one section per song, three subsections per strategy.

### Running all three required experiments (Assignment 3 sec. 9)

The assignment requires a baseline + two melody-conditioned variants. Run them with distinct `--run_name` values so the output folders don't collide:

```bash
# 1. Lyrics-only baseline (no MIDI)
python main.py --run_name ex1_baseline --task_type language_modeling \
  --data_type lyrics --architecture lstm --midi_variant none \
  --local_dataset_path ./Data/lyrics_train_set.csv \
  --word2vec_path ./GoogleNews-vectors-negative300.bin

# 2. Simple MIDI: global features
python main.py --run_name ex2_simple --task_type language_modeling \
  --data_type lyrics --architecture lstm --midi_variant simple \
  --local_dataset_path ./Data/lyrics_train_set.csv \
  --midi_dir ./Data/midi_files \
  --word2vec_path ./GoogleNews-vectors-negative300.bin \
  --melody_probe true --run_decoding_comparison true

# 3. Per-word MIDI: time-aligned features
python main.py --run_name ex3_per_word --task_type language_modeling \
  --data_type lyrics --architecture lstm --midi_variant per_word \
  --local_dataset_path ./Data/lyrics_train_set.csv \
  --midi_dir ./Data/midi_files \
  --word2vec_path ./GoogleNews-vectors-negative300.bin \
  --melody_probe true --run_decoding_comparison true
```

These can be launched in parallel.

### Outputs (LM-specific)

```
experiments/<RUN_NAME>_*/final/
├── model_checkpoint.pt      ← weights + metadata for re-generation
├── generated_lyrics.txt     ← all test-song outputs
├── melody_probe.json        ← (if --melody_probe true)
├── decoding_comparison.txt  ← (if --run_decoding_comparison true)
├── model.py                 ← standalone code
└── ...
```

### Vocabulary & embeddings

- Vocabulary is built from the training lyrics, with a configurable minimum frequency (`--min_word_count`).
- The line-separator token (default `&`, configurable via `--line_separator_token`) is treated as a regular vocabulary token, so the model learns when to break lines.
- Word2Vec embeddings are loaded once and aligned to the vocab; out-of-vocab tokens get small random vectors.
- By default embeddings are frozen; set `freeze_embeddings: false` in `lstm.yaml` to fine-tune them.

---

## 🧭 Where to Go Next

- For project overview: [`README.md`](README.md)
- For the FTTS algorithm: [`search_strategies/README.md`](search_strategies/README.md)
- For Loss/Accuracy concepts and the Final Phase: [`core/README.md`](core/README.md)
- For data handling: [`data_loaders/README.md`](data_loaders/README.md)
- For extension ideas: [`FUTURE_WORK.md`](FUTURE_WORK.md)
