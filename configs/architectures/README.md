# 🏗️ Architecture Configurations

One YAML file per neural network architecture. Each file is divided into five thematic sections:

| Section | Contains |
|---|---|
| **parameters** | Scalar parameters that define the network — depth, width, activation function. |
| **architectures** | Structural variants — layer shape, bidirectional toggle, pooling method. |
| **methods** | Regularization and training-stability methods — dropout, batch_norm, weight_decay, gradient_clipping, early_stopping, label_smoothing, etc. |
| **optimization** | Optimizer name, learning rate, momentum, learning-rate scheduler. |
| **training** | Batch size, epochs, validation split, and data-specific settings (image_size, sequence_length). |

See the [parent README](../README.md) for the parameter format specification.

---

## 📂 Supported Architectures

### [`mlp.yaml`](mlp.yaml) — Multi-Layer Perceptron

Dense fully-connected network. Best for tabular data.

- **Depth:** 1-5 hidden layers
- **Width:** 32-512 neurons per layer
- **Layer shapes:** uniform / funnel / pyramid / hourglass

### [`cnn.yaml`](cnn.yaml) — Convolutional Neural Network

VGG-style CNN. Best for images.

- Conv blocks (Conv → Activation → Pool), 2-5 blocks
- Filters: start at `base_filters`, double each block
- CNN-specific methods: `data_augmentation`, `mixup`

### [`rnn.yaml`](rnn.yaml) — Vanilla RNN

Simple recurrent network. Best for short sequences and prototyping.

- Depth: 1-3 layers (limited by vanishing gradient)
- Supports `bidirectional`

### [`lstm.yaml`](lstm.yaml) — Long Short-Term Memory

Recurrent network with gated cells. Best for long sequences and NLP.

- Depth: 1-4 layers (handles depth better than vanilla RNN)
- Supports `bidirectional`

### [`transformer.yaml`](transformer.yaml) — Transformer Encoder

Attention-based network. Best for NLP and complex sequences.

- Transformer-specific methods: `attention_dropout`, `lr_warmup`, `positional_encoding`
- Requires `d_model` divisible by `nhead` — system auto-fixes mismatches

---

## 🎛️ Common Parameters (across multiple architectures)

Full documentation for each parameter is in the YAML file itself as docstrings. Summary:

### Core training parameters

| Parameter | Purpose | Raising the value... |
|---|---|---|
| `learning_rate` | Step size for weight updates | Makes training faster, but risks divergence |
| `weight_decay` | L2 regularization strength | Reduces overfitting, may underfit |
| `dropout` | Fraction of neurons to drop | Stronger regularization |
| `batch_size` | Samples per gradient update | Faster, less noise, less ability to escape local minima |
| `num_layers` / `num_hidden_layers` | Network depth | More capacity, slower training, risk of overfit |
| `hidden_size` / `d_model` | Network width | More capacity, slower training |
| `gradient_clipping` | Max gradient norm | Stabilizes training, prevents exploding gradients |

### Methods (on/off + value)

| Method | Purpose |
|---|---|
| `batch_norm` | Normalizes activations for stable training |
| `early_stopping` | Stops a trial when its val_loss plateaus (trial-level, not run-level) |
| `label_smoothing` | Softens classification targets to reduce overconfidence |
| `data_augmentation` | Expands training data with transformations (CNN only) |
| `mixup` | Blends training samples (CNN only) |

### Optimizers

The optimizer is configured via the `optimizer_name` parameter (under `optimization:` in YAML).

| Optimizer | When to prefer |
|---|---|
| `adam` | Default good choice, works for most tasks |
| `adamw` | Deep networks, transformers — correct weight decay handling |
| `sgd` | Final convergence, often beats Adam on image classification |
| `rmsprop` | RNNs — often more stable than Adam |

> ℹ️ Adam's `beta1` (0.9) and `beta2` (0.999) are used internally with their standard values. The `adam_beta1` and `adam_beta2` parameters in the YAML are **disabled by default** — enable only if you need to tune them for very large training runs.

### Learning-rate schedulers

| Scheduler | Behavior |
|---|---|
| `none` | Constant LR throughout |
| `cosine` | Smooth decay to near-zero by end of training |
| `step` | Piecewise reductions every N epochs |
| `reduce_on_plateau` | Halves LR when val_loss stops improving |

---

## 🛠️ Parameter Configuration Format

Every parameter follows this shape:

```yaml
# Active parameter, known starting value
dropout:
  enabled: true
  initial_value: 0.2
  range: [0.0, 0.5]

# Active parameter, discrete choices
activation:
  enabled: true
  initial_value: "relu"
  choices: ["relu", "gelu", "leaky_relu", "tanh"]

# Active parameter, system picks starting value
weight_decay:
  enabled: true
  initial_value: null            # null = middle of range
  range: [1.0e-6, 1.0e-2]
  log: true                      # log-scale sampling

# Disabled parameter - never used in any trial
l1_regularization:
  enabled: false
```

**Three things you can do with any parameter:**

| Goal | How |
|---|---|
| Use parameter, specific starting value | `enabled: true` + set `initial_value` |
| Use parameter, system picks default | `enabled: true` + `initial_value: null` |
| Disable parameter completely | `enabled: false` |

> 💡 **About `log: true`**: use this for parameters spanning orders of magnitude (like `learning_rate` from 1e-5 to 1e-1, or `weight_decay` from 1e-6 to 1e-2). It samples uniformly on the log scale, which matches how these parameters actually behave.

---

## 🎯 MLP Layer Shape — Automatic Pattern Selection

`layer_shape` controls how widths evolve across MLP layers. With `hidden_size=128, num_hidden_layers=4`:

| Shape | Layer widths | When it fits |
|---|---|---|
| `uniform` | 128 → 128 → 128 → 128 | Default; equal capacity per layer |
| `funnel` | 128 → 64 → 32 → 16 | High-dim input gradually compressed |
| `pyramid` | 16 → 32 → 64 → 128 | Low-dim input gradually expanded |
| `hourglass` | 64 → 128 → 128 → 64 | Rich middle representation |
| `bottleneck` | 128 → 64 → 64 → 128 | Forces compression in the middle (autoencoder-like) |

**The Analyzer picks the right pattern automatically** based on the verdict:

| Verdict | Suggested patterns (with priority) | Reasoning |
|---|---|---|
| `overfit` | `bottleneck` (0.55), `funnel` (0.40) | Compression acts as implicit regularizer |
| `failed_to_learn` | `hourglass` (0.55), `uniform` (0.40) | More capacity in the middle helps learning |
| `slow` | `funnel` (0.45) | Fewer params in deep layers may speed up convergence |
| `healthy` | `bottleneck` (0.30), `pyramid` (0.25) | Low-priority exploration |

> ℹ️ The user doesn't have to choose manually. The Analyzer skips the current shape (no-op) and FTTS deduplicates configurations already tried via different paths (DAG-dedup).

To restrict the search to one specific pattern, narrow `choices` in `mlp.yaml`:
```yaml
layer_shape:
  enabled: true
  initial_value: "bottleneck"
  choices: ["bottleneck"]   # only this pattern, no exploration
```

---

## 🎚️ Forcing a Parameter to a Single Value

To make a parameter **never change** during the search, narrow `choices` (for discrete) or set `range` to a single point — though the cleanest way is:

```yaml
# Force activation = "gelu" forever
activation:
  enabled: true
  initial_value: "gelu"
  choices: ["gelu"]              # only this option = analyzer can't change it
```

For continuous parameters, narrow the range:

```yaml
# Force learning_rate = 1e-3 forever
learning_rate:
  enabled: true
  initial_value: 1.0e-3
  range: [1.0e-3, 1.0e-3]        # min == max, no movement possible
  log: false                     # safer with a single point
```

---

## 🆕 New Parameters (latest revision)

These were added recently and apply to most architectures:

| Parameter | Where | Purpose |
|---|---|---|
| `loss_function` | `training` | `auto`/`cross_entropy`/`focal`/`mse`. See [`core/README.md`](../../core/README.md) for selection logic. |
| `focal_gamma` | `training` | Focusing strength for Focal Loss. Used only when `loss_function: focal`. |
| `normalization` (MLP only) | `training` | `none`/`standardize`/`min_max`/`max_abs`. See [`data_loaders/README.md`](../../data_loaders/README.md). |
| `num_workers` | `training` | DataLoader subprocess workers. **System setting - not tuned by Analyzer.** |
| `bottleneck` (in `layer_shape` choices) | `architectures` | Autoencoder-like shape: shrinks toward middle, then grows back. |


---

## 🎨 Data Augmentation Support by Architecture

Different architectures get different augmentation tools because what regularizes one data type doesn't apply to another:

| Architecture | Data type | Augmentation parameter | Choices |
|---|---|---|---|
| MLP | Tabular | (none) | Tabular augmentation is rarely useful; standardization handles it. |
| CNN | Image | `data_augmentation` + `mixup` | `none`/`light`/`medium` (flips, crops, jitter); mixup blends pairs of samples |
| RNN/LSTM/Transformer | Text/Sequence | `text_augmentation` + `text_augmentation_prob` | `none`/`token_dropout`/`word_shuffle` |

### What each does

- **CNN `data_augmentation`**:
  - `light` = horizontal flip + random crop with padding=2
  - `medium` = adds rotation + larger crop padding
- **CNN `mixup`**: takes two random samples, blends both inputs and labels with weight α drawn from Beta(α, α). Strong regularizer.
- **Text `token_dropout`**: replaces tokens with the pad token at probability `text_augmentation_prob`. Acts like word-level dropout.
- **Text `word_shuffle`**: shuffles tokens within small windows (size 3) at probability `text_augmentation_prob`. Preserves locality but breaks exact ordering.

The Analyzer can propose more augmentation when overfitting is detected (`change_text_augmentation`, `increase_text_augmentation`, `increase_augmentation` for CNN, `add_mixup`).


---

## 🆕 Latest Additions

### Mixed Precision & Gradient Accumulation (all architectures)

```yaml
gradient_accumulation_steps:
  enabled: true
  initial_value: 1
  range: [1, 8]

mixed_precision:
  enabled: true
  initial_value: false
  choices: [true, false]
```

- `gradient_accumulation_steps`: simulate larger batch by accumulating gradients before stepping. Effective batch = `batch_size × gradient_accumulation_steps`.
- `mixed_precision`: enable fp16 training on GPU; ~2x speedup, halved VRAM.

### Adam Optimizer Betas (all architectures, disabled by default)

```yaml
adam_beta1:
  enabled: false              # disabled by default
  initial_value: 0.9
  range: [0.85, 0.95]

adam_beta2:
  enabled: false              # disabled by default
  initial_value: 0.999
  range: [0.9, 0.9999]
```

When disabled, PyTorch's defaults (β1=0.9, β2=0.999) are used. Enable only for advanced tuning of very large training runs.

### Stochastic Depth (Transformer only)

```yaml
stochastic_depth:
  enabled: true
  initial_value: 0.0
  range: [0.0, 0.3]
```

Randomly drops entire encoder layers during training. **Only effective when `num_encoder_layers >= 4`** — at lower depth, the regularization effect is negligible. Recommended for transformer overfit on deep configurations.

### Image Augmentation (CNN only)

```yaml
cutout:
  enabled: true
  initial_value: 0.0
  range: [0.0, 0.4]

cutmix:
  enabled: true
  initial_value: 0.0
  range: [0.0, 1.0]
```

- `cutout`: masks a random square (size = `cutout × image_size`) with zeros.
- `cutmix`: cuts a patch from one image, pastes into another, mixes labels. The value is the Beta distribution α.

### Text Augmentation (RNN/LSTM/Transformer)

```yaml
text_augmentation:
  enabled: true
  initial_value: "none"
  choices: ["none", "token_dropout", "word_shuffle", "ngram_shuffle"]

text_augmentation_prob:
  enabled: true
  initial_value: 0.1
  range: [0.0, 0.3]
```

- `token_dropout`: replaces tokens with pad token at rate `prob`.
- `word_shuffle`: shuffles within size-3 windows at rate `prob`.
- `ngram_shuffle`: shuffles 2-3-grams across the whole sequence at rate `prob`.

### Embedding Dropout (RNN/LSTM/Transformer)

```yaml
embedding_dropout:
  enabled: true
  initial_value: 0.0
  range: [0.0, 0.4]
```

Dropout applied to the embedding layer's output before feeding it into the recurrent/transformer layers. Different from `dropout` — strong regularizer specific to NLP.
