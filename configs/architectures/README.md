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
- **Width:** 32-1024 neurons per layer
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

---

## 🎵 LSTM Language Modeling Parameters

When `task_type="language_modeling"`, the LSTM YAML has 4 additional parameters specific to dual-input (Word2Vec + MIDI) training:

| Parameter | Type | Default | When to tune |
|---|---|---|---|
| `fusion_method` | choices | `"concatenate"` | If the model isn't learning well with `concatenate`, try `project` (each modality passes through a Linear+ReLU first). |
| `fusion_proj_dim` | choices | `128` | Only used when `fusion_method=project`. Width of the projection layers. |
| `freeze_embeddings` | bool | `true` | Default freezes Word2Vec. If the model plateaus (slow verdict), the Analyzer will suggest `false` to add learnable capacity. |
| `tie_weights` | bool | `false` | Tie output projection to embedding weights. Requires `hidden_size == embedding_dim`. Disabled by default. |

For classification mode, these parameters exist but are silently ignored - the standard `_SequenceClassifier` is used instead of `_LSTMLanguageModel`.

---

## 🆕 New LM Parameters (Assignment 3 enhancements)

Beyond the original LSTM-LM parameters, the YAML now exposes these tunable knobs:

| Parameter | Type | Default | Choices/Range | Purpose |
|---|---|---|---|---|
| `sequence_length` | discrete | `32` | `[32, 64, 128, 256]` | Context window during training. Longer = more context, but harder gradient flow. Tuned by FTTS via `increase_sequence_length` / `decrease_sequence_length`. |
| `teacher_forcing_ratio` | discrete | `1.0` | `[0.3, 0.5, 0.7, 1.0]` | Probability per training step of feeding the ground-truth previous word vs noise/own output. Lower = robust generation; higher = fast convergence. Tuned via `increase/decrease_teacher_forcing`. |
| `max_words_per_line` | discrete | `12` | `[8, 10, 12, 16, 100]` | Hard constraint at generation: force `<line_sep>` if line reaches this length. Set 100 to effectively disable. |
| `min_words_per_line` | discrete | `2` | `[0, 2, 3]` | Hard constraint at generation: suppress `<line_sep>` until line is at least this long. |
| `bidirectional` | bool | `true` (cls) / `false` (LM auto) | `[true, false]` | LM cannot use bidirectional context for autoregressive generation. The Analyzer proposes `disable_bidirectional` if it sees bidirectional + LM. |

### Analyzer actions triggered for LM

When `task_type="language_modeling"`, the Analyzer can propose these LM-specific moves (in addition to the universal ones):
- `slow` verdict → `unfreeze_embeddings`, `change_fusion_method`, `increase_sequence_length`, `decrease_teacher_forcing`, `increase_gradient_accumulation`
- `overfit` verdict → `decrease_sequence_length`, `increase_teacher_forcing`, `disable_bidirectional`

---

## 🔧 LSTM YAML defaults updated (post-empirical analysis)

After the first lyrics-generation run, the LSTM YAML was tuned with two complementary moves:

**1. Initial values moved to the middle of the explored range.**
The first run started with `hidden_size=128` and `sequence_length=32` - both at the **low end** of what's reasonable for LM. This biased the early FTTS exploration toward "small model" trials. Moving the starts to the middle lets the analyzer explore upward AND downward.

**2. choices widened, not narrowed.**
Rather than restricting the search to "good" values, the range was widened in both directions. The analyzer's adaptive-step logic decides which direction to go based on the verdict.

| Parameter | Old initial | New initial | Old choices | New choices |
|---|---|---|---|---|
| `hidden_size` | 128 | **256** | `[64, 128, 256, 512]` | **`[32, 64, 128, 256, 512, 1024]`** |
| `sequence_length` | 32 | **128** | `[32, 64, 128, 256]` | **`[32, 64, 128, 256, 512, 1024]`** |

### Why widen instead of narrow?
- `hidden_size=32/64` is sometimes the right answer for tiny corpora or fast experimentation. Keep it available.
- `hidden_size=1024` is useful for runs with abundant GPU memory and large vocabularies.
- `sequence_length=512/1024` lets the model learn long-range structure (multi-verse coherence) when memory allows.
- The analyzer's `increase_X` / `decrease_X` actions navigate the choices list step by step, so a wider list just means more exploration room, not slower convergence.

These changes affect only the **initial** trial (T0001) and the search space. FTTS can still reach any value within the choices list during exploration.

---

## 🆕 Analyzer Action Coverage Expansion

Empirical analysis revealed 9 ACTION_TYPES that had FTTS handlers but were **never emitted** by any verdict — they existed in code but were dead in practice. After expansion, every action is reachable through the analyzer:

| Action | Verdict | Priority | When triggered |
|---|---|---|---|
| `decrease_dropout` | `slow` | 0.45 | Model not learning - try less regularization |
| `increase_batch_size` | `slow` | 0.40 | Large batches can stabilize tricky losses |
| `enable_batch_norm` | `failed_to_learn` | 0.55 | Stabilize gradients when training fails to start |
| `change_normalization` | `failed_to_learn` | 0.50 | Input normalization strategy may be wrong |
| `reduce_depth` | `overfit` | 0.35 | Reduce model capacity |
| `toggle_bidirectional` | `overfit` | 0.30 | Test whether bi-LSTM is hurting generalization |
| `adjust_attention_dropout` | `slow` | 0.30 | Transformer-specific dropout tuning |
| `adjust_adam_beta1` | `slow` | 0.15 | Advanced Adam momentum tuning (off-by-default) |
| `adjust_adam_beta2` | `slow` | 0.15 | Advanced Adam squared-gradient tuning (off-by-default) |

Note: the `adjust_adam_beta1/2` actions are emitted only when those parameters are explicitly enabled in YAML (default is `enabled: false`).

---

## 📏 Large-Model Support (up to 1024-wide structures)

All architectures now support structure sizes up to 1024, not just 512. This lets FTTS explore high-capacity models when the data and GPU memory justify it.

| Architecture | Parameter | Old max | New max |
|---|---|---|---|
| MLP | `hidden_size` | 512 | **1024** |
| RNN | `hidden_size` | 256 | **1024** |
| LSTM | `hidden_size` | 1024 | 1024 (already) |
| LSTM | `sequence_length` | 512 | **1024** |
| LSTM | `fusion_proj_dim` | 256 | **512** |
| CNN | `fc_size` | 512 | **1024** |
| CNN | `base_filters` | 64 | **128** |
| Transformer | `d_model` | 512 | **1024** |

### Notes
- **CNN `base_filters`** is capped at 128 (not 1024) because filters multiply across conv blocks: with `base_filters=128` and 5 conv blocks, the deepest block already reaches `128 * 2^4 = 2048` channels. Going higher would be impractical.
- **Transformer `d_model=1024`** is divisible by all `nhead` choices (2, 4, 8), so no compatibility issue. The model also has a soft-correction in `models/transformer.py` that picks a compatible `nhead` if an invalid combination is ever proposed.
- **No code changes needed in the models** — every architecture reads its size from `hp` dynamically (`hp["hidden_size"]`, `hp["d_model"]`, etc.) with no hard-coded ceiling. The 1024 support is purely a YAML `choices` widening.
- The Analyzer is size-agnostic: it proposes `add_width` / `reduce_width`, and FTTS navigates the `choices` list step by step. A wider list just means more exploration room.
