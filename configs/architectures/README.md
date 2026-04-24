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

| Optimizer | When to prefer |
|---|---|
| `adam` | Default good choice, works for most tasks |
| `adamw` | Deep networks, transformers — correct weight decay handling |
| `sgd` | Final convergence, often beats Adam on image classification |
| `rmsprop` | RNNs — often more stable than Adam |

### Learning-rate schedulers

| Scheduler | Behavior |
|---|---|
| `none` | Constant LR throughout |
| `cosine` | Smooth decay to near-zero by end of training |
| `step` | Piecewise reductions every N epochs |
| `reduce_on_plateau` | Halves LR when val_loss stops improving |
