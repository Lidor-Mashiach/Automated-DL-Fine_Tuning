# 🏗️ Models

Model builders for each supported architecture. Each `build_*(hp, data_info)` function takes a flat hyperparameters dict and dataset info, and returns a ready-to-train `nn.Module`.

---

## 📂 Files

| File | Architecture |
|---|---|
| `mlp.py` | Multi-Layer Perceptron - dense fully-connected network |
| `cnn.py` | Convolutional Neural Network - VGG-style for images |
| `rnn.py` | Vanilla RNN - simple recurrent network |
| `lstm.py` | LSTM - recurrent network with gated memory |
| `transformer.py` | Transformer encoder with positional encoding |
| `__init__.py` | `build_model()` dispatcher - picks the right builder by name |

---

## 🗂️ When to Use Each

| Architecture | Best for | Dataset size |
|---|---|---|
| MLP | Tabular data, structured features | Any |
| CNN | Images, spatial grids | 1K+ images |
| RNN | Short sequences (<50 tokens), prototyping | Small-medium |
| LSTM | Long sequences, basic NLP | Medium-large |
| Transformer | NLP, long-range dependencies | Large |

---

## 📐 Layer Shapes (MLP)

`layer_shape` in `configs/architectures/mlp.yaml` controls how widths evolve. Five patterns are supported, all derived from `hidden_size` and `num_hidden_layers`:

| Shape | Example (hidden_size=128, num_layers=4) | Use case |
|---|---|---|
| `uniform` | 128 - 128 - 128 - 128 | Default; equal capacity per layer |
| `funnel` | 128 - 64 - 32 - 16 | Gradual compression / dimensionality reduction |
| `pyramid` | 16 - 32 - 64 - 128 | Gradual expansion; useful from low-dim inputs |
| `hourglass` | 64 - 128 - 128 - 64 | Rich middle representation, then compress |
| `bottleneck` | 128 - 64 - 64 - 128 | Autoencoder-like; forced compression at the middle |

Implemented in `compute_layer_widths()` in `mlp.py`.

---

## 🧩 Shared Pattern

All builders follow the same flow:

```python
def build_xxx(hp: dict, data_info: dict) -> nn.Module:
    # 1. Extract hyperparameters with safe defaults
    # 2. Compute derived values (layer widths, etc.)
    # 3. Build the nn.Module
    return module
```

The Analyzer can change any of these via the config; the model builder reads them and constructs the network accordingly.

---

## ➕ Adding a New Architecture

To add (e.g.) a GRU architecture:

1. Create `models/gru.py` with a `build_gru(hp, data_info)` function.
2. Add `"gru"` to the registry in `models/__init__.py`.
3. Create `configs/architectures/gru.yaml` with the same 5 sections.
4. Add `"gru"` to `_VALID_ARCH` in `core/run_config.py`.
5. Add it to `main.py`'s `ARCHITECTURE` docstring.
6. Add a `_build_gru_body()` to `core/code_generator.py` and register it in `_ARCHITECTURE_BUILDERS`.

Optionally, extend `core/analyzer.py` with GRU-specific actions (e.g. `toggle_bidirectional` is already supported across recurrent architectures).

---

## 🔗 Related Documents

- [`SETUP_GUIDE.md`](../SETUP_GUIDE.md#-sizing-your-network-how-deep-how-wide) — sizing rules
- [`configs/architectures/README.md`](../configs/architectures/README.md) — parameter formats
- [`core/README.md`](../core/README.md) — Loss vs Accuracy explanation
