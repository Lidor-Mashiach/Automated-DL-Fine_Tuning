# 🏗️ Models

Model builders for each supported architecture. Each `build_*(hp, data_info)` function takes a flat hyperparameters dict and dataset info, and returns a ready-to-train `nn.Module`.

---

## 📂 Files

| File | Architecture |
|---|---|
| `mlp.py` | Multi-Layer Perceptron — dense fully-connected network |
| `cnn.py` | Convolutional Neural Network — VGG-style for images |
| `rnn.py` | Vanilla RNN — simple recurrent network |
| `lstm.py` | LSTM — recurrent network with gated memory |
| `transformer.py` | Transformer encoder with positional encoding |
| `__init__.py` | `build_model()` dispatcher — picks the right builder by name |

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

## 🧩 Shared Structure

All builders follow the same pattern:

```python
def build_mlp(hp: dict, data_info: dict) -> nn.Module:
    # 1. Extract hyperparameters with safe defaults
    num_layers = int(hp.get("num_hidden_layers", 2))
    hidden_size = int(hp.get("hidden_size", 128))
    # ...

    # 2. Compute derived values (e.g., layer widths)
    widths = _compute_widths(layer_shape, num_layers, hidden_size)

    # 3. Build the nn.Module
    layers = []
    for w in widths:
        layers.append(nn.Linear(prev, w))
        # ... activation, dropout, batch_norm, ...

    return nn.Sequential(*layers)
```

---

## 📐 Layer Shapes (MLP)

`layer_shape` in `configs/architectures/mlp.yaml` controls how widths evolve:

| Shape | Example (hidden_size=128, num_layers=3) |
|---|---|
| `uniform` | `128 → 128 → 128` |
| `funnel` | `128 → 64 → 32` |
| `pyramid` | `32 → 64 → 128` |
| `hourglass` | `64 → 128 → 64` |

---

## ➕ Adding a New Architecture

To add (e.g.) a GRU architecture:

1. Create `models/gru.py` with a `build_gru(hp, data_info)` function.
2. Add `"gru"` to the registry in `models/__init__.py`.
3. Create `configs/architectures/gru.yaml` with the same 5 sections.
4. Add `"gru"` to `_VALID_ARCH` in `core/run_config.py`.
5. Add it to `main.py`'s `ARCHITECTURE` docstring.

Optionally, extend `core/analyzer.py` with GRU-specific actions (e.g. `toggle_bidirectional` is already supported).
