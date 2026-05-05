"""
Code generator for the final standalone model.py
------------------------------------------------
Writes a self-contained Python file that:
  - Defines `build_model(input_dim, output_dim)` for the chosen architecture.
  - Loads the dataset the way the tuning run did.
  - Trains on Train+Val combined.
  - Evaluates on Test_set.
  - Prints the metrics.

The generated file has no dependency on the AutoTune-NN package itself - it is
intended to be portable: the user can copy it to another project and run it
anywhere Python + the listed pip packages are available.
"""

from pathlib import Path


_HEADER_TEMPLATE = '''"""
================================================================================
 Standalone model produced by AutoTune-NN
================================================================================

Architecture     : {architecture}
Total trials     : {total_trials}
Quality Score    : {quality:.4f}  (composite score in [0, 1])

Val accuracy:
    - Smoothed (avg of last {smoothing_window} epochs) : {val_smoothed:.4f}
    - Raw peak (single best epoch)    : {val_raw:.4f}

Test accuracy (final eval on held-out Test_set): {test_metric:.4f}

Final loss type: {loss_name}
Final test loss: {test_loss:.4f}
    Lower loss = the model is more confident in its correct predictions.
    See core/README.md (in the AutoTune-NN repo) for the full Loss vs
    Accuracy explanation.


Quality Score combines four components:
    1. best_metric (50%)        - smoothed peak val accuracy
                                  Rewards high performance on Val_set.
    2. stability (20%)          - low variance near the peak
                                  Rewards models that hold their performance
                                  rather than spiking once.
    3. convergence_speed (15%)  - reached 90% of peak at a healthy pace
                                  Penalizes both too-fast (local minimum)
                                  and too-slow (insufficient capacity).
    4. generalization_gap (15%) - small train-val gap
                                  Rewards models that don't memorize.

For the exact formulas, see core/quality_scorer.py in the AutoTune-NN repo.
For the conceptual explanation, see search_strategies/README.md.
================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Hyperparameters that produced the best result
HYPERPARAMETERS = {hp_repr}

# Where the dataset lives
DATASET_PATH = "{dataset_path}"
TASK_TYPE    = "{task_type}"

# Random seed for reproducibility
RANDOM_SEED = {seed}
'''


def generate_model_py(path: Path, cfg, hp: dict, data_info: dict,
                      total_trials: int, best_quality: float,
                      best_metric_raw: float, best_metric_smoothed: float,
                      test_metric: float, test_loss: float,
                      smoothing_window: int) -> None:
    """Generate the standalone model.py at `path`."""
    arch = cfg.architecture
    task_type = cfg.task_type
    seed = cfg.random_seed if cfg.random_seed is not None else 42
    loss_name = _resolve_loss_name(hp, task_type, data_info)

    if cfg.dataset_mode == "local":
        dataset_path = cfg.local_dataset_path
    else:
        dataset_path = f"<imported: {cfg.imported_dataset_name}>"

    header = _HEADER_TEMPLATE.format(
        architecture=arch.upper(),
        total_trials=total_trials,
        quality=best_quality,
        smoothing_window=smoothing_window,
        val_smoothed=best_metric_smoothed,
        val_raw=best_metric_raw,
        test_metric=test_metric,
        test_loss=test_loss,
        loss_name=loss_name,
        hp_repr=_format_hp(hp),
        dataset_path=dataset_path,
        task_type=task_type,
        seed=seed,
    )

    body = _ARCHITECTURE_BUILDERS[arch](hp, data_info, cfg)

    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n\n")
        f.write(body)


# --------------------------------------------------------------- helpers

def _format_hp(hp: dict) -> str:
    """Pretty-print HP dict for inclusion in generated code."""
    keep_keys = sorted(k for k in hp if "__" not in k)
    lines = ["{"]
    for k in keep_keys:
        v = hp[k]
        if isinstance(v, str):
            lines.append(f"    {k!r}: {v!r},")
        elif isinstance(v, float):
            if abs(v) < 1e-3 or abs(v) >= 1e4:
                lines.append(f"    {k!r}: {v:.3e},")
            else:
                lines.append(f"    {k!r}: {v},")
        else:
            lines.append(f"    {k!r}: {v!r},")
    lines.append("}")
    return "\n".join(lines)


def _resolve_loss_name(hp: dict, task_type: str, data_info: dict) -> str:
    if task_type == "regression":
        return "MSE"
    choice = hp.get("loss_function", "auto")
    ratio = float(data_info.get("imbalance_ratio", 1.0))
    if choice == "auto":
        if ratio < 3.0:
            return "Cross-Entropy"
        elif ratio < 10.0:
            return "Focal Loss (gamma=1.5)"
        else:
            return "Focal Loss (gamma=2.5)"
    if choice == "focal":
        return f"Focal Loss (gamma={hp.get('focal_gamma', 2.0)})"
    if choice == "cross_entropy":
        return "Cross-Entropy"
    return choice


# =============================================================================
# Architecture-specific builders
# =============================================================================

_DATA_LOADER_BLOCK = '''
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load and split the dataset. Returns (train_loader, val_loader, test_loader, info)."""
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    df = pd.read_csv(DATASET_PATH)
    label_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != label_col]

    X = pd.get_dummies(df[feature_cols], drop_first=False).values.astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y_raw = df[label_col].values

    if TASK_TYPE == "classification":
        classes, y = np.unique(y_raw, return_inverse=True)
        y = y.astype(np.int64)
        output_dim = int(len(classes))
    else:
        y = y_raw.astype(np.float32).reshape(-1, 1)
        output_dim = 1

    n = len(X)
    perm = np.random.permutation(n)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_idx, val_idx, test_idx = perm[:n_train], perm[n_train:n_train+n_val], perm[n_train+n_val:]

    def make_loader(idxs, shuffle):
        ds = TensorDataset(torch.from_numpy(X[idxs]), torch.from_numpy(y[idxs]))
        return DataLoader(ds, batch_size=HYPERPARAMETERS.get("batch_size", 64),
                           shuffle=shuffle)

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)
    test_loader = make_loader(test_idx, False)
    return train_loader, val_loader, test_loader, {"input_dim": X.shape[1], "output_dim": output_dim}
'''


_TRAIN_EVAL_BLOCK = '''
# ---------------------------------------------------------------------------
# Train and evaluate
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, epochs=None):
    """Train on Train+Val combined (refit on full training data)."""
    epochs = epochs or HYPERPARAMETERS.get("epochs", 30)
    lr = HYPERPARAMETERS.get("learning_rate", 1e-3)
    wd = HYPERPARAMETERS.get("weight_decay", 0.0)
    opt_name = HYPERPARAMETERS.get("optimizer_name", "adam")

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                                     momentum=HYPERPARAMETERS.get("momentum", 0.9))
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if TASK_TYPE == "classification":
        loss_fn = nn.CrossEntropyLoss(label_smoothing=float(HYPERPARAMETERS.get("label_smoothing", 0.0)))
    else:
        loss_fn = nn.MSELoss()

    # Combine train + val for final refit (the reason: AutoTune-NN already used
    # val to select hyperparameters; now we use all available training data).
    from torch.utils.data import ConcatDataset
    combined_ds = ConcatDataset([train_loader.dataset, val_loader.dataset])
    combined_loader = DataLoader(
        combined_ds, batch_size=HYPERPARAMETERS.get("batch_size", 64), shuffle=True
    )

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in combined_loader:
            optimizer.zero_grad()
            out = model(xb)
            if TASK_TYPE == "regression":
                yb_loss = yb.float().view(-1, 1)
            else:
                yb_loss = yb
            loss = loss_fn(out, yb_loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"epoch {epoch+1}/{epochs}: loss={epoch_loss/len(combined_loader):.4f}")
    return model


def evaluate(model, loader, name="set"):
    """Evaluate the model on a loader. Prints loss and metric."""
    model.eval()
    if TASK_TYPE == "classification":
        loss_fn = nn.CrossEntropyLoss(label_smoothing=float(HYPERPARAMETERS.get("label_smoothing", 0.0)))
    else:
        loss_fn = nn.MSELoss()

    total_loss, total_metric, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            if TASK_TYPE == "regression":
                yb_loss = yb.float().view(-1, 1)
            else:
                yb_loss = yb
            loss = loss_fn(out, yb_loss)
            total_loss += loss.item()
            if TASK_TYPE == "classification":
                preds = out.argmax(dim=1)
                total_metric += (preds == yb).float().mean().item()
            else:
                rmse = torch.sqrt(((out.squeeze() - yb.squeeze()) ** 2).mean()).item()
                total_metric -= rmse
            n_batches += 1
    avg_loss = total_loss / max(1, n_batches)
    avg_metric = total_metric / max(1, n_batches)
    metric_label = "accuracy" if TASK_TYPE == "classification" else "-RMSE"
    print(f"{name}: loss={avg_loss:.4f}  {metric_label}={avg_metric:.4f}")
    return avg_loss, avg_metric


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    train_loader, val_loader, test_loader, info = load_data()

    print(f"Building {(__doc__.split('Architecture')[1].split(':')[1].strip().split()[0]).upper()} model...")
    model = build_model(info["input_dim"], info["output_dim"])

    print("Training on Train+Val (refit on full training data)...")
    model = train_model(model, train_loader, val_loader)

    print()
    print("Evaluating on the held-out Test_set:")
    test_loss, test_metric = evaluate(model, test_loader, "Test")

    print()
    print("Done.")
'''


def _build_mlp_body(hp, data_info, cfg):
    return '''
# ---------------------------------------------------------------------------
# Architecture: MLP
# ---------------------------------------------------------------------------

def build_model(input_dim: int, output_dim: int) -> nn.Module:
    """Build the MLP architecture chosen by AutoTune-NN."""
    num_layers = HYPERPARAMETERS["num_hidden_layers"]
    base_width = HYPERPARAMETERS["hidden_size"]
    shape = HYPERPARAMETERS.get("layer_shape", "uniform")
    activation_name = HYPERPARAMETERS.get("activation", "relu")
    use_bn = HYPERPARAMETERS.get("batch_norm", False)
    dropout_p = HYPERPARAMETERS.get("dropout", 0.0)

    widths = _compute_widths(shape, num_layers, base_width)
    activation_cls = {
        "relu": nn.ReLU, "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh,
        "elu": nn.ELU, "selu": nn.SELU, "silu": nn.SiLU,
    }.get(activation_name, nn.ReLU)

    layers = []
    prev_dim = input_dim
    for w in widths:
        layers.append(nn.Linear(prev_dim, w))
        if use_bn:
            layers.append(nn.BatchNorm1d(w))
        layers.append(activation_cls())
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        prev_dim = w
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def _compute_widths(shape: str, num_layers: int, base: int):
    """Compute per-layer widths from the chosen shape pattern."""
    if num_layers <= 0:
        return []
    if shape == "uniform":
        return [base] * num_layers
    if shape == "funnel":
        return [max(base // (2 ** i), 8) for i in range(num_layers)]
    if shape == "pyramid":
        return [max(base // (2 ** (num_layers - 1 - i)), 8) for i in range(num_layers)]
    if shape == "hourglass":
        mid = (num_layers - 1) / 2.0
        return [max(base // (2 ** int(round(abs(i - mid)))), 8)
                for i in range(num_layers)]
    if shape == "bottleneck":
        mid = (num_layers - 1) / 2.0
        return [max(base // (2 ** int(round(mid - abs(i - mid)))), 8)
                for i in range(num_layers)]
    return [base] * num_layers

''' + _DATA_LOADER_BLOCK + _TRAIN_EVAL_BLOCK


def _build_cnn_body(hp, data_info, cfg):
    return '''
# ---------------------------------------------------------------------------
# Architecture: CNN (VGG-style)
# ---------------------------------------------------------------------------

def build_model(input_channels: int, output_dim: int, image_size: int) -> nn.Module:
    """Build the CNN architecture chosen by AutoTune-NN."""
    num_blocks = HYPERPARAMETERS.get("num_conv_blocks", 3)
    base_filters = HYPERPARAMETERS.get("base_filters", 32)
    kernel = HYPERPARAMETERS.get("kernel_size", 3)
    pool = HYPERPARAMETERS.get("pooling", "max")
    num_fc = HYPERPARAMETERS.get("num_fc_layers", 1)
    fc_size = HYPERPARAMETERS.get("fc_size", 256)
    activation_name = HYPERPARAMETERS.get("activation", "relu")
    use_bn = HYPERPARAMETERS.get("batch_norm", True)
    dropout_p = HYPERPARAMETERS.get("dropout", 0.0)

    activation_cls = {
        "relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU,
    }.get(activation_name, nn.ReLU)
    pool_cls = nn.MaxPool2d if pool == "max" else nn.AvgPool2d
    padding = kernel // 2

    conv_layers = []
    prev_c = input_channels
    current_size = image_size
    for i in range(num_blocks):
        filters = base_filters * (2 ** i)
        conv_layers.append(nn.Conv2d(prev_c, filters, kernel_size=kernel, padding=padding))
        if use_bn:
            conv_layers.append(nn.BatchNorm2d(filters))
        conv_layers.append(activation_cls())
        conv_layers.append(pool_cls(kernel_size=2))
        prev_c = filters
        current_size = current_size // 2
        if current_size < 2:
            break

    flatten_dim = prev_c * max(1, current_size) * max(1, current_size)

    fc_layers = [nn.Flatten()]
    prev_dim = flatten_dim
    for i in range(num_fc):
        width = max(32, fc_size // (2 ** i))
        fc_layers.append(nn.Linear(prev_dim, width))
        fc_layers.append(activation_cls())
        if dropout_p > 0:
            fc_layers.append(nn.Dropout(dropout_p))
        prev_dim = width
    fc_layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*conv_layers, *fc_layers)


# ---------------------------------------------------------------------------
# Data loading - minimal CNN demo using torchvision MNIST
# (replace with your own dataset path / loader if needed)
# ---------------------------------------------------------------------------

def load_data():
    """Load and split the dataset using torchvision."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    image_size = HYPERPARAMETERS.get("image_size", 32)
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    cache = "./.data_cache"
    train_full = datasets.MNIST(cache, train=True, download=True, transform=tf)
    test_full = datasets.MNIST(cache, train=False, download=True, transform=tf)

    n_total = len(train_full)
    n_val = int(n_total * 0.25)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(RANDOM_SEED)
    train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=g)

    bs = HYPERPARAMETERS.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_full, batch_size=bs, shuffle=False)
    sample, _ = train_full[0]
    info = {
        "input_channels": sample.shape[0],
        "image_size": sample.shape[1],
        "output_dim": 10,
    }
    return train_loader, val_loader, test_loader, info

''' + _TRAIN_EVAL_BLOCK_CNN


_TRAIN_EVAL_BLOCK_CNN = '''
# ---------------------------------------------------------------------------
# Train and evaluate (CNN)
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, epochs=None):
    """Train on Train+Val combined."""
    from torch.utils.data import ConcatDataset, DataLoader
    epochs = epochs or HYPERPARAMETERS.get("epochs", 30)
    lr = HYPERPARAMETERS.get("learning_rate", 1e-3)
    wd = HYPERPARAMETERS.get("weight_decay", 0.0)
    opt_name = HYPERPARAMETERS.get("optimizer_name", "adam")

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                                     momentum=HYPERPARAMETERS.get("momentum", 0.9))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(HYPERPARAMETERS.get("label_smoothing", 0.0)))
    combined = ConcatDataset([train_loader.dataset, val_loader.dataset])
    combined_loader = DataLoader(combined, batch_size=HYPERPARAMETERS.get("batch_size", 64),
                                  shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in combined_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"epoch {epoch+1}/{epochs}: loss={epoch_loss/len(combined_loader):.4f}")
    return model


def evaluate(model, loader, name="set"):
    """Evaluate the model. Returns (loss, metric)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(HYPERPARAMETERS.get("label_smoothing", 0.0)))
    total_loss, total_metric, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            loss = loss_fn(out, yb)
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            total_metric += (preds == yb).float().mean().item()
            n_batches += 1
    avg_loss = total_loss / max(1, n_batches)
    avg_metric = total_metric / max(1, n_batches)
    print(f"{name}: loss={avg_loss:.4f}  accuracy={avg_metric:.4f}")
    return avg_loss, avg_metric


if __name__ == "__main__":
    print("Loading data...")
    train_loader, val_loader, test_loader, info = load_data()
    print("Building CNN model...")
    model = build_model(info["input_channels"], info["output_dim"], info["image_size"])
    print("Training on Train+Val (refit on full training data)...")
    model = train_model(model, train_loader, val_loader)
    print()
    print("Evaluating on the held-out Test_set:")
    evaluate(model, test_loader, "Test")
    print()
    print("Done.")
'''


def _build_rnn_body(hp, data_info, cfg):
    rnn_class = "nn.LSTM" if cfg.architecture == "lstm" else "nn.RNN"
    return f'''
# ---------------------------------------------------------------------------
# Architecture: {cfg.architecture.upper()}
# ---------------------------------------------------------------------------

class _SequenceClassifier(nn.Module):
    """Wrapper: take last hidden state for classification."""

    def __init__(self, rnn, embedding, hidden_size, output_dim, dropout, bidirectional):
        super().__init__()
        self.embedding = embedding
        self.rnn = rnn
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        direction_mult = 2 if bidirectional else 1
        self.head = nn.Linear(hidden_size * direction_mult, output_dim)

    def forward(self, x):
        if self.embedding is not None:
            x = self.embedding(x)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        output, _ = self.rnn(x)
        last = output[:, -1, :]
        return self.head(self.dropout(last))


def build_model(input_dim_or_vocab: int, output_dim: int) -> nn.Module:
    """Build the {cfg.architecture.upper()} architecture chosen by AutoTune-NN."""
    hidden_size = HYPERPARAMETERS["hidden_size"]
    num_layers = HYPERPARAMETERS["num_layers"]
    bidirectional = HYPERPARAMETERS.get("bidirectional", False)
    dropout_p = HYPERPARAMETERS.get("dropout", 0.0)
    embedding_dim = HYPERPARAMETERS.get("embedding_dim", 128)

    # Treat input_dim_or_vocab as vocab_size if integer > some threshold,
    # but typically the caller knows. Here we assume vocab.
    embedding = nn.Embedding(input_dim_or_vocab, embedding_dim, padding_idx=0)
    input_size = embedding_dim

    rnn_dropout = dropout_p if num_layers > 1 else 0.0
    rnn_kwargs = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=bidirectional,
        dropout=rnn_dropout,
    )
    {"# Vanilla RNN uses tanh nonlinearity by default" if cfg.architecture == "rnn" else ""}
    rnn = {rnn_class}(**rnn_kwargs)
    return _SequenceClassifier(rnn, embedding, hidden_size, output_dim,
                                dropout_p, bidirectional)

''' + _DATA_LOADER_BLOCK + _TRAIN_EVAL_BLOCK


def _build_transformer_body(hp, data_info, cfg):
    return '''
# ---------------------------------------------------------------------------
# Architecture: Transformer Encoder
# ---------------------------------------------------------------------------

import math


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class _TransformerClassifier(nn.Module):
    def __init__(self, input_projection, encoder, pos_encoding, d_model, output_dim, dropout):
        super().__init__()
        self.input_projection = input_projection
        self.pos_encoding = pos_encoding
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(self.dropout(x))


def build_model(input_dim_or_vocab: int, output_dim: int) -> nn.Module:
    d_model = HYPERPARAMETERS["d_model"]
    nhead = HYPERPARAMETERS["nhead"]
    num_layers = HYPERPARAMETERS["num_encoder_layers"]
    dim_ff = HYPERPARAMETERS["dim_feedforward"]
    dropout_p = HYPERPARAMETERS.get("dropout", 0.1)
    activation = HYPERPARAMETERS.get("activation", "relu")

    if d_model % nhead != 0:
        candidates = [h for h in (1, 2, 4, 8, 16) if d_model % h == 0]
        nhead = max(candidates) if candidates else 1

    input_projection = nn.Embedding(input_dim_or_vocab, d_model, padding_idx=0)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
        dropout=dropout_p, activation=activation, batch_first=True,
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    pos = _PositionalEncoding(d_model)
    return _TransformerClassifier(input_projection, encoder, pos, d_model,
                                   output_dim, dropout_p)

''' + _DATA_LOADER_BLOCK + _TRAIN_EVAL_BLOCK


_ARCHITECTURE_BUILDERS = {
    "mlp": _build_mlp_body,
    "cnn": _build_cnn_body,
    "rnn": _build_rnn_body,
    "lstm": _build_rnn_body,
    "transformer": _build_transformer_body,
}
