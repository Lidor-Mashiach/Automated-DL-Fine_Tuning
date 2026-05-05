"""
MLP builder
-----------
Builds a fully-connected network with configurable depth, width, layer shape,
activation, dropout, and batch normalization.
"""

import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "silu": nn.SiLU,
}


def compute_layer_widths(shape: str, num_layers: int, base_width: int) -> list[int]:
    """
    Compute the per-layer widths for the chosen pattern.

    Patterns (using base_width=128, num_layers=4 as example):
      uniform    : 128, 128, 128, 128
      funnel     : 128, 64, 32, 16    (each layer halves)
      pyramid    : 16,  32, 64, 128   (each layer doubles)
      hourglass  : 64, 128, 128, 64   (grows toward middle, then shrinks)
      bottleneck : 128, 64, 64, 128   (shrinks toward middle, then grows)
    """
    if num_layers <= 0:
        return []

    if shape == "uniform":
        return [base_width] * num_layers

    if shape == "funnel":
        return [max(8, base_width // (2 ** i)) for i in range(num_layers)]

    if shape == "pyramid":
        return [max(8, base_width // (2 ** (num_layers - 1 - i)))
                for i in range(num_layers)]

    if shape == "hourglass":
        mid = (num_layers - 1) / 2.0
        return [max(8, base_width // (2 ** int(round(abs(i - mid)))))
                for i in range(num_layers)]

    if shape == "bottleneck":
        mid = (num_layers - 1) / 2.0
        return [max(8, base_width // (2 ** int(round(mid - abs(i - mid)))))
                for i in range(num_layers)]

    return [base_width] * num_layers


def build_mlp(hp: dict, data_info: dict) -> nn.Module:
    """
    Build an MLP from hyperparameters.

    Expected hp keys:
        num_hidden_layers, hidden_size, layer_shape, activation,
        dropout, batch_norm
    """
    input_dim = data_info["input_dim"]
    output_dim = data_info["output_dim"]

    num_layers = int(hp.get("num_hidden_layers", 2))
    hidden_size = int(hp.get("hidden_size", 128))
    layer_shape = hp.get("layer_shape", "uniform")
    activation = hp.get("activation", "relu")
    dropout_p = float(hp.get("dropout", 0.0))
    use_bn = bool(hp.get("batch_norm", False))

    widths = compute_layer_widths(layer_shape, num_layers, hidden_size)
    act_cls = _ACTIVATIONS.get(activation, nn.ReLU)

    layers: list[nn.Module] = []
    prev_dim = input_dim
    for w in widths:
        layers.append(nn.Linear(prev_dim, w))
        if use_bn:
            layers.append(nn.BatchNorm1d(w))
        layers.append(act_cls())
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        prev_dim = w

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)
