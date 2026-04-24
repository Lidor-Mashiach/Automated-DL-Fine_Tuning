"""
MLP builder - בונה רשת fully-connected גמישה לפי hyperparameters.
"""

import torch.nn as nn


# מפת שמות אקטיבציה למימוש torch
_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
}


def build_mlp(hp: dict, data_info: dict) -> nn.Module:
    """
    בונה MLP לפי hyperparameters.

    Args:
        hp: dict עם המפתחות: num_hidden_layers, hidden_size, layer_shape,
            activation, dropout, batch_norm.
        data_info: dict עם input_dim ו-output_dim.
    """
    input_dim = data_info["input_dim"]
    output_dim = data_info["output_dim"]

    num_layers = int(hp.get("num_hidden_layers", 2))
    hidden_size = int(hp.get("hidden_size", 128))
    layer_shape = hp.get("layer_shape", "uniform")
    activation = hp.get("activation", "relu")
    dropout_p = float(hp.get("dropout", 0.0))
    use_bn = bool(hp.get("batch_norm", False))

    # חישוב רוחב כל שכבה לפי layer_shape
    if layer_shape == "funnel":
        widths = [max(8, hidden_size // (2 ** i)) for i in range(num_layers)]
    else:  # uniform
        widths = [hidden_size] * num_layers

    act_cls = _ACTIVATIONS.get(activation, nn.ReLU)

    # הרכבת רשימת השכבות
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

    # ראש פלט - לינארי ללא אקטיבציה (softmax/loss יטפלו).
    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)
