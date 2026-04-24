"""
CNN builder - VGG-style CNN גמיש.
"""

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}


def build_cnn(hp: dict, data_info: dict) -> nn.Module:
    """
    בונה CNN בסגנון VGG: בלוקים של Conv-Act-Pool, ואז FC head.

    Args:
        hp: num_conv_blocks, base_filters, kernel_size, pooling, num_fc_layers,
            fc_size, activation, dropout, batch_norm.
        data_info: input_channels, image_size, output_dim.
    """
    in_channels = data_info.get("input_channels", 3)
    image_size = data_info.get("image_size", 64)
    output_dim = data_info["output_dim"]

    num_blocks = int(hp.get("num_conv_blocks", 3))
    base_filters = int(hp.get("base_filters", 32))
    kernel = int(hp.get("kernel_size", 3))
    pool = hp.get("pooling", "max")
    num_fc = int(hp.get("num_fc_layers", 1))
    fc_size = int(hp.get("fc_size", 256))
    activation = hp.get("activation", "relu")
    dropout_p = float(hp.get("dropout", 0.0))
    use_bn = bool(hp.get("batch_norm", True))

    act_cls = _ACTIVATIONS.get(activation, nn.ReLU)
    pool_cls = nn.MaxPool2d if pool == "max" else nn.AvgPool2d
    padding = kernel // 2  # "same" padding

    # בלוקים קונבולוציוניים
    conv_layers: list[nn.Module] = []
    prev_c = in_channels
    current_size = image_size
    for i in range(num_blocks):
        filters = base_filters * (2 ** i)
        conv_layers.append(
            nn.Conv2d(prev_c, filters, kernel_size=kernel, padding=padding)
        )
        if use_bn:
            conv_layers.append(nn.BatchNorm2d(filters))
        conv_layers.append(act_cls())
        conv_layers.append(pool_cls(kernel_size=2))
        prev_c = filters
        current_size = current_size // 2
        # הגנה: אם התמונה כבר קטנה מדי, עצור להוסיף בלוקים
        if current_size < 2:
            break

    # חישוב גודל הפלאט אחרי כל הבלוקים (מסתמך על המעקב ב-current_size)
    flatten_dim = prev_c * max(1, current_size) * max(1, current_size)

    # FC head
    fc_layers: list[nn.Module] = [nn.Flatten()]
    prev_dim = flatten_dim
    for i in range(num_fc):
        width = max(32, fc_size // (2 ** i))
        fc_layers.append(nn.Linear(prev_dim, width))
        fc_layers.append(act_cls())
        if dropout_p > 0:
            fc_layers.append(nn.Dropout(dropout_p))
        prev_dim = width
    fc_layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*conv_layers, *fc_layers)
