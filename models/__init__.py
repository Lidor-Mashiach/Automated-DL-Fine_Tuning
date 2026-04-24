"""
ModelBuilder
------------
פונקציית "מפעל" אחת שמקבלת ארכיטקטורה, hyperparameters ומידע על הדאטה,
ומחזירה מודל PyTorch מוכן לאימון. המימושים הספציפיים לכל ארכיטקטורה
נמצאים במודולים האחרים בתיקייה.
"""

import torch.nn as nn

from models.mlp import build_mlp
from models.cnn import build_cnn
from models.rnn import build_rnn
from models.lstm import build_lstm
from models.transformer import build_transformer


# מיפוי ארכיטקטורה -> פונקציית בנייה
_BUILDERS = {
    "mlp": build_mlp,
    "cnn": build_cnn,
    "rnn": build_rnn,
    "lstm": build_lstm,
    "transformer": build_transformer,
}


def build_model(architecture: str, hp: dict, data_info: dict) -> nn.Module:
    """
    בניית מודל בהתאם לארכיטקטורה.

    Args:
        architecture: "mlp" | "cnn" | "rnn" | "lstm" | "transformer".
        hp: hyperparameters שנבחרו ל-trial הנוכחי (dict שטוח של שם -> ערך).
        data_info: מידע על הדאטה (input_dim, num_classes, וכו' - מ-DataLoader).

    Returns:
        nn.Module מאותחל וזמין לאימון.
    """
    if architecture not in _BUILDERS:
        raise ValueError(f"Unknown architecture: {architecture}")
    return _BUILDERS[architecture](hp, data_info)
