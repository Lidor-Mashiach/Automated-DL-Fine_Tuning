"""
Strategy config loader.

Strategy configs (configs/strategies/*.yaml) are simpler than architecture
configs: they're just a nested dict of settings. No enabled/initial_value
parameter model needed.

This module just loads and returns the raw dict, with minimal validation.
"""

from pathlib import Path
from typing import Any

import yaml


def load_strategy_config(strategy: str,
                         configs_dir: str | Path = "configs/strategies") -> dict[str, Any]:
    """
    Load the YAML config for a search strategy.

    Args:
        strategy: "ftts" | "bayesian" | "grid".
        configs_dir: base dir.

    Returns:
        dict of strategy settings, loaded from YAML.
    """
    path = Path(configs_dir) / f"{strategy}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Strategy config not found: {path}. "
            f"Expected at {configs_dir}/{strategy}.yaml."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def require(cfg: dict, key_path: str, default: Any = None) -> Any:
    """
    Get a nested value from the config. Supports dotted paths like
    "stopping.max_trials". Returns default if missing.
    """
    parts = key_path.split(".")
    cur: Any = cfg
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
