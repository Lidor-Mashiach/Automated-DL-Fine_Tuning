"""
Config manager.

Loads a YAML architecture config file, validates it, and exposes a flat list
of parameters to the rest of the system.

Parameter model (v2 - simplified):
    enabled       : bool. Whether the parameter is used in trials at all.
    initial_value : Any or None. Starting value for trial 1 (root of tree).
                    If None, system picks the middle of the range.
    range         : [min, max]. For continuous/integer parameters.
    choices       : [...]. For discrete parameters.
    log           : bool. Sample on log scale (for learning_rate, etc.).

A parameter MUST have either `range` or `choices`, not both. Exception:
parameters without either (pure flags like `early_stopping: enabled: true`)
that have a simple value - handled via the `initial_value` alone.

"Extras" are non-standard fields (like `patience` for early_stopping, or
a `profile` for a method). They're preserved on the parameter dict for the
Trainer to consume.
"""

from pathlib import Path
from typing import Any

import yaml


# Thematic sections expected in each YAML architecture file.
PARAMETER_GROUPS = (
    "parameters",
    "architectures",
    "methods",
    "optimization",
    "training",
)


class ConfigManager:
    """Loads and manages a single architecture config."""

    def __init__(self, architecture: str,
                 configs_dir: str | Path = "configs/architectures"):
        self.architecture = architecture
        self.configs_dir = Path(configs_dir)
        self._raw: dict[str, Any] = {}
        self._parameters: list[dict[str, Any]] = []
        self._load()
        self._normalize()

    # -------------------------------------------------- load & normalize
    def _load(self) -> None:
        """Load the YAML file for the requested architecture."""
        path = self.configs_dir / f"{self.architecture}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}. "
                f"Expected at {self.configs_dir}/{self.architecture}.yaml."
            )
        with open(path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f) or {}

        declared = self._raw.get("architecture_name")
        if declared and declared != self.architecture:
            print(
                f"[ConfigManager] Warning: file declares "
                f"architecture_name='{declared}' but requested "
                f"'{self.architecture}'."
            )

    def _normalize(self) -> None:
        """Flatten the nested YAML into a list of parameter dicts."""
        self._parameters = []
        for group in PARAMETER_GROUPS:
            group_dict = self._raw.get(group, {}) or {}
            for name, spec in group_dict.items():
                self._parameters.append(self._build_param(name, group, spec))

    def _build_param(self, name: str, group: str, spec: dict) -> dict:
        """Build a normalized parameter dict with safe defaults."""
        # Extras: non-standard fields preserved on the param.
        standard_keys = {
            "enabled", "initial_value", "range", "choices", "log",
        }
        extras = {k: v for k, v in spec.items() if k not in standard_keys}

        return {
            "name": name,
            "group": group,
            "enabled": bool(spec.get("enabled", True)),
            "initial_value": spec.get("initial_value"),
            "range": tuple(spec["range"]) if "range" in spec else None,
            "choices": list(spec["choices"]) if "choices" in spec else None,
            "log": bool(spec.get("log", False)),
            "extras": extras,
        }

    # -------------------------------------------------- public API
    @property
    def parameters(self) -> list[dict]:
        """All parameters, enabled or not."""
        return self._parameters

    def active_parameters(self) -> list[dict]:
        """Parameters with enabled=True."""
        return [p for p in self._parameters if p["enabled"]]

    def tunable_parameters(self) -> list[dict]:
        """
        Parameters that the Analyzer/search can modify.
        A parameter is tunable iff it is enabled AND has either a `range` or
        a `choices` list. A plain `initial_value` with no range/choices is
        treated as a constant.
        """
        out = []
        for p in self._parameters:
            if not p["enabled"]:
                continue
            if p["range"] is not None or p["choices"] is not None:
                out.append(p)
        return out

    def constant_values(self) -> dict[str, Any]:
        """
        Parameters that are enabled but have no range/choices -
        their initial_value is used as a fixed constant across all trials.
        """
        out: dict[str, Any] = {}
        for p in self._parameters:
            if not p["enabled"]:
                continue
            if p["range"] is None and p["choices"] is None:
                out[p["name"]] = p["initial_value"]
        return out

    def get_param(self, name: str) -> dict | None:
        """Find an enabled parameter by name. Returns None if not enabled."""
        for p in self._parameters:
            if p["name"] == name and p["enabled"]:
                return p
        return None

    def get_param_any(self, name: str) -> dict | None:
        """Find a parameter by name, enabled or not."""
        for p in self._parameters:
            if p["name"] == name:
                return p
        return None

    def extras_of(self, name: str) -> dict:
        """Extra fields of a parameter (e.g., patience). Empty dict if none."""
        p = self.get_param(name)
        return p["extras"] if p else {}
