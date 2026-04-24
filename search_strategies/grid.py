"""
Grid Search strategy
--------------------
Exhaustively tries every combination of discrete parameter values. Continuous
`range` parameters are sampled at N points (configurable via grid_points in
the strategy YAML).

Best suited for: few parameters, small search spaces, when you want certainty
that every combination was tried. For many parameters, prefer FTTS or bayesian.
"""

import itertools
import math


class GridSearchStrategy:
    """Exhaustive grid search."""

    def __init__(self, config_manager, strategy_config: dict):
        self.cm = config_manager
        self.grid_points = int(strategy_config.get("grid_points", 3))
        self._combinations = self._build_grid()
        self._index = 0

    def _build_grid(self) -> list[dict]:
        """Enumerate all hyperparameter combinations."""
        tunable = self.cm.active_parameters()
        names: list[str] = []
        values_per_param: list[list] = []

        for p in tunable:
            names.append(p["name"])
            if p["choices"] is not None:
                values_per_param.append(list(p["choices"]))
            elif p["range"] is not None:
                lo, hi = p["range"]
                if p["log"]:
                    log_lo, log_hi = math.log(max(lo, 1e-12)), math.log(max(hi, 1e-12))
                    steps = [log_lo + (log_hi - log_lo) * i / (self.grid_points - 1)
                             for i in range(self.grid_points)]
                    values_per_param.append([math.exp(s) for s in steps])
                elif isinstance(lo, int) and isinstance(hi, int):
                    step = max(1, (hi - lo) // max(1, self.grid_points - 1))
                    vals = list(range(lo, hi + 1, step))[:self.grid_points]
                    values_per_param.append(vals)
                else:
                    step = (hi - lo) / (self.grid_points - 1)
                    values_per_param.append([lo + step * i for i in range(self.grid_points)])
            elif p["initial_value"] is not None:
                values_per_param.append([p["initial_value"]])
            else:
                values_per_param.append([None])

        combos = []
        for combo in itertools.product(*values_per_param):
            hp = dict(zip(names, combo))
            # Attach extras
            for p in tunable:
                for k, v in p["extras"].items():
                    hp[f"{p['name']}__{k}"] = v
            combos.append(hp)
        return combos

    def has_more(self) -> bool:
        return self._index < len(self._combinations)

    def propose_next(self) -> dict | None:
        if not self.has_more():
            return None
        hp = self._combinations[self._index]
        self._index += 1
        return hp

    def total(self) -> int:
        return len(self._combinations)
