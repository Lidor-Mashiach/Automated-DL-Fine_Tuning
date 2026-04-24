"""
Bayesian strategy (Optuna wrapper)
-----------------------------------
Uses Optuna's TPE sampler for hyperparameter selection. The Analyzer still
runs on every trial and its diagnosis is included in the report, but Optuna
alone decides what hyperparameters to try next.

This is a simpler strategy than FTTS: no tree, no adaptive steps. It's
included for users who prefer a well-established statistical approach.

If Optuna is not installed, the constructor raises ImportError with a clear
message. The user can switch to `ftts` or `grid` as alternatives.
"""

import math

try:
    import optuna
    from optuna.samplers import TPESampler
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


class BayesianStrategy:
    """Optuna TPE-based search."""

    def __init__(self, config_manager, strategy_config: dict, seed: int | None = None):
        if not _HAS_OPTUNA:
            raise ImportError(
                "optuna is required for the 'bayesian' strategy. "
                "Install it with: pip install optuna"
            )
        self.cm = config_manager
        self.cfg = strategy_config
        n_startup = int(strategy_config.get("n_startup_trials", 10))
        sampler = TPESampler(seed=seed, n_startup_trials=n_startup)
        self._study = optuna.create_study(direction="maximize", sampler=sampler)
        self._trial_mapping: dict[str, Any] = {}  # trial_id -> optuna Trial

    def propose_next(self, trial_id: str) -> dict:
        """Ask Optuna for the next set of hyperparameters."""
        trial = self._study.ask()
        self._trial_mapping[trial_id] = trial

        hp = {}
        for p in self.cm.active_parameters():
            name = p["name"]
            if p["choices"] is not None:
                hp[name] = trial.suggest_categorical(name, p["choices"])
            elif p["range"] is not None:
                lo, hi = p["range"]
                if p["log"]:
                    hp[name] = trial.suggest_float(name, lo, hi, log=True)
                elif isinstance(lo, int) and isinstance(hi, int):
                    hp[name] = trial.suggest_int(name, lo, hi)
                else:
                    hp[name] = trial.suggest_float(name, lo, hi)
            elif p["initial_value"] is not None:
                hp[name] = p["initial_value"]

            # Pass extras (patience, etc.)
            for k, v in p["extras"].items():
                hp[f"{name}__{k}"] = v

        return hp

    def report_result(self, trial_id: str, quality_score: float,
                      status: str) -> None:
        """Feed the trial's outcome back to Optuna."""
        trial = self._trial_mapping.pop(trial_id, None)
        if trial is None:
            return
        if status in ("failed", "diverged"):
            self._study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            self._study.tell(trial, quality_score)
