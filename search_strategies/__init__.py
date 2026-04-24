"""
Search strategies package
-------------------------
Three strategies are available:
  - `ftts`     : Fine-Tuning Tree Search. The default; explores a tree of
                 experiments with Analyzer-guided priorities.
  - `bayesian` : Optuna TPE. Well-established statistical approach. Less
                 interpretable but often effective.
  - `grid`     : Exhaustive grid search. Best for few parameters.

Each strategy is loaded with its own YAML config from configs/strategies/.
"""

from core.strategy_config import load_strategy_config


def build_strategy(strategy_name: str, config_manager, seed: int | None = None):
    """
    Factory function that returns a strategy instance.

    Args:
        strategy_name: "ftts" | "bayesian" | "grid".
        config_manager: architecture ConfigManager.
        seed: random seed for reproducibility.

    Returns:
        An instance of the chosen strategy class.
    """
    strat_cfg = load_strategy_config(strategy_name)

    if strategy_name == "ftts":
        from search_strategies.ftts import FTTS
        return FTTS(config_manager, strat_cfg)

    if strategy_name == "bayesian":
        from search_strategies.bayesian import BayesianStrategy
        return BayesianStrategy(config_manager, strat_cfg, seed=seed)

    if strategy_name == "grid":
        from search_strategies.grid import GridSearchStrategy
        return GridSearchStrategy(config_manager, strat_cfg)

    raise ValueError(
        f"Unknown strategy '{strategy_name}'. "
        f"Available: 'ftts', 'bayesian', 'grid'."
    )
