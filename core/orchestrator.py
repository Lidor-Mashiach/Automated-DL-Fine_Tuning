"""
Orchestrator
------------
The operational brain. Runs the full tuning loop:
  1. Loads architecture config + strategy config + data.
  2. Executes the first (root) trial.
  3. For each subsequent trial:
     - strategy proposes next hyperparameters,
     - trainer runs the trial,
     - analyzer diagnoses,
     - quality_scorer computes the composite score,
     - strategy absorbs the result,
     - reporter logs everything,
     - plotter updates the best-so-far plot if the trial is a new best,
     - stop conditions are checked.
  4. Returns a summary dict.

Parallelism: controlled by max_parallel_experiments in the strategy config.
A ThreadPoolExecutor with a bounded worker count runs trials concurrently.
Each trial builds its own model, so there is no shared mutable state between
trials apart from the ExperimentTree (which uses a Lock internally).
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.analyzer import analyze
from core.config_manager import ConfigManager
from core.quality_scorer import compute_quality_score, resolve_weights
from core.run_config import RunConfig
from core.strategy_config import load_strategy_config
from core.trainer import TrialResult, train_trial
from data_loaders import load_data
from models import build_model
from reporting.plotter import plot_best_trial
from reporting.reporter import Reporter
from search_strategies import build_strategy


class Orchestrator:
    """Runs a full AutoTune-NN session."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self._set_seeds()

        # Output directory (unique per run)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{cfg.architecture}_{cfg.search_strategy}"
        self.run_dir = Path(cfg.experiments_root) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.report_path = self.run_dir / "report.txt"
        self.best_plot_path = self.run_dir / "best_trial.png"

        # Load configs
        self.cm = ConfigManager(cfg.architecture)
        self.strategy_cfg = load_strategy_config(cfg.search_strategy)

        # Scoring setup (from strategy config)
        scoring_cfg = self.strategy_cfg.get("scoring", {}) or {}
        self.smoothing_window = int(scoring_cfg.get("smoothing_window", 5))
        profile = scoring_cfg.get("profile", "balanced")
        custom_weights = scoring_cfg.get("custom_weights")
        self.quality_weights = resolve_weights(profile, custom_weights)

        # Stopping conditions (all optional; at least one must be set)
        stopping = self.strategy_cfg.get("stopping", {}) or {}
        self.stop_max_trials = stopping.get("max_trials")
        self.stop_time_minutes = stopping.get("time_limit_minutes")
        self.stop_target_accuracy = stopping.get("target_accuracy")
        self.stop_convergence_patience = stopping.get("convergence_patience")
        self._validate_stopping_active()

        # Parallelism
        exec_cfg = self.strategy_cfg.get("execution", {}) or {}
        self.max_parallel = int(exec_cfg.get("max_parallel_experiments", 1))
        self._warn_if_unsafe_parallelism()

        # Data + strategy
        self.make_loaders, self.data_info = load_data(cfg)
        self.data_info.setdefault("task_type", cfg.task_type)
        self.strategy = build_strategy(
            cfg.search_strategy, self.cm, seed=cfg.random_seed
        )

        # Reporter
        self.reporter = Reporter(
            self.report_path,
            {**cfg.to_dict(),
             "strategy_profile": profile,
             "smoothing_window": self.smoothing_window,
             "quality_weights": self.quality_weights},
            self.cm,
        )

        # Runtime state
        self.device = torch.device(cfg.device)
        self.trial_counter = 0
        self.best_quality = float("-inf")
        self.best_trial_id: str | None = None
        self.best_result: TrialResult | None = None
        self.no_improvement_count = 0
        self.start_time = time.time()
        self.stop_reason: str = ""

    # ========================================================= public API

    def run(self) -> dict:
        """Run the full tuning session. Returns a summary dict."""
        print(f"\n[orchestrator] Starting run: {self.run_dir.name}")
        print(f"[orchestrator] Architecture: {self.cfg.architecture}")
        print(f"[orchestrator] Strategy:     {self.cfg.search_strategy}")
        print(f"[orchestrator] Device:       {self.cfg.device}")

        try:
            if self.cfg.search_strategy == "ftts":
                self._run_ftts()
            elif self.cfg.search_strategy == "bayesian":
                self._run_bayesian()
            elif self.cfg.search_strategy == "grid":
                self._run_grid()
        except KeyboardInterrupt:
            self.stop_reason = "Interrupted by user (Ctrl+C)"
            print(f"\n[orchestrator] {self.stop_reason}")

        self._finalize()
        return self._summary()

    # ========================================================= FTTS runner

    def _run_ftts(self):
        strategy = self.strategy  # FTTS instance

        # Root trial
        root_hp = strategy.initial_hyperparameters()
        root_id = self._new_trial_id()
        strategy.mark_root(root_id, root_hp)
        self._execute_and_record(
            trial_id=root_id, parent_id=None, hp=root_hp,
            action_applied=None,
            rationale="Root trial: starting from initial_value fields of the config.",
        )

        # Main loop
        while not self._should_stop():
            proposal = strategy.propose_next()
            if proposal is None:
                self.stop_reason = "Queue empty - search tree fully explored."
                break
            parent_id, child_hp, action, rationale = proposal
            trial_id = self._new_trial_id()
            self._execute_and_record(
                trial_id=trial_id, parent_id=parent_id, hp=child_hp,
                action_applied=action, rationale=rationale,
            )

    # ========================================================= Bayesian runner

    def _run_bayesian(self):
        strategy = self.strategy
        while not self._should_stop():
            trial_id = self._new_trial_id()
            hp = strategy.propose_next(trial_id)
            # For bayesian: parent is whichever trial was best so far (for reporting)
            parent_id = self.best_trial_id
            rationale = (
                f"Optuna TPE proposal (startup={self.strategy_cfg.get('n_startup_trials', 10)})."
            )
            result, diagnosis, breakdown = self._execute_and_record(
                trial_id=trial_id, parent_id=parent_id, hp=hp,
                action_applied=None, rationale=rationale,
                return_artifacts=True,
            )
            strategy.report_result(trial_id, breakdown.total if breakdown else 0.0,
                                   result.status)

    # ========================================================= Grid runner

    def _run_grid(self):
        strategy = self.strategy
        total = strategy.total()
        print(f"[orchestrator] Grid search will run up to {total} combinations.")
        while not self._should_stop() and strategy.has_more():
            hp = strategy.propose_next()
            if hp is None:
                break
            trial_id = self._new_trial_id()
            rationale = f"Grid combination {self.trial_counter} of {total}."
            self._execute_and_record(
                trial_id=trial_id, parent_id=None, hp=hp,
                action_applied=None, rationale=rationale,
            )
        if not strategy.has_more():
            self.stop_reason = "Grid fully enumerated."

    # ========================================================= trial execution

    def _execute_and_record(self, trial_id: str, parent_id: str | None,
                            hp: dict, action_applied, rationale: str,
                            return_artifacts: bool = False):
        """Build model, train, analyze, score, report, plot, update stats."""
        try:
            model = build_model(self.cfg.architecture, hp, self.data_info)
            epochs_param = self.cm.get_param("epochs")
            epochs = int(epochs_param["initial_value"]) if epochs_param else 30

            es_param = self.cm.get_param("early_stopping")
            patience = None
            if es_param and es_param["initial_value"]:
                patience = es_param["extras"].get("patience", 10)

            result = train_trial(
                trial_id=trial_id,
                parent_trial_id=parent_id,
                model=model,
                hp=hp,
                make_loaders=self.make_loaders,
                data_info=self.data_info,
                device=self.device,
                epochs=epochs,
                early_stopping_patience=patience,
                smoothing_window=self.smoothing_window,
                should_stop=self._time_limit_reached,
            )
        except Exception as exc:
            result = TrialResult(
                trial_id=trial_id, parent_trial_id=parent_id,
                hyperparameters=hp, status="failed",
                best_metric=float("-inf"), best_epoch=-1,
                failure_reason=f"Build/runtime error: {exc}",
            )

        # Analyze
        diagnosis = analyze(result, self.cm, self.smoothing_window)

        # Score
        if result.val_metric_curve and result.train_metric_curve:
            breakdown = compute_quality_score(
                result.train_metric_curve,
                result.val_metric_curve,
                self.quality_weights,
                self.smoothing_window,
                self.data_info.get("task_type", "classification"),
            )
        else:
            breakdown = None

        # Feed back to FTTS tree
        if self.cfg.search_strategy == "ftts":
            self.strategy.register_completed(
                trial_id=trial_id,
                parent_id=parent_id,
                hp=hp,
                quality_breakdown=breakdown,
                diagnosis=diagnosis,
                status="done" if result.status in ("completed", "early_stopped") else result.status,
                rationale=rationale,
                applied_action=action_applied,
            )

        # Track best
        quality = breakdown.total if breakdown else float("-inf")
        improved = quality > self.best_quality
        if improved:
            self.best_quality = quality
            self.best_trial_id = trial_id
            self.best_result = result
            self.no_improvement_count = 0
            plot_best_trial(result, self.best_plot_path)
        else:
            self.no_improvement_count += 1

        # Report
        self.reporter.add_trial(
            trial_index=self.trial_counter,
            result=result,
            diagnosis=diagnosis,
            quality_breakdown=breakdown,
            rationale=rationale,
            parent_id=parent_id,
        )

        # Console log
        tag = "✓ NEW BEST" if improved else "  "
        raw_str = (f"raw={result.raw_best_metric:.4f} "
                   f"smooth={result.best_metric:.4f}") if result.val_metric_curve else "no-metrics"
        print(f"[{trial_id}] {result.status:<14} {raw_str} "
              f"quality={quality:.4f} {tag}")

        if return_artifacts:
            return result, diagnosis, breakdown

    # ================================================= stopping logic

    def _should_stop(self) -> bool:
        # Check each active condition; stop on first hit.
        if self.stop_max_trials is not None:
            if self.trial_counter >= self.stop_max_trials:
                self.stop_reason = f"max_trials={self.stop_max_trials} reached."
                return True
        if self._time_limit_reached():
            self.stop_reason = f"time_limit_minutes={self.stop_time_minutes} reached."
            return True
        if self.stop_target_accuracy is not None and self.best_result is not None:
            if self.best_result.best_metric >= self.stop_target_accuracy:
                self.stop_reason = (
                    f"target_accuracy={self.stop_target_accuracy} reached "
                    f"(smoothed) at {self.best_trial_id}."
                )
                return True
        if self.stop_convergence_patience is not None:
            if self.no_improvement_count >= self.stop_convergence_patience:
                self.stop_reason = (
                    f"convergence: {self.stop_convergence_patience} trials "
                    f"without improvement."
                )
                return True
        return False

    def _time_limit_reached(self) -> bool:
        if self.stop_time_minutes is None:
            return False
        return (time.time() - self.start_time) / 60.0 >= self.stop_time_minutes

    def _validate_stopping_active(self):
        """At least one stopping condition must be set."""
        if (self.stop_max_trials is None and
                self.stop_time_minutes is None and
                self.stop_target_accuracy is None and
                self.stop_convergence_patience is None):
            raise ValueError(
                "At least one stopping condition must be set in the strategy "
                "config (stopping.max_trials / time_limit_minutes / "
                "target_accuracy / convergence_patience). "
                "All are null - the run would be infinite."
            )

    def _warn_if_unsafe_parallelism(self):
        """Warn if GPU + threads > 1."""
        if self.max_parallel > 1 and self.cfg.device.startswith("cuda"):
            print(f"[orchestrator] ⚠️  Warning: max_parallel_experiments="
                  f"{self.max_parallel} with GPU device. Multiple trials will "
                  f"share one GPU, likely hurting throughput. Prefer 1 for GPU runs.")

    # ================================================= utilities

    def _new_trial_id(self) -> str:
        self.trial_counter += 1
        return f"T{self.trial_counter:04d}"

    def _set_seeds(self):
        if self.cfg.random_seed is None:
            return
        random.seed(self.cfg.random_seed)
        np.random.seed(self.cfg.random_seed)
        torch.manual_seed(self.cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.random_seed)

    def _finalize(self):
        summary = self._summary()
        self.reporter.finalize(summary)

    def _summary(self) -> dict:
        return {
            "total_trials": self.trial_counter,
            "best_trial_id": self.best_trial_id or "none",
            "best_quality": self.best_quality if self.best_quality > float("-inf") else 0.0,
            "best_metric_smoothed": self.best_result.best_metric if self.best_result else 0.0,
            "best_metric_raw": self.best_result.raw_best_metric if self.best_result else 0.0,
            "stop_reason": self.stop_reason or "All conditions satisfied.",
            "results_dir": str(self.run_dir),
        }
