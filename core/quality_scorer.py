"""
Quality scorer.

Computes a single quality score per trial, used by FTTS to rank nodes in the
priority queue. The score is a weighted combination of four components, each
measuring a different aspect of model quality:

  1. best_metric      - smoothed peak performance (how good).
  2. stability        - how stable the model is around its peak (how reliable).
  3. convergence_speed - how healthily the model learned (not too fast / slow).
  4. generalization_gap - the train-val gap (is it overfitting?).

All four components are normalized to [0, 1] before weighting, so the final
quality_score is also in [0, 1]. This makes thresholds and comparisons across
runs intuitive.

Weight profiles are defined in configs/strategies/ftts.yaml. Users can pick
a named profile (performance / balanced / robust) or define custom weights.
"""

from dataclasses import dataclass
from typing import Sequence

import math
import numpy as np

from core.smoothing import moving_average, tail_average


# =============================================================================
# Profile presets - weights for the four score components
# =============================================================================
# Each profile sums to 1.0. The user picks one via `scoring.profile` in
# configs/strategies/ftts.yaml, or provides custom weights.
PROFILES = {
    # "performance": prioritize peak metric over everything else
    "performance": {
        "best_metric":        0.70,
        "stability":          0.10,
        "convergence_speed":  0.10,
        "generalization_gap": 0.10,
    },
    # "balanced" (default): good tradeoff for most tasks
    "balanced": {
        "best_metric":        0.50,
        "stability":          0.20,
        "convergence_speed":  0.15,
        "generalization_gap": 0.15,
    },
    # "robust": prioritize stability and generalization
    "robust": {
        "best_metric":        0.35,
        "stability":          0.30,
        "convergence_speed":  0.15,
        "generalization_gap": 0.20,
    },
}


@dataclass
class QualityBreakdown:
    """Detailed breakdown of a trial's quality score, for reporting."""
    total: float
    best_metric_component: float
    stability_component: float
    convergence_speed_component: float
    generalization_gap_component: float
    # Raw values (before normalization), useful for reports
    raw_best_smoothed: float
    raw_stability_std: float
    raw_convergence_epoch_ratio: float
    raw_gen_gap: float

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "best_metric_component": self.best_metric_component,
            "stability_component": self.stability_component,
            "convergence_speed_component": self.convergence_speed_component,
            "generalization_gap_component": self.generalization_gap_component,
            "raw_best_smoothed": self.raw_best_smoothed,
            "raw_stability_std": self.raw_stability_std,
            "raw_convergence_epoch_ratio": self.raw_convergence_epoch_ratio,
            "raw_gen_gap": self.raw_gen_gap,
        }


# =============================================================================
# Component calculators
# =============================================================================

def _compute_best_metric_component(val_metric: Sequence[float],
                                   window: int,
                                   task_type: str) -> tuple[float, float]:
    """
    Best metric component: smoothed peak of val_metric.

    Returns (normalized_component_in_0_1, raw_smoothed_best).

    For classification, val_metric is already in [0, 1] (accuracy), so no
    normalization needed. For regression, val_metric is -RMSE (higher=better),
    and we map to [0, 1] via a soft squash.

    NaN/Inf handling: if the trial diverged (curve contains NaN or Inf),
    we still consider the pre-divergence portion of the curve but cap the
    final score at 0 to mark it as unusable. This prevents a trial that
    briefly looked good before exploding from being chosen as 'best'.
    """
    if not val_metric:
        return 0.0, 0.0

    # Detect divergence: any non-finite value in the curve disqualifies the trial.
    has_diverged = any(not math.isfinite(v) for v in val_metric)

    # Compute the smoothed best from the FINITE portion of the curve
    finite_curve = [v for v in val_metric if math.isfinite(v)]
    if not finite_curve:
        # Entire curve is NaN/Inf - completely failed
        return 0.0, float("-inf")

    smoothed = moving_average(finite_curve, window)
    best = max(smoothed)

    if task_type == "classification":
        # Accuracy already in [0, 1]
        return max(0.0, min(1.0, best)), best
    elif task_type == "language_modeling":
        # val_metric is -loss (higher=better). Convert to a 0-1 score via
        # perplexity squash: score = 1 / (1 + ppl/100)
        # - loss=0 -> ppl=1   -> score ~ 0.99
        # - loss=2 -> ppl~7.4 -> score ~ 0.93
        # - loss=5 -> ppl~148 -> score ~ 0.40
        # - loss=10 -> ppl~22026 -> score ~ 0.005
        loss_val = -best  # convert back to positive loss
        try:
            ppl = float(np.exp(loss_val))
        except (OverflowError, FloatingPointError):
            ppl = 1e9
        score = 1.0 / (1.0 + ppl / 100.0)
        return max(0.0, min(1.0, score)), best
    else:
        # Regression: best is -RMSE, maps to [0, 1] via 1 / (1 + RMSE)
        # RMSE=0 -> score=1. RMSE=1 -> score=0.5. RMSE=infinity -> score=0.
        rmse = -best
        return 1.0 / (1.0 + max(0.0, rmse)), best


def _compute_stability_component(val_metric: Sequence[float],
                                 window: int) -> tuple[float, float]:
    """
    Stability: standard deviation of val_metric in the last `window` epochs
    (around the peak region). Lower std -> higher stability component.

    Returns (normalized_component_in_0_1, raw_std).
    """
    if len(val_metric) < 2:
        return 1.0, 0.0  # too short to measure; assume stable

    tail = list(val_metric[-window:]) if len(val_metric) >= window else list(val_metric)
    mean = sum(tail) / len(tail)
    variance = sum((v - mean) ** 2 for v in tail) / len(tail)
    std = variance ** 0.5

    # Map std -> [0, 1]. std=0 -> 1.0. std=0.1 -> ~0.5. std>=0.3 -> near 0.
    # This squash is heuristic but effective for both classification and regression.
    component = 1.0 / (1.0 + 10.0 * std)
    return component, std


def _compute_convergence_speed_component(val_metric: Sequence[float],
                                         window: int) -> tuple[float, float]:
    """
    Convergence speed: at what fraction of total epochs did we reach 90% of
    the smoothed peak?

    Returns (normalized_component_in_0_1, raw_ratio).

    Scoring:
      - Too early (<20% of epochs): suspicious (local minimum), partial credit.
      - Sweet spot (30-60% of epochs): full credit.
      - Too late (>85% of epochs): under-trained, partial credit.
    """
    if len(val_metric) < 3:
        return 0.5, 0.5  # too short to judge

    smoothed = moving_average(val_metric, window)
    peak = max(smoothed)

    if peak <= 0:
        return 0.0, 0.0

    # Find first epoch where we reached 90% of peak
    target = 0.9 * peak
    reach_idx = next((i for i, v in enumerate(smoothed) if v >= target),
                     len(smoothed) - 1)
    ratio = reach_idx / max(1, len(smoothed) - 1)

    # Scoring curve: triangle-like, peaking at 0.45
    if ratio < 0.2:
        # Too fast - probably converged to local min
        component = 0.5 + 2.5 * ratio   # 0.5 at ratio=0, 1.0 at ratio=0.2
    elif ratio <= 0.6:
        # Sweet spot
        component = 1.0
    elif ratio <= 0.85:
        # Getting late
        component = 1.0 - (ratio - 0.6) * 2.0   # 1.0 at 0.6, 0.5 at 0.85
    else:
        # Too late
        component = max(0.1, 0.5 - (ratio - 0.85) * 3.0)

    return max(0.0, min(1.0, component)), ratio


def _compute_generalization_gap_component(
    train_metric: Sequence[float],
    val_metric: Sequence[float],
    window: int,
) -> tuple[float, float]:
    """
    Generalization gap: average (train_metric - val_metric) over the last
    `window` epochs. Smaller gap -> better generalization.

    Returns (normalized_component_in_0_1, raw_gap).
    """
    if not train_metric or not val_metric:
        return 0.5, 0.0

    n = min(len(train_metric), len(val_metric))
    if n == 0:
        return 0.5, 0.0

    train_tail = tail_average(train_metric[:n], window)
    val_tail = tail_average(val_metric[:n], window)
    gap = train_tail - val_tail

    # Negative gap (val > train) is also healthy, treat as 0 gap
    abs_gap = max(0.0, gap)

    # Map abs_gap -> [0, 1]. gap=0 -> 1.0. gap=0.1 -> ~0.5. gap>=0.3 -> near 0.
    component = 1.0 / (1.0 + 10.0 * abs_gap)
    return component, gap


# =============================================================================
# Main entry point
# =============================================================================

def compute_quality_score(
    train_metric: Sequence[float],
    val_metric: Sequence[float],
    weights: dict[str, float],
    smoothing_window: int,
    task_type: str,
) -> QualityBreakdown:
    """
    Compute the full quality score and its breakdown.

    Args:
        train_metric: per-epoch training metric curve.
        val_metric: per-epoch validation metric curve.
        weights: dict with keys best_metric, stability, convergence_speed,
                 generalization_gap. Must sum to 1.0 (not strictly validated
                 here, but recommended).
        smoothing_window: window size for moving average.
        task_type: "classification" or "regression".

    Returns:
        QualityBreakdown with total and all components.
    """
    best_c, raw_best = _compute_best_metric_component(
        val_metric, smoothing_window, task_type
    )
    stab_c, raw_std = _compute_stability_component(val_metric, smoothing_window)
    speed_c, raw_ratio = _compute_convergence_speed_component(
        val_metric, smoothing_window
    )
    gap_c, raw_gap = _compute_generalization_gap_component(
        train_metric, val_metric, smoothing_window
    )

    total = (
        weights["best_metric"]        * best_c +
        weights["stability"]          * stab_c +
        weights["convergence_speed"]  * speed_c +
        weights["generalization_gap"] * gap_c
    )

    # Divergence penalty: if the val_metric curve contains NaN or Inf, the
    # trial likely exploded mid-training. Even if the pre-divergence portion
    # looked OK, the HP combination is unreliable for refit. We zero out the
    # quality so this trial cannot be picked as 'best'. The Analyzer's
    # `diverged` verdict already excludes it from FTTS exploitation; this
    # ensures the orchestrator's best-tracking also excludes it.
    if val_metric and any(not math.isfinite(v) for v in val_metric):
        total = 0.0

    return QualityBreakdown(
        total=total,
        best_metric_component=best_c,
        stability_component=stab_c,
        convergence_speed_component=speed_c,
        generalization_gap_component=gap_c,
        raw_best_smoothed=raw_best,
        raw_stability_std=raw_std,
        raw_convergence_epoch_ratio=raw_ratio,
        raw_gen_gap=raw_gap,
    )


def resolve_weights(profile: str, custom_weights: dict | None = None) -> dict:
    """
    Resolve the weight dict from a profile name or custom weights.

    Args:
        profile: "performance" | "balanced" | "robust" | "custom".
        custom_weights: required if profile == "custom". Must have the 4 keys.

    Returns:
        dict with the four weights, always summing to 1.0 (normalized).
    """
    if profile == "custom":
        if custom_weights is None:
            raise ValueError(
                "profile='custom' requires custom_weights to be provided."
            )
        required = {"best_metric", "stability", "convergence_speed",
                    "generalization_gap"}
        missing = required - set(custom_weights)
        if missing:
            raise ValueError(
                f"custom_weights is missing keys: {missing}."
            )
        total = sum(custom_weights[k] for k in required)
        if total <= 0:
            raise ValueError("custom_weights must sum to a positive value.")
        # Normalize to sum to 1.0
        return {k: custom_weights[k] / total for k in required}

    if profile not in PROFILES:
        raise ValueError(
            f"Unknown profile '{profile}'. "
            f"Available: {sorted(PROFILES)} or 'custom'."
        )
    return dict(PROFILES[profile])
