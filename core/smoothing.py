"""
Smoothing utilities.

Moving-average smoothing for training curves. Used everywhere in the system:
  - Early stopping decisions (trial-level).
  - Quality scoring (run-level).
  - Target accuracy comparisons.
  - Analyzer diagnostics.

The smoothing_window size is read from the strategy config, but callers pass it
explicitly so this module stays config-free.
"""

from typing import Sequence


def moving_average(values: Sequence[float], window: int) -> list[float]:
    """
    Return a trailing moving average of `values` with given `window`.

    For position i, the output is the mean of values[max(0, i-window+1) : i+1].
    The first (window-1) positions use shorter windows (partial averages),
    which is intentional — it means early-epoch smoothing is less stable but
    never drops data.

    Args:
        values: sequence of numeric values (e.g., val_loss per epoch).
        window: window size. window <= 1 returns `values` unchanged.

    Returns:
        list of the same length as `values`, smoothed.
    """
    if window <= 1 or not values:
        return list(values)
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start: i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def best_smoothed_value(values: Sequence[float], window: int,
                        higher_is_better: bool = True) -> tuple[float, int]:
    """
    Find the best smoothed value in a metric curve.

    Args:
        values: raw metric curve (e.g., val_accuracy per epoch).
        window: smoothing window.
        higher_is_better: True for accuracy, False for loss.

    Returns:
        (best_smoothed_value, epoch_index_where_it_occurred).
        Returns (float('-inf') or float('inf'), -1) if values is empty.
    """
    if not values:
        return (float("-inf") if higher_is_better else float("inf"), -1)
    smoothed = moving_average(values, window)
    if higher_is_better:
        best_val = max(smoothed)
    else:
        best_val = min(smoothed)
    best_idx = smoothed.index(best_val)
    return best_val, best_idx


def tail_average(values: Sequence[float], window: int) -> float:
    """
    Average of the last `window` values. Useful for "where is the curve now?"
    without needing full smoothing. Returns mean of all if window > len.
    """
    if not values:
        return 0.0
    tail = values[-window:] if len(values) >= window else list(values)
    return sum(tail) / len(tail)
