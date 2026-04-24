"""
Analyzer.

The brain of the system. Takes a TrialResult + ConfigManager and produces:

  1. A diagnosis (verdict) describing what happened during training.
  2. A list of prioritized actions to try in the next trial.
  3. A conclusion text for the report.

All curve analysis uses smoothed metrics (moving average) so single-epoch
spikes don't mislead the diagnosis.

Each action has `priority` in [0, 1] reflecting Analyzer confidence. FTTS
uses it in: child_score = parent_quality * action_priority.
"""

from dataclasses import dataclass, field
from typing import Any

from core.smoothing import moving_average


ACTION_TYPES = {
    # learning rate
    "decrease_lr", "increase_lr",
    # regularization
    "increase_dropout", "decrease_dropout",
    "increase_weight_decay", "decrease_weight_decay",
    "add_label_smoothing",
    # capacity
    "add_depth", "reduce_depth",
    "add_width", "reduce_width",
    "change_layer_shape",
    # activation / optimizer
    "change_activation", "change_optimizer",
    # stability
    "add_gradient_clipping", "enable_batch_norm",
    "enable_early_stopping", "add_lr_scheduler", "add_warmup",
    # data / batch
    "increase_augmentation", "add_mixup",
    "reduce_batch_size", "increase_batch_size",
    # recurrent
    "toggle_bidirectional",
    # transformer
    "increase_attention_dropout",
}


@dataclass
class Action:
    """A proposed change for the next trial."""
    type: str
    reason: str
    target_param: str | None = None
    suggested_value: Any = None
    priority: float = 0.5


@dataclass
class Diagnosis:
    """Full Analyzer output for one trial."""
    trial_id: str
    verdict: str
    observations: list[str] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    conclusion: str = ""


# ------------------------------------------------- curve helpers

def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n)) or 1e-9
    return num / den


def _relative_change(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    first, last = values[0], values[-1]
    base = max(abs(first), 1e-9)
    return (last - first) / base


# ------------------------------------------------- param helpers

def _is_tunable(cm, name: str) -> bool:
    """Enabled and has range or choices."""
    p = cm.get_param(name)
    if not p:
        return False
    return p["range"] is not None or p["choices"] is not None


def _is_enabled(cm, name: str) -> bool:
    """Just enabled."""
    return cm.get_param(name) is not None


# ------------------------------------------------- main entry

def analyze(result, config_manager, smoothing_window: int = 5) -> Diagnosis:
    diag = Diagnosis(trial_id=result.trial_id, verdict="unknown")

    # failures
    if result.status == "failed":
        diag.verdict = "failed"
        diag.observations.append(
            f"Trial failed at runtime: {result.failure_reason or 'unknown'}."
        )
        if _is_tunable(config_manager, "batch_size"):
            diag.actions.append(Action(
                type="reduce_batch_size",
                reason="Runtime failures often caused by OOM.",
                target_param="batch_size", priority=0.7,
            ))
        diag.conclusion = "Will try smaller batch and conservative settings."
        return diag

    if result.status == "diverged":
        diag.verdict = "diverged"
        diag.observations.append("Loss became non-finite (NaN/Inf).")
        if _is_tunable(config_manager, "learning_rate"):
            diag.actions.append(Action(
                type="decrease_lr",
                reason="High LR causes gradient explosion.",
                target_param="learning_rate", priority=0.95,
            ))
        if _is_enabled(config_manager, "gradient_clipping"):
            diag.actions.append(Action(
                type="add_gradient_clipping",
                reason="Clipping prevents explosion.",
                target_param="gradient_clipping",
                suggested_value=1.0, priority=0.85,
            ))
        diag.conclusion = "Will lower LR and add gradient clipping."
        return diag

    # no curves
    if not result.val_loss_curve or not result.train_loss_curve:
        diag.verdict = "insufficient_data"
        diag.observations.append("No training curves recorded.")
        diag.conclusion = "Trial ended before curves could be analyzed."
        return diag

    # smoothed curves for decisions
    train_loss_s = moving_average(result.train_loss_curve, smoothing_window)
    val_loss_s = moving_average(result.val_loss_curve, smoothing_window)
    val_metric_s = moving_average(result.val_metric_curve, smoothing_window) \
        if result.val_metric_curve else []

    val_change = _relative_change(val_loss_s)
    tail = val_loss_s[-5:] if len(val_loss_s) >= 5 else val_loss_s
    tail_slope_val = _slope(tail)
    best_val_loss = min(val_loss_s)
    best_idx = val_loss_s.index(best_val_loss)
    final_val_loss = val_loss_s[-1]
    final_gap = train_loss_s[-1] - val_loss_s[-1]
    peak_drop = final_val_loss - best_val_loss
    total_epochs = len(val_loss_s)

    diag.observations.append(f"Completed {result.epochs_completed} epochs.")
    diag.observations.append(
        f"Smoothed val_loss: {val_loss_s[0]:.4f} -> best {best_val_loss:.4f} "
        f"@ epoch {best_idx} -> final {final_val_loss:.4f}."
    )
    if val_metric_s:
        diag.observations.append(
            f"Best smoothed val_metric: {max(val_metric_s):.4f}."
        )

    # 1. failed_to_learn
    if abs(val_change) < 0.05 and tail_slope_val > -1e-4:
        diag.verdict = "failed_to_learn"
        diag.observations.append("val_loss barely moved - not learning.")
        _add_failed_to_learn(diag, config_manager)
        diag.conclusion = (
            "Model did not learn. Try higher LR, more capacity, "
            "different activation, or verify data."
        )
        return diag

    # 2. peaked_and_dropped
    if peak_drop > 0.1 * best_val_loss and best_idx < total_epochs - 2:
        diag.verdict = "peaked_and_dropped"
        diag.observations.append(
            f"val_loss rose by {peak_drop:.4f} after peak (overfitting)."
        )
        _add_overfit(diag, config_manager)
        diag.conclusion = "Model overfit. Strengthen regularization."
        return diag

    # 3. learning_too_slow
    if val_change > -0.2 and abs(tail_slope_val) < 1e-3 and total_epochs >= 10:
        diag.verdict = "learning_too_slow"
        diag.observations.append("Slow learning - small total reduction.")
        _add_slow(diag, config_manager)
        diag.conclusion = "Speed up via higher LR or more capacity."
        return diag

    # 4. learning_too_fast
    if total_epochs >= 9:
        first_third = val_loss_s[: total_epochs // 3]
        rest = val_loss_s[total_epochs // 3:]
        if first_third and rest:
            early_change = _relative_change(first_third)
            rest_change = _relative_change(rest)
            if early_change < -0.4 and abs(rest_change) < 0.05 and final_gap < -0.05:
                diag.verdict = "learning_too_fast"
                diag.observations.append("Sharp early drop then flat with gap.")
                _add_fast(diag, config_manager)
                diag.conclusion = (
                    "LR too high - rushed to local min. Slow down and regularize."
                )
                return diag

    # 5. converged
    if abs(tail_slope_val) < 1e-4 and peak_drop < 0.05 * best_val_loss:
        diag.verdict = "converged"
        diag.observations.append("val_loss stabilized - converged.")
        _add_converged(diag, config_manager)
        diag.conclusion = "Trial converged. Try variations or accept."
        return diag

    # 6. healthy
    diag.verdict = "healthy"
    diag.observations.append("Training progressing reasonably.")
    _add_healthy(diag, config_manager)
    diag.conclusion = "Healthy training. Try minor variations."
    return diag


# ------------------------------------------------- actions per verdict

def _add_failed_to_learn(diag: Diagnosis, cm):
    if _is_tunable(cm, "learning_rate"):
        diag.actions.append(Action(
            type="increase_lr",
            reason="Low LR prevents meaningful updates.",
            target_param="learning_rate", priority=0.85,
        ))
    if _is_tunable(cm, "hidden_size") or _is_tunable(cm, "num_hidden_layers") \
            or _is_tunable(cm, "num_layers") or _is_tunable(cm, "d_model"):
        diag.actions.append(Action(
            type="add_width",
            reason="Wider layers = more capacity.", priority=0.7,
        ))
        diag.actions.append(Action(
            type="add_depth",
            reason="More layers = more expressivity.", priority=0.6,
        ))
    if _is_tunable(cm, "activation"):
        diag.actions.append(Action(
            type="change_activation",
            reason="Activation may saturate - try GELU.",
            target_param="activation",
            suggested_value="gelu", priority=0.45,
        ))
    if _is_tunable(cm, "name"):
        diag.actions.append(Action(
            type="change_optimizer",
            reason="AdamW may unlock learning.",
            target_param="name",
            suggested_value="adamw", priority=0.35,
        ))


def _add_overfit(diag: Diagnosis, cm):
    if _is_tunable(cm, "dropout"):
        diag.actions.append(Action(
            type="increase_dropout",
            reason="Higher dropout improves generalization.",
            target_param="dropout", priority=0.85,
        ))
    if _is_tunable(cm, "weight_decay"):
        diag.actions.append(Action(
            type="increase_weight_decay",
            reason="Stronger L2 regularization.",
            target_param="weight_decay", priority=0.8,
        ))
    if _is_tunable(cm, "data_augmentation"):
        diag.actions.append(Action(
            type="increase_augmentation",
            reason="More augmentation = more effective data.",
            target_param="data_augmentation",
            suggested_value="medium", priority=0.7,
        ))
    if _is_enabled(cm, "mixup"):
        diag.actions.append(Action(
            type="add_mixup",
            reason="Mixup prevents memorization.",
            target_param="mixup", priority=0.6,
        ))
    if _is_enabled(cm, "label_smoothing"):
        diag.actions.append(Action(
            type="add_label_smoothing",
            reason="Label smoothing prevents overconfidence.",
            target_param="label_smoothing",
            suggested_value=0.1, priority=0.55,
        ))
    if _is_tunable(cm, "hidden_size") or _is_tunable(cm, "num_hidden_layers"):
        diag.actions.append(Action(
            type="reduce_width",
            reason="Smaller model generalizes better.", priority=0.4,
        ))


def _add_slow(diag: Diagnosis, cm):
    if _is_tunable(cm, "learning_rate"):
        diag.actions.append(Action(
            type="increase_lr",
            reason="Higher LR accelerates convergence.",
            target_param="learning_rate", priority=0.8,
        ))
    if _is_tunable(cm, "batch_size"):
        diag.actions.append(Action(
            type="reduce_batch_size",
            reason="Smaller batch = more gradient updates.",
            target_param="batch_size", priority=0.6,
        ))
    if _is_tunable(cm, "hidden_size") or _is_tunable(cm, "num_hidden_layers"):
        diag.actions.append(Action(
            type="add_width",
            reason="More capacity may help.", priority=0.55,
        ))
    if _is_tunable(cm, "name"):
        diag.actions.append(Action(
            type="change_optimizer",
            reason="Try a different optimizer.",
            target_param="name",
            suggested_value="adamw", priority=0.3,
        ))


def _add_fast(diag: Diagnosis, cm):
    if _is_tunable(cm, "learning_rate"):
        diag.actions.append(Action(
            type="decrease_lr",
            reason="Lower LR escapes local minimum.",
            target_param="learning_rate", priority=0.9,
        ))
    if _is_tunable(cm, "lr_scheduler"):
        diag.actions.append(Action(
            type="add_lr_scheduler",
            reason="Cosine scheduler reduces LR over time.",
            target_param="lr_scheduler",
            suggested_value="cosine", priority=0.8,
        ))
    if _is_enabled(cm, "lr_warmup"):
        diag.actions.append(Action(
            type="add_warmup",
            reason="Warmup prevents early huge updates.",
            target_param="lr_warmup", priority=0.7,
        ))
    if _is_tunable(cm, "dropout"):
        diag.actions.append(Action(
            type="increase_dropout",
            reason="Slow learning and prevent overfit.",
            target_param="dropout", priority=0.6,
        ))


def _add_converged(diag: Diagnosis, cm):
    if _is_tunable(cm, "layer_shape"):
        diag.actions.append(Action(
            type="change_layer_shape",
            reason="Try different shape (pyramid/funnel/hourglass).",
            target_param="layer_shape", priority=0.55,
        ))
    if _is_tunable(cm, "hidden_size"):
        diag.actions.append(Action(
            type="add_width",
            reason="Slightly more capacity may help.", priority=0.5,
        ))
    if _is_tunable(cm, "weight_decay"):
        diag.actions.append(Action(
            type="decrease_weight_decay",
            reason="Lighter regularization may allow better fit.",
            target_param="weight_decay", priority=0.45,
        ))
    if _is_tunable(cm, "activation"):
        diag.actions.append(Action(
            type="change_activation",
            reason="Different activation, different optimum.",
            target_param="activation", priority=0.3,
        ))


def _add_healthy(diag: Diagnosis, cm):
    if _is_tunable(cm, "learning_rate"):
        diag.actions.append(Action(
            type="decrease_lr",
            reason="Fine-tune LR downward.",
            target_param="learning_rate", priority=0.6,
        ))
        diag.actions.append(Action(
            type="increase_lr",
            reason="Test if slightly higher LR helps.",
            target_param="learning_rate", priority=0.5,
        ))
    if _is_tunable(cm, "dropout"):
        diag.actions.append(Action(
            type="increase_dropout",
            reason="Test stronger regularization.",
            target_param="dropout", priority=0.5,
        ))
    if _is_tunable(cm, "hidden_size"):
        diag.actions.append(Action(
            type="add_width",
            reason="Explore more capacity.", priority=0.45,
        ))
