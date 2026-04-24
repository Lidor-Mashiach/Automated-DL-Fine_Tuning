"""
Trainer
-------
Runs a single trial: builds optimizer + scheduler, trains for up to `epochs`
epochs, tracks per-epoch loss and metric on train/val, and returns a
TrialResult with:
  * status (completed / early_stopped / failed / diverged)
  * the full training curves (needed for the Analyzer)
  * the best smoothed metric and which epoch produced it
  * runtime, failure_reason if any

Early stopping decisions use the SMOOTHED val_loss curve (moving average
with a window defined by the strategy config). This avoids stopping
prematurely on noisy spikes.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau,
)

from core.smoothing import moving_average, best_smoothed_value


_OPTIMIZERS = {
    "adam": Adam, "adamw": AdamW, "sgd": SGD, "rmsprop": RMSprop,
}


@dataclass
class TrialResult:
    """Outcome of a single trial's training."""
    trial_id: str
    parent_trial_id: str | None
    hyperparameters: dict
    status: str
    best_metric: float                       # smoothed best (higher=better)
    best_epoch: int
    train_loss_curve: list[float] = field(default_factory=list)
    val_loss_curve: list[float] = field(default_factory=list)
    train_metric_curve: list[float] = field(default_factory=list)
    val_metric_curve: list[float] = field(default_factory=list)
    duration_seconds: float = 0.0
    failure_reason: str | None = None
    epochs_completed: int = 0
    raw_best_metric: float = 0.0
    raw_best_epoch: int = -1


def _build_optimizer(model, hp):
    name = hp.get("name", "adam")
    lr = float(hp.get("learning_rate", 1e-3))
    wd = float(hp.get("weight_decay", 0.0))
    momentum = float(hp.get("momentum", 0.9))
    opt_cls = _OPTIMIZERS.get(name, Adam)
    if name == "sgd":
        return opt_cls(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    if name == "rmsprop":
        return opt_cls(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    return opt_cls(model.parameters(), lr=lr, weight_decay=wd)


def _build_scheduler(optimizer, hp, epochs):
    name = hp.get("lr_scheduler", "none")
    if name in (None, "none"):
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    if name == "reduce_on_plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    return None


def _compute_metric(outputs, targets, task_type):
    if task_type == "classification":
        preds = outputs.argmax(dim=1)
        return (preds == targets).float().mean().item()
    diff = outputs.squeeze() - targets.squeeze()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    return -rmse


def _run_epoch(model, loader, optimizer, loss_fn, device, task_type,
               grad_clip, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_metric = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(xb)
            yb_loss = yb.float().view(-1, 1) if task_type == "regression" else yb
            loss = loss_fn(outputs, yb_loss)
            if is_train:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    max_norm=grad_clip)
                optimizer.step()
            total_loss += loss.item()
            total_metric += _compute_metric(outputs, yb, task_type)
            n_batches += 1
    return total_loss / max(1, n_batches), total_metric / max(1, n_batches)


def train_trial(
    trial_id: str,
    parent_trial_id: str | None,
    model: nn.Module,
    hp: dict,
    make_loaders: Callable,
    data_info: dict,
    device: torch.device,
    epochs: int,
    early_stopping_patience: int | None = None,
    smoothing_window: int = 5,
    should_stop: Callable[[], bool] | None = None,
) -> TrialResult:
    """
    Train a single trial.

    Early stopping uses smoothed val_loss (moving average of `smoothing_window`
    epochs). Stops if smoothed val_loss hasn't improved for
    `early_stopping_patience` consecutive epochs.
    """
    task_type = data_info.get("task_type", "classification")
    batch_size = int(hp.get("batch_size", 64))
    val_split = float(hp.get("validation_split", 0.2))
    grad_clip = hp.get("gradient_clipping", None)
    if grad_clip == 0:
        grad_clip = None

    extra_loader_kwargs = {}
    if data_info.get("data_type") == "image":
        extra_loader_kwargs["image_size"] = int(hp.get("image_size", 64))
        extra_loader_kwargs["augmentation"] = hp.get("data_augmentation", "none")
    elif data_info.get("data_type") == "text":
        extra_loader_kwargs["seq_len"] = int(hp.get(
            "sequence_length", data_info.get("default_seq_len", 128)
        ))

    try:
        train_loader, val_loader = make_loaders(
            batch_size=batch_size, val_split=val_split, **extra_loader_kwargs
        )
    except Exception as exc:
        return TrialResult(
            trial_id=trial_id, parent_trial_id=parent_trial_id,
            hyperparameters=hp, status="failed",
            best_metric=float("-inf"), best_epoch=-1,
            failure_reason=f"DataLoader error: {exc}",
        )

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    optimizer = _build_optimizer(model, hp)
    scheduler = _build_scheduler(optimizer, hp, epochs)

    result = TrialResult(
        trial_id=trial_id, parent_trial_id=parent_trial_id,
        hyperparameters=hp, status="completed",
        best_metric=float("-inf"), best_epoch=-1,
    )

    start = time.time()
    best_smooth_loss = float("inf")
    epochs_since_improvement = 0

    try:
        for epoch in range(epochs):
            if should_stop is not None and should_stop():
                result.status = "early_stopped"
                result.failure_reason = "Global stop condition triggered"
                break

            tr_loss, tr_metric = _run_epoch(
                model, train_loader, optimizer, loss_fn, device,
                task_type, grad_clip, True,
            )
            val_loss, val_metric = _run_epoch(
                model, val_loader, optimizer, loss_fn, device,
                task_type, grad_clip, False,
            )

            if not math.isfinite(tr_loss) or not math.isfinite(val_loss):
                result.status = "diverged"
                result.failure_reason = f"Non-finite loss at epoch {epoch}"
                break

            result.train_loss_curve.append(tr_loss)
            result.val_loss_curve.append(val_loss)
            result.train_metric_curve.append(tr_metric)
            result.val_metric_curve.append(val_metric)
            result.epochs_completed = epoch + 1

            # Smoothed early stopping
            smoothed = moving_average(result.val_loss_curve, smoothing_window)
            current_smooth_loss = smoothed[-1]
            if current_smooth_loss < best_smooth_loss - 1e-6:
                best_smooth_loss = current_smooth_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if (early_stopping_patience is not None and
                    epochs_since_improvement >= early_stopping_patience):
                result.status = "early_stopped"
                break

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

    except Exception as exc:
        result.status = "failed"
        result.failure_reason = f"Runtime error: {exc}"

    result.duration_seconds = time.time() - start

    if result.val_metric_curve:
        best_smooth, best_smooth_idx = best_smoothed_value(
            result.val_metric_curve, smoothing_window, higher_is_better=True
        )
        result.best_metric = best_smooth
        result.best_epoch = best_smooth_idx
        result.raw_best_metric = max(result.val_metric_curve)
        result.raw_best_epoch = result.val_metric_curve.index(result.raw_best_metric)

    return result
