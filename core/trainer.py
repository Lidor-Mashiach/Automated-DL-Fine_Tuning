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

import numpy as np
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


# =============================================================================
# Focal Loss
# =============================================================================
# Focal Loss = (1 - p_correct)^gamma * CrossEntropy
# Down-weights easy examples (p_correct close to 1) so the model focuses on
# hard / minority-class examples. Useful when class distribution is imbalanced.
class FocalLoss(nn.Module):
    """Focal Loss for classification with class imbalance."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)            # per-sample cross-entropy
        p_correct = torch.exp(-ce_loss)               # probability assigned to correct class
        focal_weight = (1.0 - p_correct) ** self.gamma
        return (focal_weight * ce_loss).mean()


def _build_loss_function(hp: dict, task_type: str, data_info: dict):
    """
    Return the appropriate loss function based on hp['loss_function'] and
    the data characteristics.

    Resolution rules:
      - regression -> MSE always.
      - language_modeling -> CrossEntropy with ignore_index=pad_idx (always).
      - classification + loss_function='auto':
          ratio < 3:1   -> CrossEntropy
          ratio < 10:1  -> Focal (gamma 1.5)
          ratio >= 10:1 -> Focal (gamma 2.5)
      - classification + loss_function='cross_entropy' -> CrossEntropy
      - classification + loss_function='focal' -> Focal with hp['focal_gamma']
      - classification + loss_function='mse' -> ignored (use CE) with warning

    `label_smoothing` (when > 0) is applied to all CrossEntropy variants.
    Focal Loss has its own focusing mechanism; label smoothing not combined.
    """
    if task_type == "regression":
        return nn.MSELoss()

    if task_type == "language_modeling":
        pad_idx = 0
        vocab = data_info.get("vocab")
        if vocab is not None:
            pad_idx = vocab.pad_idx
        smoothing = float(hp.get("label_smoothing", 0.0))
        return nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=smoothing)

    loss_choice = hp.get("loss_function", "auto")
    imbalance_ratio = float(data_info.get("imbalance_ratio", 1.0))
    smoothing = float(hp.get("label_smoothing", 0.0))

    if loss_choice == "auto":
        if imbalance_ratio < 3.0:
            return nn.CrossEntropyLoss(label_smoothing=smoothing)
        elif imbalance_ratio < 10.0:
            return FocalLoss(gamma=1.5)
        else:
            return FocalLoss(gamma=2.5)

    if loss_choice == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=smoothing)

    if loss_choice == "focal":
        gamma = float(hp.get("focal_gamma", 2.0))
        return FocalLoss(gamma=gamma)

    # Fallback (e.g. mse on classification): use CE with label smoothing
    return nn.CrossEntropyLoss(label_smoothing=smoothing)


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
    val_perplexity_curve: list[float] = field(default_factory=list)
    duration_seconds: float = 0.0
    failure_reason: str | None = None
    epochs_completed: int = 0
    raw_best_metric: float = 0.0
    raw_best_epoch: int = -1


def _build_optimizer(model, hp):
    name = hp.get("optimizer_name", "adam")
    lr = float(hp.get("learning_rate", 1e-3))
    wd = float(hp.get("weight_decay", 0.0))
    momentum = float(hp.get("momentum", 0.9))
    opt_cls = _OPTIMIZERS.get(name, Adam)

    if name == "sgd":
        return opt_cls(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    if name == "rmsprop":
        return opt_cls(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    # Adam / AdamW: optionally use custom betas if provided
    beta1 = float(hp.get("adam_beta1", 0.9))
    beta2 = float(hp.get("adam_beta2", 0.999))
    return opt_cls(model.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2))


def _build_scheduler(optimizer, hp, epochs):
    """
    Build the LR scheduler from hp.

    If `lr_warmup` is set (>0), the scheduler is wrapped so that the first
    `lr_warmup` epochs use a linear warmup from 0 -> base_lr, and the main
    scheduler kicks in afterwards.
    """
    name = hp.get("lr_scheduler", "none")
    warmup_epochs = int(hp.get("lr_warmup", 0))

    # Build the main scheduler
    main = None
    if name == "cosine":
        # Account for warmup epochs in T_max so cosine completes by end-of-training
        main = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    elif name == "step":
        main = StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    elif name == "reduce_on_plateau":
        main = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    if warmup_epochs <= 0:
        return main

    # Wrap in a LambdaLR-style warmup that switches to main after N epochs.
    # We use SequentialLR to chain: warmup -> main.
    from torch.optim.lr_scheduler import LambdaLR, SequentialLR
    warmup = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (epoch + 1) / max(1, warmup_epochs),
    )
    if main is None:
        # Just warmup, then stay at base LR (use ConstantLR-like behavior via lambda)
        return warmup
    # ReduceLROnPlateau can't be combined via SequentialLR (it's epoch-based,
    # but its step() takes a metric arg). For that case, skip warmup-wrapping.
    if name == "reduce_on_plateau":
        return main
    try:
        return SequentialLR(
            optimizer, schedulers=[warmup, main],
            milestones=[warmup_epochs],
        )
    except Exception:
        # Older torch versions may not support SequentialLR cleanly - fall back
        return main


def _compute_metric(outputs, targets, task_type, loss_fn=None, pad_idx=0):
    """
    Compute a 'higher-is-better' metric for a batch.

    classification    -> accuracy in [0, 1]
    regression        -> -RMSE  (higher = better)
    language_modeling -> -loss  (higher = better; loss == NLL)
    """
    if task_type == "classification":
        preds = outputs.argmax(dim=1)
        return (preds == targets).float().mean().item()
    if task_type == "language_modeling":
        # outputs: (B, T, V), targets: (B, T)
        if loss_fn is not None:
            B, T, V = outputs.shape
            loss = loss_fn(outputs.reshape(B * T, V), targets.reshape(B * T))
            return -float(loss.item())
        # Fallback: token-level accuracy
        preds = outputs.argmax(dim=-1)
        mask = (targets != pad_idx)
        if mask.sum() == 0:
            return 0.0
        correct = ((preds == targets) & mask).float().sum()
        return float(correct / mask.sum())
    diff = outputs.squeeze() - targets.squeeze()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    return -rmse


def _apply_cutmix(xb, yb, alpha: float = 1.0, num_classes: int = 0):
    """
    CutMix: cut a random rectangular patch from another image, paste it into
    the current one, and mix the labels proportionally to the patch area.

    Returns (mixed_x, mixed_y_one_hot or None, lam).
    For classification only (regression skipped).
    """
    if alpha <= 0 or num_classes <= 0 or xb.ndim != 4:
        return xb, None, 1.0

    lam = float(np.random.beta(alpha, alpha))
    batch_size = xb.size(0)
    permuted = torch.randperm(batch_size, device=xb.device)

    _, _, h, w = xb.shape
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    if cut_h <= 0 or cut_w <= 0:
        return xb, None, 1.0

    cy = np.random.randint(h)
    cx = np.random.randint(w)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)

    mixed_x = xb.clone()
    mixed_x[:, :, y1:y2, x1:x2] = xb[permuted, :, y1:y2, x1:x2]

    # Adjust lam to match actual patch area (boundary effects)
    actual_area = (y2 - y1) * (x2 - x1)
    lam = 1.0 - actual_area / float(h * w)

    yb_a = yb
    yb_b = yb[permuted]
    return mixed_x, (yb_a, yb_b), lam



def _apply_mixup(xb, yb, alpha: float = 0.2, num_classes: int = 0):
    """
    Mixup: blend two samples linearly.
        mixed_x = lam * x_a + (1 - lam) * x_b
        loss = lam * CE(out, y_a) + (1 - lam) * CE(out, y_b)
    Returns (mixed_x, (y_a, y_b), lam) or (xb, None, 1.0) when disabled.
    """
    if alpha <= 0 or num_classes <= 0:
        return xb, None, 1.0

    lam = float(np.random.beta(alpha, alpha))
    batch_size = xb.size(0)
    permuted = torch.randperm(batch_size, device=xb.device)
    mixed_x = lam * xb + (1 - lam) * xb[permuted]
    return mixed_x, (yb, yb[permuted]), lam


def _run_epoch(model, loader, optimizer, loss_fn, device, task_type,
               grad_clip, is_train, grad_accum_steps: int = 1,
               use_amp: bool = False, scaler=None,
               cutmix_alpha: float = 0.0, mixup_alpha: float = 0.0,
               num_classes: int = 0,
               tf_ratio: float = 1.0, unk_idx: int = 1):
    """Run one epoch.

    Args:
        grad_accum_steps: accumulate gradients over this many batches before
                          stepping the optimizer (effective batch size grows).
        use_amp: use torch.cuda.amp autocast for mixed-precision (fp16).
        scaler: GradScaler instance for AMP (required when use_amp=True).
        cutmix_alpha: if > 0 (and classification), apply CutMix on each batch.
        mixup_alpha: if > 0 (and classification), apply Mixup on each batch.
        num_classes: number of classes (required for cutmix/mixup).

    When both cutmix_alpha and mixup_alpha are > 0, the system picks one
    per batch with equal probability.
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_metric = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    grad_accum_steps = max(1, int(grad_accum_steps))
    is_lm = (task_type == "language_modeling")
    pad_idx = 0
    with ctx:
        if is_train:
            optimizer.zero_grad()
        for batch_idx, batch in enumerate(loader):
            # Support either (x, y) or (x, y, midi) batches
            if len(batch) == 3:
                xb, yb, midi = batch
                midi = midi.to(device) if midi is not None else None
            else:
                xb, yb = batch
                midi = None
            xb = xb.to(device)
            yb = yb.to(device)

            if is_lm:
                yb_loss = yb  # (B, T) int targets
                # Teacher forcing simulation via input dropout: with probability
                # (1 - tf_ratio), replace a fraction of input tokens with <unk>
                # to expose the model to noisy/own-prediction-like contexts.
                # Pure parallel LSTMs can't do true scheduled sampling without
                # going step-by-step; this is a pragmatic approximation.
                if is_train and tf_ratio < 1.0:
                    drop_prob = 1.0 - tf_ratio
                    mask = (torch.rand_like(xb.float()) < drop_prob)
                    xb = torch.where(mask, torch.full_like(xb, unk_idx), xb)
            elif task_type == "regression":
                yb_loss = yb.float().view(-1, 1)
            else:
                yb_loss = yb

            # Apply CutMix or Mixup (training only, classification only)
            mix_targets = None
            mix_lam = 1.0
            if is_train and task_type == "classification" and num_classes > 0:
                use_cutmix = cutmix_alpha > 0 and xb.ndim == 4
                use_mixup = mixup_alpha > 0
                if use_cutmix and use_mixup:
                    # Both enabled - pick one per batch
                    if np.random.rand() < 0.5:
                        xb, mix_targets, mix_lam = _apply_cutmix(
                            xb, yb, alpha=cutmix_alpha, num_classes=num_classes
                        )
                    else:
                        xb, mix_targets, mix_lam = _apply_mixup(
                            xb, yb, alpha=mixup_alpha, num_classes=num_classes
                        )
                elif use_cutmix:
                    xb, mix_targets, mix_lam = _apply_cutmix(
                        xb, yb, alpha=cutmix_alpha, num_classes=num_classes
                    )
                elif use_mixup:
                    xb, mix_targets, mix_lam = _apply_mixup(
                        xb, yb, alpha=mixup_alpha, num_classes=num_classes
                    )

            if use_amp and is_train:
                with torch.cuda.amp.autocast():
                    if is_lm:
                        outputs, _ = model(xb, midi)
                        # outputs: (B, T, V), targets: (B, T) -> flatten for CE
                        B, T, V = outputs.shape
                        loss = loss_fn(outputs.reshape(B * T, V), yb_loss.reshape(B * T))
                    else:
                        outputs = model(xb)
                        if mix_targets is not None:
                            ya, yb_perm = mix_targets
                            loss = (mix_lam * loss_fn(outputs, ya)
                                    + (1 - mix_lam) * loss_fn(outputs, yb_perm))
                        else:
                            loss = loss_fn(outputs, yb_loss)
                    loss = loss / grad_accum_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if (batch_idx + 1) % grad_accum_steps == 0:
                        if grad_clip is not None and grad_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
            else:
                if is_lm:
                    outputs, _ = model(xb, midi)
                    B, T, V = outputs.shape
                    loss = loss_fn(outputs.reshape(B * T, V), yb_loss.reshape(B * T))
                else:
                    outputs = model(xb)
                    if mix_targets is not None:
                        ya, yb_perm = mix_targets
                        loss = (mix_lam * loss_fn(outputs, ya)
                                + (1 - mix_lam) * loss_fn(outputs, yb_perm))
                    else:
                        loss = loss_fn(outputs, yb_loss)
                if is_train:
                    (loss / grad_accum_steps).backward()
                    if (batch_idx + 1) % grad_accum_steps == 0:
                        if grad_clip is not None and grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=grad_clip)
                        optimizer.step()
                        optimizer.zero_grad()

            total_loss += loss.item() * (grad_accum_steps if is_train else 1)
            total_metric += _compute_metric(outputs, yb, task_type,
                                              loss_fn=loss_fn, pad_idx=pad_idx)
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
    extra_loader_kwargs["num_workers"] = int(hp.get("num_workers", 0))
    if data_info.get("data_type") == "image":
        extra_loader_kwargs["image_size"] = int(hp.get("image_size", 64))
        extra_loader_kwargs["augmentation"] = hp.get("data_augmentation", "none")
        extra_loader_kwargs["cutout"] = float(hp.get("cutout", 0.0))
    elif data_info.get("data_type") in ("text", "lyrics"):
        extra_loader_kwargs["seq_len"] = int(hp.get(
            "sequence_length", data_info.get("default_seq_len", 32)
        ))
        # text augmentation only applies to plain text classification, not LM
        if data_info.get("data_type") == "text":
            extra_loader_kwargs["text_augmentation"] = hp.get("text_augmentation", "none")
            extra_loader_kwargs["text_augmentation_prob"] = float(
                hp.get("text_augmentation_prob", 0.0)
            )

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
    loss_fn = _build_loss_function(hp, task_type, data_info)
    optimizer = _build_optimizer(model, hp)
    scheduler = _build_scheduler(optimizer, hp, epochs)

    # Mixed precision setup
    use_amp = bool(hp.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    grad_accum_steps = max(1, int(hp.get("gradient_accumulation_steps", 1)))
    cutmix_alpha = float(hp.get("cutmix", 0.0))
    mixup_alpha = float(hp.get("mixup", 0.0))
    # LM-specific (used only when task_type == "language_modeling")
    teacher_forcing_ratio = float(hp.get("teacher_forcing_ratio", 1.0))
    max_words_per_line = int(hp.get("max_words_per_line", 100))
    min_words_per_line = int(hp.get("min_words_per_line", 0))
    num_classes = int(data_info.get("output_dim", 0))

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
                grad_accum_steps=grad_accum_steps,
                use_amp=use_amp, scaler=scaler,
                cutmix_alpha=cutmix_alpha, mixup_alpha=mixup_alpha,
                num_classes=num_classes,
                tf_ratio=teacher_forcing_ratio,
                unk_idx=(data_info["vocab"].unk_idx
                         if task_type == "language_modeling" and data_info.get("vocab")
                         else 1),
            )
            val_loss, val_metric = _run_epoch(
                model, val_loader, optimizer, loss_fn, device,
                task_type, grad_clip, False,
                grad_accum_steps=1, use_amp=False, scaler=None,
                tf_ratio=1.0,  # never apply teacher forcing perturbation on val
                unk_idx=1,
            )

            if not math.isfinite(tr_loss) or not math.isfinite(val_loss):
                result.status = "diverged"
                result.failure_reason = f"Non-finite loss at epoch {epoch}"
                break

            result.train_loss_curve.append(tr_loss)
            result.val_loss_curve.append(val_loss)
            result.train_metric_curve.append(tr_metric)
            result.val_metric_curve.append(val_metric)
            if task_type == "language_modeling":
                ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                result.val_perplexity_curve.append(ppl)
            result.epochs_completed = epoch + 1

            # Compact per-epoch progress log (every 5 epochs, plus last)
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                if task_type == "language_modeling":
                    # Perplexity = exp(loss). Loss is the optimization signal;
                    # perplexity is the human-friendly metric for LM.
                    ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                    print(f"[{trial_id}]   epoch {epoch+1:>3}/{epochs}  "
                          f"loss={val_loss:.4f}  ppl={ppl:.2f}")
                else:
                    print(f"[{trial_id}]   epoch {epoch+1:>3}/{epochs}  "
                          f"loss={val_loss:.4f}  metric={val_metric:.4f}")

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
