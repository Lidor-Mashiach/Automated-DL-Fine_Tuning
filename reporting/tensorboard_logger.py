"""
tensorboard_logger.py
---------------------
Optional TensorBoard logging for training curves.

If tensorboard is not installed, the logger silently no-ops. Logs are written
to <run_dir>/tensorboard/ and viewable with `tensorboard --logdir <run_dir>`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class TensorBoardLogger:
    """Wrapper around torch.utils.tensorboard.SummaryWriter with safe no-op."""

    def __init__(self, run_dir: Path, enabled: bool = True):
        self.enabled = bool(enabled)
        self.writer = None
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path(run_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
        except ImportError:
            print("[tensorboard] tensorboard not installed; logging disabled.")
            self.enabled = False
        except Exception as e:
            print(f"[tensorboard] Setup failed ({e}); logging disabled.")
            self.enabled = False

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if not self.enabled or self.writer is None:
            return
        try:
            self.writer.add_scalar(tag, float(value), int(step))
        except Exception:
            pass

    def log_trial_curves(self, trial_id: str, train_loss: list, val_loss: list,
                          train_metric: list = None, val_metric: list = None,
                          task_type: str = None) -> None:
        """Log a full trial's curves under a tag prefix.

        For language_modeling, also logs val/perplexity = exp(val_loss) at every
        epoch (loss is the optimization signal; perplexity is the human metric).
        """
        if not self.enabled or self.writer is None:
            return
        for i, v in enumerate(train_loss):
            self.log_scalar(f"{trial_id}/train_loss", v, i)
        for i, v in enumerate(val_loss):
            self.log_scalar(f"{trial_id}/val_loss", v, i)
        if train_metric:
            for i, v in enumerate(train_metric):
                self.log_scalar(f"{trial_id}/train_metric", v, i)
        if val_metric:
            for i, v in enumerate(val_metric):
                self.log_scalar(f"{trial_id}/val_metric", v, i)
        if task_type == "language_modeling":
            import math
            for i, v in enumerate(val_loss):
                ppl = math.exp(v) if v < 20 else float("inf")
                self.log_scalar(f"{trial_id}/val_perplexity", ppl, i)

    def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
