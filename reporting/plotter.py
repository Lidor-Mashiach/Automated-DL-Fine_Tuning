"""
Plotter
-------
יצירת גרפי אימון. נשמר רק הגרף של הניסוי הטוב ביותר (העדכני-הטוב), כך שלא
מצטברים עשרות קבצים.

שימוש ב-matplotlib עם backend "Agg" - לא פותח חלונות ולא חוסם. כל הגרפים
נשמרים כ-PNG לתיקיית הניסוי.
"""

from pathlib import Path

# חובה: הגדרת backend לפני ייבוא של pyplot, כך שלא יפתחו חלונות
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_best_trial(result, target_path: Path) -> None:
    """
    יוצר גרף training/validation loss + metric עבור ניסוי אחד.
    מחליף כל גרף קודם בנתיב הזה.

    Args:
        result: TrialResult.
        target_path: נתיב הקובץ (למשל experiments/<run>/best_trial.png).
                     יוחלף כל פעם.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not result.train_loss_curve:
        return  # אין מה לצייר

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = list(range(1, len(result.train_loss_curve) + 1))

    # ---- loss ----
    ax = axes[0]
    ax.plot(epochs, result.train_loss_curve, label="train", marker="o", markersize=3)
    if result.val_loss_curve:
        ax.plot(epochs, result.val_loss_curve, label="val", marker="s", markersize=3)
    ax.set_title(f"Loss | trial {result.trial_id}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- metric ----
    ax = axes[1]
    if result.train_metric_curve:
        ax.plot(epochs, result.train_metric_curve, label="train", marker="o", markersize=3)
    if result.val_metric_curve:
        ax.plot(epochs, result.val_metric_curve, label="val", marker="s", markersize=3)
    ax.set_title(f"Metric (higher=better) | best={result.best_metric:.4f} @ epoch {result.best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    # כתיבה אטומית: קודם temp ואז rename, למניעת קובץ חצי-כתוב
    tmp = target_path.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=120)
    plt.close(fig)
    tmp.replace(target_path)
