"""
Final Trainer (post-tuning phase)
---------------------------------
After AutoTune-NN finishes tuning, this module:
  1. Rebuilds the model with the best hyperparameters.
  2. Retrains it on Train+Val combined (refit on full training set).
  3. Evaluates on Test_set (the only time Test is touched).
  4. Generates a standalone model.py that the user can run on their own.
  5. Writes test_evaluation.txt with the final results.
  6. Writes final_training.png with the final training curves.

This is the "deliverable" phase - the user takes the artifacts in final/
and uses them as the result of the tuning session.
"""

from pathlib import Path
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

from core.code_generator import generate_model_py
from core.trainer import _build_optimizer, _build_loss_function, _run_epoch
from models import build_model


def run_final_phase(
    cfg, best_hp: dict, best_quality: float,
    best_metric_raw: float, best_metric_smoothed: float,
    data_info: dict, cm, total_trials: int, final_dir: Path,
    ranking_metric: str, quality_weights: dict, smoothing_window: int,
) -> None:
    """Execute the final phase end-to-end."""

    # 1. Build the model architecture for refit
    model = build_model(cfg.architecture, best_hp, data_info)
    device = torch.device(cfg.device)
    model = model.to(device)

    # 2. Train on Train+Val combined (refit on full training data)
    test_metric, test_loss, training_curves = _refit_and_eval(
        model=model, cfg=cfg, hp=best_hp, data_info=data_info,
        cm=cm, device=device,
    )

    # 3. Write test_evaluation.txt
    eval_path = final_dir / "test_evaluation.txt"
    _write_test_evaluation(
        path=eval_path, cfg=cfg, total_trials=total_trials,
        best_quality=best_quality, best_metric_raw=best_metric_raw,
        best_metric_smoothed=best_metric_smoothed,
        test_metric=test_metric, test_loss=test_loss,
        ranking_metric=ranking_metric, quality_weights=quality_weights,
        smoothing_window=smoothing_window,
    )
    print(f"[final] Test evaluation -> {eval_path.name}")

    # 4. Generate model.py
    model_py_path = final_dir / "model.py"
    generate_model_py(
        path=model_py_path, cfg=cfg, hp=best_hp, data_info=data_info,
        total_trials=total_trials, best_quality=best_quality,
        best_metric_raw=best_metric_raw,
        best_metric_smoothed=best_metric_smoothed,
        test_metric=test_metric, test_loss=test_loss,
        smoothing_window=smoothing_window,
    )
    print(f"[final] Generated standalone code -> {model_py_path.name}")

    # 5. Plot final training curves (if matplotlib available)
    plot_path = final_dir / "final_training.png"
    try:
        _plot_final_training(plot_path, training_curves, cfg.task_type)
        print(f"[final] Final training plot -> {plot_path.name}")
    except Exception as exc:
        print(f"[final] [WARN] Could not generate plot: {exc}")

    # 6. Save model checkpoint (state_dict + metadata for generation)
    checkpoint_path = final_dir / "model_checkpoint.pt"
    try:
        _save_checkpoint(checkpoint_path, model, cfg, best_hp, data_info)
        print(f"[final] Model checkpoint -> {checkpoint_path.name}")
    except Exception as exc:
        print(f"[final] [WARN] Could not save checkpoint: {exc}")

    # 7. For language_modeling: generate lyrics on the test set
    if cfg.task_type == "language_modeling":
        # Skip generation if refit diverged - the model weights are NaN,
        # which would produce NaN logits and trigger CUDA assertions during
        # sampling. Saving the (broken) checkpoint is still useful for debug.
        if not math.isfinite(test_loss):
            print(f"[final] [WARN] Skipping lyrics generation: refit "
                  f"diverged (test_loss={test_loss}). The model's weights "
                  f"are not usable for generation.")
        else:
            try:
                _run_generation_for_lm(
                    model=model, cfg=cfg, data_info=data_info,
                    final_dir=final_dir, device=device,
                )
            except Exception as exc:
                print(f"[final] [WARN] Generation phase failed: {exc}")


def _save_checkpoint(path, model, cfg, hp, data_info):
    """Save model state + metadata needed to reload for generation."""
    import torch
    vocab = data_info.get("vocab")
    payload = {
        "state_dict": model.state_dict(),
        "architecture": cfg.architecture,
        "task_type": cfg.task_type,
        "hyperparameters": hp,
        "vocab_itos": vocab.itos if vocab else None,
        "vocab_stoi": vocab.stoi if vocab else None,
        "line_separator_token": cfg.line_separator_token,
        "midi_variant": cfg.midi_variant,
        "midi_dim": data_info.get("midi_dim", 0),
        "embedding_dim": data_info.get("embedding_dim"),
        "vocab_size": data_info.get("vocab_size"),
        "output_dim": data_info.get("output_dim"),
    }
    torch.save(payload, path)


def _run_generation_for_lm(model, cfg, data_info, final_dir, device):
    """Generate lyrics for the test songs using the trained model."""
    from core.generator import run_generation_for_test_set

    test_songs = data_info.get("test_songs", [])
    if not test_songs:
        print("[final] No test songs available - skipping generation.")
        return

    vocab = data_info["vocab"]
    midi_variant = data_info["midi_variant"]
    midi_dim = data_info["midi_dim"]
    line_sep = cfg.line_separator_token

    # Default initial words: 3 common starters if user didn't supply
    initial_words = cfg.initial_words or ["love", "the", "i"]

    # Sampling kwargs by strategy
    sampling_kwargs = {
        "temperature": cfg.sampling_temperature,
        "k": cfg.sampling_top_k,
        "p": cfg.sampling_top_p,
    }

    print(f"[final] Generating lyrics for {len(test_songs)} test songs "
          f"x {len(initial_words)} initial words "
          f"using {cfg.sampling_strategy} sampling...")

    run_generation_for_test_set(
        model=model, vocab=vocab, test_songs=test_songs,
        initial_words=initial_words,
        midi_variant=midi_variant, midi_dim=midi_dim,
        device=device, output_dir=final_dir,
        sampling_strategy=cfg.sampling_strategy,
        sampling_kwargs=sampling_kwargs,
        max_words=cfg.max_generated_words,
        line_separator_token=line_sep,
        run_probe=cfg.melody_probe,
    )

    # Decoding strategy comparison (Assignment 3 sec. 13) -
    # generate the same prompt with proportional / temperature / nucleus
    # so the user can directly compare diversity vs coherence.
    if cfg.run_decoding_comparison:
        from core.generator import run_decoding_comparison
        print(f"[final] Running decoding strategy comparison on first 2 test songs...")
        run_decoding_comparison(
            model=model, vocab=vocab, test_songs=test_songs,
            initial_word=initial_words[0],
            midi_variant=midi_variant, midi_dim=midi_dim,
            device=device, output_dir=final_dir,
            max_words=cfg.max_generated_words,
            line_separator_token=line_sep,
            num_songs_to_compare=2,
        )


def _refit_and_eval(model, cfg, hp, data_info, cm, device):
    """Train on Train+Val combined; evaluate on Test."""
    task_type = cfg.task_type
    batch_size = int(hp.get("batch_size", 64))
    num_workers = int(hp.get("num_workers", 0))

    # Build train+val combined loader.
    # Strategy: use the existing loader to get train+val, then merge them.
    train_loader, val_loader = _make_loaders_for_refit(
        data_info, hp, cfg, num_workers
    )

    combined_train_dataset = _combine_datasets(train_loader, val_loader)
    combined_loader = DataLoader(
        combined_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
    )

    # Build test loader from the saved Test_set.csv (local) or saved test ds
    test_loader = _build_test_loader(cfg, data_info, batch_size, num_workers, hp)

    # Refit (no early stopping - we use the same epochs as the trial)
    epochs = int(hp.get("epochs") or 30)
    loss_fn = _build_loss_function(hp, task_type, data_info)
    optimizer = _build_optimizer(model, hp)
    grad_clip = hp.get("gradient_clipping", None)
    if grad_clip == 0:
        grad_clip = None

    use_amp = bool(hp.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    grad_accum_steps = max(1, int(hp.get("gradient_accumulation_steps", 1)))
    cutmix_alpha = float(hp.get("cutmix", 0.0))
    mixup_alpha = float(hp.get("mixup", 0.0))
    num_classes = int(data_info.get("output_dim", 0))

    train_loss_curve, train_metric_curve = [], []
    for epoch in range(epochs):
        tr_loss, tr_metric = _run_epoch(
            model, combined_loader, optimizer, loss_fn, device,
            task_type, grad_clip, True,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp, scaler=scaler,
            cutmix_alpha=cutmix_alpha, mixup_alpha=mixup_alpha,
            num_classes=num_classes,
        )
        train_loss_curve.append(tr_loss)
        train_metric_curve.append(tr_metric)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"[final]   refit epoch {epoch+1}/{epochs}: "
                  f"loss={tr_loss:.4f} metric={tr_metric:.4f}")
        # Abort refit immediately if loss diverged - subsequent epochs would
        # produce more NaNs and the final test eval + lyrics generation
        # would fail with CUDA assertions on NaN logits. The user is better
        # served by an early stop with a clear warning than by a downstream
        # CUDA crash.
        if not math.isfinite(tr_loss):
            print(f"[final] [WARN] refit diverged at epoch {epoch+1} "
                  f"(loss={tr_loss}). Aborting refit. The best trial's "
                  f"hyperparameters do not produce a stable model on the "
                  f"combined Train+Val set; test evaluation and generation "
                  f"will be skipped.")
            return float("nan"), float("nan"), {
                "train_loss": train_loss_curve,
                "train_metric": train_metric_curve,
            }

    # Evaluate on Test_set
    if test_loader is None:
        print("[final] No test loader available - skipping test evaluation.")
        test_loss, test_metric = 0.0, 0.0
    else:
        test_loss, test_metric = _run_epoch(
            model, test_loader, optimizer, loss_fn, device,
            task_type, grad_clip, False,
            grad_accum_steps=1, use_amp=False, scaler=None,
        )
        print(f"[final] Test results: loss={test_loss:.4f} metric={test_metric:.4f}")

    return test_metric, test_loss, {
        "train_loss": train_loss_curve,
        "train_metric": train_metric_curve,
    }


def _make_loaders_for_refit(data_info, hp, cfg, num_workers):
    """Recreate train+val loaders using the data loader callback stored in cfg."""
    from data_loaders import load_data
    make_loaders, _ = load_data(cfg)
    batch_size = int(hp.get("batch_size", 64))

    # Default seq_len: 32 for LM, 128 for other sequence tasks
    default_seq = 32 if cfg.task_type == "language_modeling" else 128
    seq_len = int(hp.get("sequence_length", default_seq))

    train_loader, val_loader = make_loaders(
        batch_size=batch_size,
        val_split=0.2,
        seed=cfg.random_seed or 42,
        num_workers=num_workers,
        image_size=int(hp.get("image_size", 64)),
        augmentation=hp.get("data_augmentation", "none"),
        cutout=float(hp.get("cutout", 0.0)),
        seq_len=seq_len,
        text_augmentation=hp.get("text_augmentation", "none"),
        text_augmentation_prob=float(hp.get("text_augmentation_prob", 0.0)),
    )
    return train_loader, val_loader


def _combine_datasets(train_loader, val_loader):
    """Concatenate the underlying datasets of two DataLoaders."""
    return ConcatDataset([train_loader.dataset, val_loader.dataset])


def _build_test_loader(cfg, data_info, batch_size, num_workers, hp=None):
    """
    Build a DataLoader for the held-out Test_set.

    Local mode (tabular): use the saved Test_set.csv from the data split sub-folder.
    Imported mode (image): use the original test split if available, else the random
    test split that random_split produced earlier.
    Language modeling: build a LyricsDataset from the held-out test_songs that
    the lyrics_loader stored in data_info.

    `hp` is the best-trial's hyperparameters; needed for LM to honor the chosen
    sequence_length (which is a tunable hyperparameter, not part of data_info).
    """
    if cfg.task_type == "language_modeling":
        return _build_test_loader_lyrics(cfg, data_info, batch_size, num_workers, hp)
    if cfg.dataset_mode == "local" and data_info.get("data_type") == "tabular":
        return _build_test_loader_local_tabular(
            cfg, data_info, batch_size, num_workers
        )
    if cfg.dataset_mode == "imported" and data_info.get("data_type") == "image":
        return _build_test_loader_imported_image(
            cfg, data_info, batch_size, num_workers
        )
    # Generic fallback
    return _build_test_loader_generic(cfg, data_info, batch_size, num_workers)


def _build_test_loader_lyrics(cfg, data_info, batch_size, num_workers, hp=None):
    """
    Build a test DataLoader for language_modeling using the test_songs that
    lyrics_loader already stored in data_info. We reuse LyricsDataset with the
    same sequence_length and midi_variant that the run used.

    If data_info is somehow missing the held-out test_songs (e.g. the orchestrator
    re-instantiated data_info), we re-run the lyrics loader from cfg to recover.
    """
    from data_loaders.lyrics_loader import LyricsDataset
    from torch.utils.data import DataLoader

    test_songs = data_info.get("test_songs", [])
    vocab = data_info.get("vocab")
    midi_variant = data_info.get("midi_variant", cfg.midi_variant)
    # Sequence length: take from the trained model's hp (it was tuned),
    # not from data_info (which doesn't track it). Fall back to a sensible default.
    if hp is not None and "sequence_length" in hp:
        seq_len = int(hp["sequence_length"])
    else:
        seq_len = int(data_info.get("sequence_length", 32))

    # Recovery path: re-load if anything is missing
    if not test_songs or vocab is None:
        print("[final] [WARN] Re-loading lyrics dataset to recover test split...")
        from data_loaders.lyrics_loader import load_lyrics
        _, recovered = load_lyrics(cfg)
        test_songs = recovered.get("test_songs", [])
        vocab = recovered.get("vocab")
        if midi_variant is None:
            midi_variant = recovered.get("midi_variant", cfg.midi_variant)

    if not test_songs or vocab is None:
        print("[final] [WARN] Could not build LM test loader - returning None. "
              "Final-phase test evaluation will be skipped.")
        return None

    ds = LyricsDataset(test_songs, vocab, midi_variant, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, drop_last=False)


def _build_test_loader_local_tabular(cfg, data_info, batch_size, num_workers):
    import pandas as pd
    import numpy as np

    test_path = Path(data_info["test_set_path"])
    test_df = pd.read_csv(test_path)

    # Use _prepare_xy from local_loader for consistent encoding/normalization
    from data_loaders.local_loader import _prepare_xy, _tabular_stats
    feature_cols = data_info.get("feature_columns")
    classes = data_info.get("classes")
    classes_arr = np.array(classes) if classes else None

    _, X_test, y_test, _, _ = _prepare_xy(
        test_df, cfg, fit_stats=False,
        precomputed_feature_cols=feature_cols,
        precomputed_classes=classes_arr,
    )
    test_dataset = TensorDataset(torch.from_numpy(X_test),
                                  torch.from_numpy(y_test))
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers)


def _build_test_loader_imported_image(cfg, data_info, batch_size, num_workers):
    """Return the original test split for torchvision datasets."""
    from torchvision import transforms
    name = cfg.imported_dataset_name.lower()
    cache = Path.home() / ".cache" / "autotune_nn" / "torchvision"

    in_channels = int(data_info.get("input_channels", 3))
    base = [transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * in_channels,
                                 std=[0.5] * in_channels)]
    tf = transforms.Compose(base)

    from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
    cls_map = {"mnist": MNIST, "fashion_mnist": FashionMNIST,
               "cifar10": CIFAR10, "cifar100": CIFAR100}
    ds_class = cls_map.get(name)
    if ds_class is None:
        raise ValueError(f"Unknown imported dataset for test loader: {name}")
    test_ds = ds_class(root=str(cache), train=False, download=False, transform=tf)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers)


def _build_test_loader_generic(cfg, data_info, batch_size, num_workers):
    """
    Generic fallback for sklearn-style imported datasets (no built-in test split).
    Reproduces the same train/val/test split deterministically and returns the
    test partition.
    """
    from torch.utils.data import random_split, DataLoader, TensorDataset
    import numpy as np

    name = cfg.imported_dataset_name.lower()
    sklearn_loaders = {
        "iris": "load_iris",
        "wine": "load_wine",
        "breast_cancer": "load_breast_cancer",
        "digits": "load_digits",
    }
    if name not in sklearn_loaders:
        raise NotImplementedError(
            f"Test loader fallback for '{name}' is not implemented. "
            f"Supported sklearn datasets: {list(sklearn_loaders)}"
        )

    from sklearn import datasets as skds
    bunch = getattr(skds, sklearn_loaders[name])()
    X = bunch.data.astype(np.float32)
    y = bunch.target.astype(np.int64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))

    n = len(ds)
    n_val = int(n * cfg.val_pct)
    n_test = int(n * cfg.test_pct)
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(cfg.random_seed or 42)
    _, _, test_ds = random_split(ds, [n_train, n_val, n_test], generator=gen)

    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers)


def _write_test_evaluation(path, cfg, total_trials, best_quality,
                           best_metric_raw, best_metric_smoothed,
                           test_metric, test_loss, ranking_metric,
                           quality_weights, smoothing_window):
    metric_name = "accuracy" if cfg.task_type == "classification" else "-RMSE"
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(" Final Test Evaluation \n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Architecture     : {cfg.architecture}\n")
        f.write(f"Total trials     : {total_trials}\n")
        f.write(f"Ranking metric   : {ranking_metric}\n")
        f.write(f"Smoothing window : {smoothing_window} epochs\n\n")
        f.write(f"Quality Score    : {best_quality:.4f} (composite, see model.py)\n\n")
        f.write(f"Val {metric_name} (smoothed) : {best_metric_smoothed:.4f}\n")
        f.write(f"Val {metric_name} (raw peak) : {best_metric_raw:.4f}\n\n")
        f.write(f"Test {metric_name}           : {test_metric:.4f}\n")
        f.write(f"Test loss               : {test_loss:.4f}\n\n")
        f.write("Quality Score components and weights:\n")
        for k, v in quality_weights.items():
            f.write(f"  {k:<25}: weight {v:.0%}\n")


def _plot_final_training(path, curves, task_type):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = list(range(1, len(curves["train_loss"]) + 1))
    ax_loss.plot(epochs, curves["train_loss"], label="Train+Val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Final Refit - Loss")
    ax_loss.grid(True, alpha=0.3)

    metric_label = "Accuracy" if task_type == "classification" else "-RMSE"
    ax_metric.plot(epochs, curves["train_metric"], label=f"Train+Val {metric_label}",
                   color="green")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel(metric_label)
    ax_metric.set_title(f"Final Refit - {metric_label}")
    ax_metric.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
