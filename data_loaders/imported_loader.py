"""
Imported data loader
--------------------
Loads well-known datasets from external libraries:
  Image (torchvision):   mnist, fashion_mnist, cifar10, cifar100
  Tabular (sklearn):     iris, wine, breast_cancer, digits

Cached under ~/.cache/autotune_nn/ to avoid polluting the project's Data/.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class _CutoutTransform:
    """Mask a random square region of the input tensor with zeros.

    Works on tensors of shape (C, H, W) - applied AFTER ToTensor + Normalize.
    """

    def __init__(self, mask_size: int):
        self.mask_size = int(mask_size)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 3:
            return img
        c, h, w = img.shape
        if self.mask_size <= 0 or self.mask_size >= min(h, w):
            return img
        cy = torch.randint(0, h, (1,)).item()
        cx = torch.randint(0, w, (1,)).item()
        y1 = max(0, cy - self.mask_size // 2)
        y2 = min(h, cy + self.mask_size // 2)
        x1 = max(0, cx - self.mask_size // 2)
        x2 = min(w, cx + self.mask_size // 2)
        out = img.clone()
        out[:, y1:y2, x1:x2] = 0.0
        return out


_REGISTRY = {
    "mnist":         {"type": "image",   "fn": "_load_mnist"},
    "fashion_mnist": {"type": "image",   "fn": "_load_fashion_mnist"},
    "cifar10":       {"type": "image",   "fn": "_load_cifar10"},
    "cifar100":      {"type": "image",   "fn": "_load_cifar100"},
    "iris":          {"type": "tabular", "fn": "_load_iris"},
    "wine":          {"type": "tabular", "fn": "_load_wine"},
    "breast_cancer": {"type": "tabular", "fn": "_load_breast_cancer"},
    "digits":        {"type": "tabular", "fn": "_load_digits"},
}


def _cache_dir() -> Path:
    p = Path.home() / ".cache" / "autotune_nn"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_imported(cfg):
    name = cfg.imported_dataset_name.lower().strip()
    if name not in _REGISTRY:
        raise ValueError(
            f"Imported dataset '{name}' not supported. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return globals()[_REGISTRY[name]["fn"]](cfg)


# ============================================================= IMAGE (torchvision)

def _image_from_torchvision(ds_class, cfg, in_channels: int, num_classes: int):
    from torchvision import transforms

    cache = _cache_dir() / "torchvision"
    cache.mkdir(parents=True, exist_ok=True)

    probe_tf = transforms.ToTensor()
    probe_train = ds_class(root=str(cache), train=True, download=True, transform=probe_tf)
    probe_test = ds_class(root=str(cache), train=False, download=True, transform=probe_tf)
    sample, _ = probe_train[0]
    _, h, _ = sample.shape

    n_orig_train = len(probe_train)
    n_orig_test = len(probe_test)

    # Compute Train:Val ratio from cfg, ignore test_pct (built-in test is used)
    ratio_sum = cfg.train_pct + cfg.val_pct
    train_frac = cfg.train_pct / ratio_sum
    n_val = int(n_orig_train * (1.0 - train_frac))
    n_train = n_orig_train - n_val

    print(f"[data] {ds_class.__name__} has built-in train/test split.")
    print(f"[data]   Test_set:  {n_orig_test:,} samples (original test, not touched).")
    print(f"[data]   Train+Val: from original train ({n_orig_train:,}) "
          f"using TRAIN_PCT:VAL_PCT ratio.")
    print(f"[data]     Train_set: {n_train:,} ({train_frac*100:.0f}% of original train)")
    print(f"[data]     Val_set:   {n_val:,} ({(1-train_frac)*100:.0f}% of original train)")
    print(f"[data]   TEST_PCT was IGNORED (built-in test set used).")

    data_info = {
        "input_channels": in_channels,
        "image_size": h,
        "output_dim": num_classes,
        "n_samples": n_train,
        "task_type": "classification",
        "data_type": "image",
        "imbalance_ratio": 1.0,  # well-known datasets are balanced
    }

    def make_loaders(batch_size: int, val_split: float = 0.2,
                     seed: int = 42, image_size: int = h,
                     augmentation: str = "none", num_workers: int = 0,
                     cutout: float = 0.0, **_):
        mean = [0.5] * in_channels
        std = [0.5] * in_channels
        base = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

        # CutOut transform: masks a square region (cutout > 0 enables it)
        post_aug = []
        if cutout > 0:
            mask_size = int(h * float(cutout))
            mask_size = max(1, min(mask_size, h - 1))
            post_aug.append(_CutoutTransform(mask_size))

        if augmentation == "none":
            tf = transforms.Compose(base + post_aug)
        elif augmentation == "light":
            tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(h, padding=2),
            ] + base + post_aug)
        else:
            tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(h, padding=4),
                transforms.RandomRotation(10),
            ] + base + post_aug)

        full_train = ds_class(root=str(cache), train=True, download=False, transform=tf)
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(
            full_train, [n_train, n_val], generator=gen
        )
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers),
        )

    return make_loaders, data_info


def _load_mnist(cfg):
    from torchvision.datasets import MNIST
    return _image_from_torchvision(MNIST, cfg, in_channels=1, num_classes=10)


def _load_fashion_mnist(cfg):
    from torchvision.datasets import FashionMNIST
    return _image_from_torchvision(FashionMNIST, cfg, in_channels=1, num_classes=10)


def _load_cifar10(cfg):
    from torchvision.datasets import CIFAR10
    return _image_from_torchvision(CIFAR10, cfg, in_channels=3, num_classes=10)


def _load_cifar100(cfg):
    from torchvision.datasets import CIFAR100
    return _image_from_torchvision(CIFAR100, cfg, in_channels=3, num_classes=100)


# ============================================================= TABULAR (sklearn)

def _tabular_from_sklearn(bunch, cfg):
    X = bunch.data.astype(np.float32)
    y = bunch.target.astype(np.int64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    # Compute imbalance ratio
    unique, counts = np.unique(y, return_counts=True)
    imbalance_ratio = float(counts.max()) / float(counts.min()) if counts.min() > 0 else 1.0
    print(f"[data] {bunch.__class__.__name__ if hasattr(bunch, '__class__') else 'sklearn'} "
          f"dataset has no built-in split.")
    print(f"[data]   Splitting using TRAIN_PCT/VAL_PCT/TEST_PCT (must sum to 1.0):")
    n_total = len(dataset)
    print(f"[data]     Train_set:  {int(n_total*cfg.train_pct):,} samples ({int(cfg.train_pct*100)}%)")
    print(f"[data]     Val_set:    {int(n_total*cfg.val_pct):,} samples ({int(cfg.val_pct*100)}%)")
    print(f"[data]     Test_set:   {int(n_total*cfg.test_pct):,} samples ({int(cfg.test_pct*100)}%)")
    if imbalance_ratio >= 3.0:
        print(f"[data] Imbalance ratio: {imbalance_ratio:.1f}:1")

    data_info = {
        "input_dim": X.shape[1],
        "output_dim": int(len(np.unique(y))),
        "n_samples": len(dataset),
        "task_type": "classification",
        "data_type": "tabular",
        "feature_columns": list(getattr(bunch, "feature_names", [])) or None,
        "imbalance_ratio": imbalance_ratio,
    }

    def make_loaders(batch_size: int, val_split: float = 0.2,
                     seed: int = 42, num_workers: int = 0, **_):
        n = len(dataset)
        n_val = int(n * cfg.val_pct)
        n_test = int(n * cfg.test_pct)
        n_train = n - n_val - n_test
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds, _test_ds = random_split(
            dataset, [n_train, n_val, n_test], generator=gen
        )
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers),
        )

    return make_loaders, data_info


def _load_iris(cfg):
    from sklearn.datasets import load_iris
    return _tabular_from_sklearn(load_iris(), cfg)


def _load_wine(cfg):
    from sklearn.datasets import load_wine
    return _tabular_from_sklearn(load_wine(), cfg)


def _load_breast_cancer(cfg):
    from sklearn.datasets import load_breast_cancer
    return _tabular_from_sklearn(load_breast_cancer(), cfg)


def _load_digits(cfg):
    from sklearn.datasets import load_digits
    return _tabular_from_sklearn(load_digits(), cfg)
