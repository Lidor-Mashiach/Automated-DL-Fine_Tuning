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
    probe_set = ds_class(root=str(cache), train=True, download=True, transform=probe_tf)
    sample, _ = probe_set[0]
    _, h, _ = sample.shape

    data_info = {
        "input_channels": in_channels,
        "image_size": h,
        "output_dim": num_classes,
        "n_samples": len(probe_set),
        "task_type": "classification",
        "data_type": "image",
    }

    def make_loaders(batch_size: int, val_split: float = 0.2,
                     seed: int = 42, image_size: int = h,
                     augmentation: str = "none", **_):
        mean = [0.5] * in_channels
        std = [0.5] * in_channels
        base = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        if augmentation == "none":
            tf = transforms.Compose(base)
        elif augmentation == "light":
            tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(h, padding=2),
            ] + base)
        else:
            tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(h, padding=4),
                transforms.RandomRotation(10),
            ] + base)

        full = ds_class(root=str(cache), train=True, download=False, transform=tf)
        n = len(full)
        # Use cfg's val_pct and test_pct for consistency with the top-level choice
        n_val = int(n * cfg.val_pct)
        n_test = int(n * cfg.test_pct)
        n_train = n - n_val - n_test
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds, _test_ds = random_split(
            full, [n_train, n_val, n_test], generator=gen
        )
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
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
    data_info = {
        "input_dim": X.shape[1],
        "output_dim": int(len(np.unique(y))),
        "n_samples": len(dataset),
        "task_type": "classification",
        "data_type": "tabular",
        "feature_columns": list(getattr(bunch, "feature_names", [])) or None,
    }

    def make_loaders(batch_size: int, val_split: float = 0.2,
                     seed: int = 42, **_):
        n = len(dataset)
        n_val = int(n * cfg.val_pct)
        n_test = int(n * cfg.test_pct)
        n_train = n - n_val - n_test
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds, _test_ds = random_split(
            dataset, [n_train, n_val, n_test], generator=gen
        )
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
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
