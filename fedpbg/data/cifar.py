"""
CIFAR-10 / CIFAR-100 loading for federated oracle studies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as T


def load_cifar_labels(dataset_name: str, data_dir: Path) -> np.ndarray:
    """Load CIFAR train labels only (no image memory)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if dataset_name == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root=str(data_dir), train=True, download=True, transform=T.ToTensor()
        )
    elif dataset_name == "cifar100":
        ds = torchvision.datasets.CIFAR100(
            root=str(data_dir), train=True, download=True, transform=T.ToTensor()
        )
    else:
        raise ValueError(f"Unknown CIFAR dataset: {dataset_name}")
    return np.array(ds.targets, dtype=np.int64)


def load_cifar_arrays(
    dataset_name: str,
    data_dir: Path,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CIFAR train split into (N, 3, 32, 32) float32 and labels.

    Parameters
    ----------
    dataset_name : 'cifar10' or 'cifar100'
    """
    transform_list = [T.ToTensor()]
    if normalize:
        if dataset_name == "cifar10":
            transform_list.append(
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            )
        else:
            transform_list.append(
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            )
    transform = T.Compose(transform_list)

    data_dir.mkdir(parents=True, exist_ok=True)
    if dataset_name == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root=str(data_dir), train=True, download=True, transform=transform
        )
    elif dataset_name == "cifar100":
        ds = torchvision.datasets.CIFAR100(
            root=str(data_dir), train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown CIFAR dataset: {dataset_name}")

    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    images, labels = [], []
    for x, y in loader:
        images.append(x.numpy())
        labels.append(y.numpy())
    X = np.concatenate(images, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    return X, y
