"""Load labels only for client-data feature computation (no image memory)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fedpbg.data.cifar import load_cifar_labels
from fedpbg.data.femnist import femnist_available, load_femnist_arrays
from fedpbg.data.registry import DATASET_REGISTRY
from paper2.src.geometry_probe import resolve_dataset_name


def load_labels_for_config(dataset_name: str, data_root: Path) -> np.ndarray:
    load_name = resolve_dataset_name(dataset_name)
    if load_name in ("cifar10", "cifar100"):
        return load_cifar_labels(load_name, data_root / load_name)
    if load_name == "femnist":
        fem_dir = data_root / "femnist"
        if femnist_available(fem_dir):
            _, y, _ = load_femnist_arrays(fem_dir)
            return y
        return load_cifar_labels("cifar10", data_root / "cifar10")
    if load_name == "ptbxl":
        # PTB-XL: use registry but labels-only path via synthetic cap if needed
        spec = DATASET_REGISTRY["ptbxl"]
        _, y, _ = spec.loader(data_root, quick=False, max_records=5000)
        return y
    spec = DATASET_REGISTRY[load_name]
    _, y, _ = spec.loader(data_root)
    return y
