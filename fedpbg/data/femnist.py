"""
FEMNIST loader (LEAF format) with automatic CIFAR-10 fallback.

LEAF FEMNIST JSON files should live under data/femnist/raw/train and .../test.
If unavailable, callers receive CIFAR-10 tensors with dataset_name='cifar10_fallback'.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _load_leaf_json_dir(json_dir: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Parse LEAF-style JSON shards into image/label lists."""
    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    if not json_dir.exists():
        return images, labels

    for json_path in sorted(json_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        user_data = data.get("user_data", {})
        for _user, content in user_data.items():
            xs = content["x"]
            ys = content["y"]
            # Each x is flattened 28x28 grayscale
            arr = np.array(xs, dtype=np.float32).reshape(-1, 1, 28, 28) / 255.0
            lbl = np.array(ys, dtype=np.int64)
            images.append(arr)
            labels.append(lbl)
    return images, labels


def femnist_available(data_dir: Path) -> bool:
    train_dir = data_dir / "raw" / "train"
    return train_dir.exists() and any(train_dir.glob("*.json"))


def load_femnist_arrays(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Load FEMNIST into (N, 1, 28, 28) and labels.

    Returns (X, y, used_fallback).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    train_imgs, train_lbls = _load_leaf_json_dir(data_dir / "raw" / "train")
    test_imgs, test_lbls = _load_leaf_json_dir(data_dir / "raw" / "test")

    if not train_imgs:
        return np.array([]), np.array([]), True

    X = np.concatenate(train_imgs + test_imgs, axis=0)
    y = np.concatenate(train_lbls + test_lbls, axis=0)
    # Per-sample standardization
    mean = X.mean(axis=(2, 3), keepdims=True)
    std = X.std(axis=(2, 3), keepdims=True) + 1e-8
    X = ((X - mean) / std).astype(np.float32)
    return X, y, False
