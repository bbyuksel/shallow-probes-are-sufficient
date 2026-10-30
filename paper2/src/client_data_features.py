"""Client-data label statistics from deterministic federated splits (no FL training)."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from fedpbg.config import CLIENT_TRAIN_RATIO, CLIENT_VAL_RATIO
from fedpbg.data.federated_split import build_client_splits


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def label_stats_for_client(labels: np.ndarray, train_idx: np.ndarray) -> Dict[str, float]:
    y = labels[train_idx]
    n = len(y)
    if n == 0:
        return {
            "num_samples": 0,
            "label_entropy": 0.0,
            "majority_class_ratio": 0.0,
            "class_count": 0,
            "class_balance_score": 0.0,
        }
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    majority = float(counts.max() / counts.sum())
    ent = _entropy(probs)
    max_ent = math.log(len(classes)) if len(classes) > 1 else 1.0
    balance = float(ent / max_ent) if max_ent > 0 else 1.0
    return {
        "num_samples": int(n),
        "label_entropy": ent,
        "majority_class_ratio": majority,
        "class_count": int(len(classes)),
        "class_balance_score": balance,
    }


def compute_client_data_features(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[Dict]:
    clients = build_client_splits(
        labels,
        num_clients=num_clients,
        alpha=alpha,
        train_ratio=CLIENT_TRAIN_RATIO,
        val_ratio=CLIENT_VAL_RATIO,
        random_seed=seed,
    )
    rows = []
    for c in clients:
        stats = label_stats_for_client(labels, c.train_idx)
        stats["client_id"] = c.client_id
        rows.append(stats)
    return rows
