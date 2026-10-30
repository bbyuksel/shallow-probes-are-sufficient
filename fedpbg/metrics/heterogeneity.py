"""
Prototype-based client heterogeneity score.

H_i = || mu_i - mu_g ||^2

where mu_i is the mean embedding of client i using the global encoder and
mu_g is the sample-size-weighted global mean embedding.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def compute_client_embeddings(
    model: nn.Module,
    dataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Return mean embedding vector for a client's samples."""
    model.eval()
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
    )
    embs: List[np.ndarray] = []
    for x, _ in loader:
        x = x.to(device)
        emb = model.encode(x).cpu().numpy()
        embs.append(emb)
    if not embs:
        return np.zeros(model.head.in_features, dtype=np.float32)
    return np.concatenate(embs, axis=0).mean(axis=0)


def heterogeneity_scores(
    model: nn.Module,
    dataset,
    client_indices: Dict[int, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> Dict[int, float]:
    """
    Compute H_i for each client.

    client_indices maps client_id -> array of sample indices (typically train set)
    """
    mus: Dict[int, np.ndarray] = {}
    sizes: Dict[int, int] = {}

    for cid, idx in client_indices.items():
        mus[cid] = compute_client_embeddings(model, dataset, idx, device, batch_size)
        sizes[cid] = len(idx)

    total = sum(sizes.values())
    mu_g = sum(mus[cid] * sizes[cid] for cid in mus) / max(total, 1)

    return {cid: float(np.sum((mus[cid] - mu_g) ** 2)) for cid in mus}
