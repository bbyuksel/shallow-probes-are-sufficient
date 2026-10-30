"""
Dirichlet-based non-IID partition of data across federated clients.

Each client receives its own train/validation/test splits so local evaluation
is isolated and oracle personalization can be computed per client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class ClientData:
    """Container for one client's data splits."""

    client_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    random_seed: int = 42,
) -> Dict[int, np.ndarray]:
    """
    Partition sample indices across clients using a Dirichlet non-IID split.

    For each class k, draw client proportions ~ Dir(alpha) and assign class-k
    samples accordingly. Lower alpha => stronger label skew per client.
    """
    rng = np.random.default_rng(random_seed)
    num_classes = int(labels.max()) + 1
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        if len(cls_idx) == 0:
            continue
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        # Ensure at least one sample per non-empty proportion bucket when possible
        split_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        chunks = np.split(cls_idx, split_points)
        for client_id, chunk in enumerate(chunks):
            if len(chunk) > 0:
                client_indices[client_id].extend(chunk.tolist())

    return {cid: np.array(idxs, dtype=np.int64) for cid, idxs in client_indices.items()}


def rebalance_empty_clients(
    partition: Dict[int, np.ndarray],
    num_clients: int,
    all_indices: np.ndarray,
    random_seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Ensure every client has at least one sample by redistributing from large clients."""
    rng = np.random.default_rng(random_seed)
    for cid in range(num_clients):
        if cid not in partition:
            partition[cid] = np.array([], dtype=np.int64)

    for cid in range(num_clients):
        if len(partition[cid]) == 0:
            donors = [d for d in range(num_clients) if len(partition[d]) > 3]
            if not donors:
                # fallback: assign random unused indices
                used = set(np.concatenate(list(partition.values()))) if partition else set()
                unused = [i for i in all_indices if i not in used]
                if unused:
                    partition[cid] = np.array([unused[0]], dtype=np.int64)
                continue
            donor = donors[int(rng.integers(0, len(donors)))]
            give = partition[donor][-1]
            partition[donor] = partition[donor][:-1]
            partition[cid] = np.array([give], dtype=np.int64)

    return partition


def client_train_val_test_split(
    indices: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split client indices into train, validation, and test sets."""
    indices = np.asarray(indices, dtype=np.int64)
    n = len(indices)
    if n == 0:
        return indices, indices, indices
    if n == 1:
        return indices, indices, indices
    if n == 2:
        return indices[:1], indices[:1], indices[1:2]
    if n < 10:
        # Small clients: ensure non-empty splits without sklearn edge cases
        rng = np.random.default_rng(random_seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(round(n * (1.0 - train_ratio - val_ratio))))
        n_val = max(1, int(round(n * val_ratio)))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            n_val = max(1, (n - n_train) // 2)
            n_test = n - n_train - n_val
        test_idx = shuffled[:n_test]
        val_idx = shuffled[n_test : n_test + n_val]
        train_idx = shuffled[n_test + n_val :]
        return train_idx, val_idx, test_idx

    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, random_state=random_seed, shuffle=True
    )
    val_rel = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_rel, random_state=random_seed, shuffle=True
    )
    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )


def build_client_splits(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> List[ClientData]:
    """Create federated client partitions with per-client train/val/test splits."""
    partition = dirichlet_partition(labels, num_clients, alpha, random_seed)
    all_idx = np.arange(len(labels))
    partition = rebalance_empty_clients(partition, num_clients, all_idx, random_seed)
    clients: List[ClientData] = []

    for client_id in range(num_clients):
        train_idx, val_idx, test_idx = client_train_val_test_split(
            partition[client_id], train_ratio, val_ratio, random_seed + client_id
        )
        clients.append(
            ClientData(
                client_id=client_id,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )
        )
    return clients
