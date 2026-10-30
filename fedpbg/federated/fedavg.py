"""
FedAvg global training phase.

Trains a shared global model before per-client personalization oracle search.
This phase does NOT implement the FedPBG scheduler — only standard FedAvg.
"""

from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def _state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def _load_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state)


def fedavg_aggregate(states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    """Weighted average of client state dicts."""
    total = sum(weights)
    agg = {}
    for key in states[0]:
        agg[key] = sum(s[key] * w for s, w in zip(states, weights)) / total
    return agg


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / max(n, 1)


def run_fedavg(
    global_model: nn.Module,
    dataset,
    client_train_indices: Dict[int, List[int] | torch.Tensor],
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    client_fraction: float = 0.5,
    random_seed: int = 42,
) -> nn.Module:
    """
    Run FedAvg to produce a global model.

    All blocks are trainable during FedAvg (full model aggregation).
    """
    import numpy as np

    rng = np.random.default_rng(random_seed)
    client_ids = sorted(client_train_indices.keys())
    criterion = nn.CrossEntropyLoss()

    for _round in range(num_rounds):
        n_sample = max(1, int(len(client_ids) * client_fraction))
        selected = rng.choice(client_ids, size=n_sample, replace=False)

        local_states = []
        weights = []
        global_state = _state_dict_cpu(global_model)

        for cid in selected:
            local_model = copy.deepcopy(global_model)
            _load_state(local_model, global_state)
            local_model.to(device)

            idx = client_train_indices[cid]
            if hasattr(idx, "tolist"):
                idx = idx.tolist()
            loader = DataLoader(
                Subset(dataset, idx),
                batch_size=batch_size,
                shuffle=True,
            )
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, local_model.parameters()),
                lr=learning_rate,
            )
            for _ in range(local_epochs):
                train_one_epoch(local_model, loader, optimizer, device, criterion)

            local_states.append(_state_dict_cpu(local_model))
            weights.append(len(idx))

        new_state = fedavg_aggregate(local_states, weights)
        _load_state(global_model, new_state)

    return global_model
