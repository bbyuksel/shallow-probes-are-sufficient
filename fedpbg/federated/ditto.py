"""
Ditto — personalized federated learning with proximal regularization.

Reference: Li et al., "Ditto: Fair and Robust Federated Learning Through
Personalization", ICML 2021.

Each client maintains a personalized model v_i trained with
    min_v  F_i(v) + (mu/2) ||v - w||^2
where w is the current global model. The global model w is updated via FedAvg.
Inference uses v_i (personalized model).
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fedpbg.federated.fedavg import (
    _load_state,
    _state_dict_cpu,
    fedavg_aggregate,
    train_one_epoch,
)


def _proximal_penalty(
    model: nn.Module,
    global_state: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """(1/2) * ||v - w||^2 summed over all parameters."""
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name in global_state:
            diff = param - global_state[name].to(param.device)
            penalty = penalty + (diff ** 2).sum()
    return 0.5 * penalty


def train_ditto_personal_epoch(
    model: nn.Module,
    global_state: Dict[str, torch.Tensor],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    proximal_mu: float,
) -> float:
    """One epoch: CE loss + (mu/2)||v - w||^2."""
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if proximal_mu > 0:
            loss = loss + proximal_mu * _proximal_penalty(model, global_state)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / max(n, 1)


def run_ditto(
    model: nn.Module,
    dataset,
    client_train_indices: Dict[int, List[int] | torch.Tensor],
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    client_fraction: float = 0.5,
    random_seed: int = 42,
    proximal_mu: float = 0.1,
) -> Dict[int, nn.Module]:
    """
    Run Ditto federated training.

    Returns personalized models v_i per client_id (for test evaluation).
    """
    import numpy as np

    rng = np.random.default_rng(random_seed)
    client_ids = sorted(client_train_indices.keys())
    criterion = nn.CrossEntropyLoss()

    global_model = copy.deepcopy(model).to(device)
    personalized: Dict[int, nn.Module] = {
        cid: copy.deepcopy(model).to(device) for cid in client_ids
    }

    for _round in range(num_rounds):
        n_sample = max(1, int(len(client_ids) * client_fraction))
        selected = list(rng.choice(client_ids, size=n_sample, replace=False))

        global_state = _state_dict_cpu(global_model)
        local_global_states: List[Dict[str, torch.Tensor]] = []
        weights: List[float] = []

        for cid in selected:
            idx = client_train_indices[cid]
            if hasattr(idx, "tolist"):
                idx = idx.tolist()
            loader = DataLoader(
                Subset(dataset, idx),
                batch_size=batch_size,
                shuffle=True,
            )

            # --- Global branch (FedAvg local update on w) ---
            local_global = copy.deepcopy(global_model)
            _load_state(local_global, global_state)
            local_global.to(device)
            opt_global = torch.optim.Adam(
                filter(lambda p: p.requires_grad, local_global.parameters()),
                lr=learning_rate,
            )
            for _ in range(local_epochs):
                train_one_epoch(local_global, loader, opt_global, device, criterion)
            local_global_states.append(_state_dict_cpu(local_global))
            weights.append(len(idx))

            # --- Personalized branch (v_i with proximal to updated local w) ---
            local_w_state = _state_dict_cpu(local_global)
            v_i = personalized[cid]
            _load_state(v_i, _state_dict_cpu(v_i))  # continue from v_i^{t}
            v_i.to(device)
            opt_personal = torch.optim.Adam(
                filter(lambda p: p.requires_grad, v_i.parameters()),
                lr=learning_rate,
            )
            for _ in range(local_epochs):
                train_ditto_personal_epoch(
                    v_i,
                    local_w_state,
                    loader,
                    opt_personal,
                    device,
                    criterion,
                    proximal_mu,
                )
        new_global = fedavg_aggregate(local_global_states, weights)
        _load_state(global_model, new_global)

    return personalized


def evaluate_ditto_clients(
    personalized_models: Dict[int, nn.Module],
    dataset,
    client_test_indices: Dict[int, torch.Tensor],
    batch_size: int,
    device: torch.device,
    flops_estimate: int,
    memory_scale: int,
    train_times: Dict[int, float] | None = None,
) -> Dict[int, Dict]:
    """
    Evaluate each client's personalized model on its test split.

    Returns per-client metrics and cost_info (full-model / P5-equivalent).
    """
    from fedpbg.metrics.evaluation import evaluate_model
    from fedpbg.models.personalization import apply_personalization_level, communication_bytes, count_trainable_params

    results = {}
    for cid, model in personalized_models.items():
        m = copy.deepcopy(model)
        apply_personalization_level(m, "P5")
        m.to(device)
        test_loader = DataLoader(
            Subset(dataset, client_test_indices[cid].tolist()),
            batch_size=batch_size,
            shuffle=False,
        )
        metrics = evaluate_model(m, test_loader, device)
        trainable = count_trainable_params(m)
        comm = communication_bytes(m)
        train_time = train_times.get(cid, 0.0) if train_times else 0.0
        results[cid] = {
            "metrics": metrics,
            "cost_info": {
                "trainable_params": trainable,
                "flops": flops_estimate,
                "train_time_sec": train_time,
                "comm_bytes": comm,
                "memory_scale": memory_scale,
            },
        }
    return results
