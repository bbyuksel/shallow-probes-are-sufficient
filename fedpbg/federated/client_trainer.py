"""
Local client training for a specific personalization level.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fedpbg.federated.fedavg import train_one_epoch
from fedpbg.metrics.evaluation import evaluate_model
from fedpbg.models.personalization import (
    apply_personalization_level,
    communication_bytes,
    count_trainable_params,
)


def train_and_evaluate_level(
    global_model: nn.Module,
    level: str,
    dataset,
    train_idx,
    val_idx,
    test_idx,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    flops_estimate: int,
    memory_scale: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Fine-tune (or evaluate only for P0) a copy of the global model at `level`.

    Returns
    -------
    metrics : test-set metrics (accuracy, macro_f1, auroc)
    cost_info : trainable_params, flops, train_time_sec, comm_bytes, memory_proxy inputs
    """
    model = copy.deepcopy(global_model)
    apply_personalization_level(model, level)
    model.to(device)

    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx.tolist()), batch_size=batch_size, shuffle=False)

    trainable = count_trainable_params(model)
    comm = communication_bytes(model)

    start = time.perf_counter()

    if level != "P0" and trainable > 0:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )
        for _ in range(local_epochs):
            train_one_epoch(model, train_loader, optimizer, device, criterion)
    else:
        # P0: global model only — no local updates
        pass

    train_time = time.perf_counter() - start

    # Use validation for monitoring; report test metrics for oracle
    _ = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)

    cost_info = {
        "trainable_params": trainable,
        "flops": flops_estimate,
        "train_time_sec": train_time,
        "comm_bytes": comm,
        "memory_scale": memory_scale,
        "batch_size": batch_size,
    }
    return test_metrics, cost_info
