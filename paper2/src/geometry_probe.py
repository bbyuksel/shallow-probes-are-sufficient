"""Lightweight geometry feature extraction via global FedAvg probe (no P0–P5 oracle rerun)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from fedpbg.config import (
    BATCH_SIZE,
    CLIENT_TRAIN_RATIO,
    CLIENT_VAL_RATIO,
    GLOBAL_ROUNDS,
    LEARNING_RATE,
    LOCAL_EPOCHS,
)
from fedpbg.data.federated_split import build_client_splits
from fedpbg.data.registry import load_registered_dataset
from fedpbg.federated.fedavg import _load_state, _state_dict_cpu, run_fedavg, train_one_epoch
from fedpbg.metrics.heterogeneity import compute_client_embeddings, heterogeneity_scores
from fedpbg.models.factory import build_model
from paper2.src.client_data_features import compute_client_data_features


def resolve_dataset_name(dataset: str) -> str:
    if dataset == "cifar10_fallback":
        return "femnist"
    return dataset


def _flat_grad_norm(model: nn.Module) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
    return float(np.sqrt(sq))


def _state_delta_norm(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    sq = 0.0
    for k in a:
        d = (a[k] - b[k]).float()
        sq += float(d.pow(2).sum().item())
    return float(np.sqrt(sq))


def _cosine_dict(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k in a:
        va = a[k].float().reshape(-1)
        vb = b[k].float().reshape(-1)
        dot += float((va * vb).sum().item())
        na += float(va.pow(2).sum().item())
        nb += float(vb.pow(2).sum().item())
    if na <= 0 or nb <= 0:
        return 0.0
    return float(dot / (np.sqrt(na) * np.sqrt(nb)))


@torch.no_grad()
def _mean_entropy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    entropies = []
    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        ent = -(probs * (probs + 1e-12).log()).sum(dim=1)
        entropies.append(ent.cpu().numpy())
    if not entropies:
        return 0.0
    return float(np.concatenate(entropies).mean())


@torch.no_grad()
def _mean_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += float(criterion(model(x), y).item())
        n += len(y)
    return total / max(n, 1)


def _prototype_distance(
    model: nn.Module,
    dataset,
    labels: np.ndarray,
    client_train_idx: np.ndarray,
    global_train_indices: Dict[int, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> float:
    """Mean L2 distance between client and global class prototype embeddings."""
    all_global = np.concatenate([global_train_indices[c] for c in global_train_indices])
    client_classes = np.unique(labels[client_train_idx])
    dists = []
    for cls in client_classes:
        g_idx = all_global[labels[all_global] == cls]
        c_idx = client_train_idx[labels[client_train_idx] == cls]
        if len(g_idx) == 0 or len(c_idx) == 0:
            continue
        mu_g = compute_client_embeddings(model, dataset, g_idx, device, batch_size)
        mu_c = compute_client_embeddings(model, dataset, c_idx, device, batch_size)
        dists.append(float(np.linalg.norm(mu_c - mu_g)))
    return float(np.mean(dists)) if dists else 0.0


def _client_grad_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 3,
) -> Tuple[float, float, Dict[str, torch.Tensor]]:
    """Gradient norm, batch-gradient variance, and last-batch grad dict."""
    criterion = nn.CrossEntropyLoss()
    batch_norms = []
    last_grad: Dict[str, torch.Tensor] = {}

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        loss.backward()
        batch_norms.append(_flat_grad_norm(model))
        last_grad = {k: v.detach().clone() for k, v in model.named_parameters() if v.grad is not None}

    if not batch_norms:
        return 0.0, 0.0, {}
    return float(np.mean(batch_norms)), float(np.var(batch_norms)), last_grad


def compute_geometry_for_config(
    dataset_name: str,
    seed: int,
    alpha: float,
    num_clients: int,
    data_root: Path,
    device: torch.device,
    global_rounds: int = GLOBAL_ROUNDS,
    local_epochs: int = LOCAL_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    max_records: Optional[int] = None,
    grad_only: bool = False,
    max_grad_batches: int = 3,
) -> pd.DataFrame:
    load_name = resolve_dataset_name(dataset_name)
    spec, dataset, labels, resolved = load_registered_dataset(
        load_name, data_root, max_records=max_records
    )
    clients = build_client_splits(
        labels, num_clients, alpha, CLIENT_TRAIN_RATIO, CLIENT_VAL_RATIO, seed
    )
    train_indices = {c.client_id: c.train_idx for c in clients}

    model = build_model(spec.modality, spec.in_channels, spec.num_classes, spec.input_size).to(device)
    global_model = run_fedavg(
        model,
        dataset,
        train_indices,
        num_rounds=global_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        random_seed=seed,
    )

    het_map = heterogeneity_scores(global_model, dataset, train_indices, device, batch_size) if not grad_only else {}
    global_state = _state_dict_cpu(global_model)

    pooled_idx = np.concatenate([train_indices[c] for c in train_indices])[: batch_size * 4]
    pooled_loader = DataLoader(Subset(dataset, pooled_idx.tolist()), batch_size=batch_size, shuffle=True)
    proxy_model = copy.deepcopy(global_model)
    _, _, global_grad = _client_grad_stats(proxy_model, pooled_loader, device, max_batches=2)

    client_data = {r["client_id"]: r for r in compute_client_data_features(labels, num_clients, alpha, seed)}

    rows = []
    for client in clients:
        cid = client.client_id
        train_loader = DataLoader(
            Subset(dataset, client.train_idx.tolist()), batch_size=batch_size, shuffle=True
        )
        eval_loader = DataLoader(
            Subset(dataset, client.train_idx.tolist()), batch_size=batch_size, shuffle=False
        )

        grad_model = copy.deepcopy(global_model)
        grad_norm, grad_var, client_grad = _client_grad_stats(
            grad_model, train_loader, device, max_batches=max_grad_batches
        )

        local_model = copy.deepcopy(global_model)
        _load_state(local_model, global_state)
        local_model.to(device)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, local_model.parameters()), lr=learning_rate)
        crit = nn.CrossEntropyLoss()
        train_one_epoch(local_model, train_loader, opt, device, crit)
        local_state = _state_dict_cpu(local_model)
        drift = _state_delta_norm(local_state, global_state)

        cos_sim = _cosine_dict(client_grad, global_grad) if client_grad and global_grad else np.nan
        h = het_map.get(cid, np.nan)
        rep_dist = float(np.sqrt(h)) if not np.isnan(h) else np.nan
        proto_dist = (
            _prototype_distance(
                global_model, dataset, labels, client.train_idx, train_indices, device, batch_size
            )
            if not grad_only
            else rep_dist
        )

        row = {
            "config_id": f"{dataset_name}_s{seed}_a{alpha}_n{num_clients}",
            "dataset": dataset_name,
            "seed": seed,
            "alpha": alpha,
            "num_clients": num_clients,
            "client_id": cid,
            "heterogeneity": h,
            "representation_distance": rep_dist,
            "prototype_distance": proto_dist,
            "client_drift_score": drift,
            "gradient_norm": grad_norm,
            "gradient_variance": grad_var,
            "cosine_to_global_update": cos_sim,
            "loss_on_global": _mean_loss(global_model, eval_loader, device),
            "mean_predictive_entropy": _mean_entropy(global_model, eval_loader, device),
        }
        row.update(client_data[cid])
        rows.append(row)

    return pd.DataFrame(rows)
