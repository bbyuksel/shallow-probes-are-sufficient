"""
Single-experiment oracle evaluation (one dataset × seed × alpha × num_clients).
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import torch

from fedpbg.config import (
    CLIENT_TRAIN_RATIO,
    CLIENT_VAL_RATIO,
    PERSONALIZATION_LEVELS,
)
from fedpbg.data.federated_split import build_client_splits
from fedpbg.data.registry import DatasetSpec
from fedpbg.federated.client_trainer import train_and_evaluate_level
from fedpbg.federated.fedavg import run_fedavg
from fedpbg.hardware.profiles import HARDWARE_PROFILES, assign_hardware_profiles
from fedpbg.metrics.cost import build_cost_record, normalize_costs_across_levels
from fedpbg.metrics.heterogeneity import heterogeneity_scores
from fedpbg.models.factory import build_model, estimate_flops


def run_oracle_client_eval(
    spec: DatasetSpec,
    dataset,
    labels,
    num_clients: int,
    dirichlet_alpha: float,
    seed: int,
    global_rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> pd.DataFrame:
    """
    Train global model and evaluate every (client, personalization level).

    Returns client-level DataFrame with metrics and costs (lambda-independent).
    """
    clients = build_client_splits(
        labels,
        num_clients=num_clients,
        alpha=dirichlet_alpha,
        train_ratio=CLIENT_TRAIN_RATIO,
        val_ratio=CLIENT_VAL_RATIO,
        random_seed=seed,
    )
    hardware_assignments = assign_hardware_profiles(num_clients, seed)

    model = build_model(
        spec.modality,
        spec.in_channels,
        spec.num_classes,
        spec.input_size,
    ).to(device)

    train_indices = {c.client_id: c.train_idx for c in clients}
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

    het_map = heterogeneity_scores(
        global_model,
        dataset,
        {c.client_id: c.train_idx for c in clients},
        device,
        batch_size=batch_size,
    )
    flops_est = estimate_flops(global_model, spec.modality, spec.input_size)

    rows: List[Dict] = []
    for client in clients:
        cid = client.client_id
        hw_type = hardware_assignments[cid]
        hw_profile = HARDWARE_PROFILES[hw_type]

        level_costs_raw = []
        level_results = []

        for level in PERSONALIZATION_LEVELS:
            metrics, cost_info = train_and_evaluate_level(
                global_model=global_model,
                level=level,
                dataset=dataset,
                train_idx=client.train_idx,
                val_idx=client.val_idx,
                test_idx=client.test_idx,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                flops_estimate=flops_est,
                memory_scale=spec.input_size,
            )
            raw = build_cost_record(
                trainable_params=cost_info["trainable_params"],
                flops=cost_info["flops"],
                train_time_sec=cost_info["train_time_sec"],
                comm_bytes=cost_info["comm_bytes"],
                batch_size=batch_size,
                signal_length=cost_info["memory_scale"],
                hardware=hw_profile,
                normalized_cost=0.0,
            )
            level_costs_raw.append(raw.raw_cost)
            level_results.append((level, metrics, raw))

        norm_costs = normalize_costs_across_levels(level_costs_raw)
        for (level, metrics, raw), norm_c in zip(level_results, norm_costs):
            rows.append(
                {
                    "client_id": cid,
                    "level": level,
                    "hardware": hw_type.value,
                    "heterogeneity": het_map[cid],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "auroc": metrics["auroc"],
                    "trainable_params": raw.trainable_params,
                    "flops": raw.flops,
                    "train_time_sec": raw.train_time_sec,
                    "comm_bytes": raw.comm_bytes,
                    "memory_proxy_mb": raw.memory_proxy_mb,
                    "hardware_multiplier": raw.hardware_multiplier,
                    "raw_cost": raw.raw_cost,
                    "normalized_cost": norm_c,
                }
            )

    return pd.DataFrame(rows)
