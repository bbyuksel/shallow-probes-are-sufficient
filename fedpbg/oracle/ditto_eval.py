"""
Ditto evaluation under Paper-1 utility U = MacroF1 - lambda * Cost.

Runs Ditto training per configuration, scores clients under U using P5-normalized
cost from the oracle cache (Ditto personalizes the full model).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

from fedpbg.config import (
    BATCH_SIZE,
    CLIENT_TRAIN_RATIO,
    CLIENT_VAL_RATIO,
    GLOBAL_ROUNDS,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    PERSONALIZATION_LEVELS,
)
from fedpbg.data.federated_split import build_client_splits
from fedpbg.data.registry import load_registered_dataset
from fedpbg.federated.ditto import evaluate_ditto_clients, run_ditto
from fedpbg.hardware.profiles import HARDWARE_PROFILES, assign_hardware_profiles
from fedpbg.metrics.cost import build_cost_record, normalize_costs_across_levels
from fedpbg.metrics.utility import compute_utility
from fedpbg.models.factory import build_model, estimate_flops
from fedpbg.oracle.policy_eval import evaluate_policy

# Standard Ditto proximal coefficient (Li et al. ICML 2021; not utility lambda)
DITTO_PROXIMAL_MU = 0.1

LAMBDAS_REGRET = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]


def _p5_normalized_cost(oracle_cache_df: pd.DataFrame, client_id: int) -> float:
    row = oracle_cache_df[
        (oracle_cache_df["client_id"] == client_id) & (oracle_cache_df["level"] == "P5")
    ]
    if row.empty:
        raise ValueError(f"No P5 row for client {client_id}")
    return float(row.iloc[0]["normalized_cost"])


def run_ditto_for_config(
    dataset_name: str,
    seed: int,
    dirichlet_alpha: float,
    num_clients: int,
    device: torch.device,
    data_root: Path,
    max_records: Optional[int] = 5000,
    oracle_cache_df: Optional[pd.DataFrame] = None,
    preloaded: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Train Ditto and return per-client results with macro_f1 and normalized_cost.
    """
    if preloaded is not None:
        spec, dataset, labels, resolved = preloaded
    else:
        spec, dataset, labels, resolved = load_registered_dataset(
            dataset_name,
            data_root,
            max_records=max_records if dataset_name == "ptbxl" else None,
        )
    if resolved not in ("cifar10", "cifar100", "ptbxl"):
        raise ValueError(f"Expected cifar10/cifar100/ptbxl, got {resolved}")

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
    )
    train_indices = {c.client_id: c.train_idx for c in clients}
    test_indices = {c.client_id: c.test_idx for c in clients}

    t0 = time.perf_counter()
    personalized = run_ditto(
        model,
        dataset,
        train_indices,
        num_rounds=GLOBAL_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=device,
        client_fraction=0.5,
        random_seed=seed,
        proximal_mu=DITTO_PROXIMAL_MU,
    )
    total_train_time = time.perf_counter() - t0
    per_client_time = total_train_time / max(len(clients), 1)

    flops_est = estimate_flops(model, spec.modality, spec.input_size)
    eval_results = evaluate_ditto_clients(
        personalized,
        dataset,
        test_indices,
        BATCH_SIZE,
        device,
        flops_est,
        spec.input_size,
        train_times={c.client_id: per_client_time for c in clients},
    )

    rows: List[Dict] = []
    for client in clients:
        cid = client.client_id
        hw = hardware_assignments[cid]
        hw_profile = HARDWARE_PROFILES[hw]
        er = eval_results[cid]
        metrics = er["metrics"]
        ci = er["cost_info"]

        if oracle_cache_df is not None:
            norm_cost = _p5_normalized_cost(oracle_cache_df, cid)
        else:
            raw = build_cost_record(
                trainable_params=ci["trainable_params"],
                flops=ci["flops"],
                train_time_sec=ci["train_time_sec"],
                comm_bytes=ci["comm_bytes"],
                batch_size=BATCH_SIZE,
                signal_length=ci["memory_scale"],
                hardware=hw_profile,
                normalized_cost=1.0,
            )
            norm_cost = raw.raw_cost

        rows.append(
            {
                "client_id": cid,
                "hardware": hw.value,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "auroc": metrics["auroc"],
                "normalized_cost": norm_cost,
                "cost_level": "P5",
                "ditto_proximal_mu": DITTO_PROXIMAL_MU,
                "train_time_sec": ci["train_time_sec"],
                "trainable_params": ci["trainable_params"],
                "dataset": resolved,
                "dataset_requested": dataset_name,
                "seed": seed,
                "dirichlet_alpha": dirichlet_alpha,
                "num_clients": num_clients,
            }
        )

    return pd.DataFrame(rows)


def utilities_for_config(
    ditto_df: pd.DataFrame,
    oracle_cache_df: pd.DataFrame,
    lambda_: float,
    seed: int,
) -> Dict[str, Dict]:
    """Compute mean utilities for ORACLE, BEST_UNIFORM, GLOBAL_ONLY, DITTO."""
    oracle_res = evaluate_policy(oracle_cache_df, lambda_, "ORACLE", seed)
    uniform_res = evaluate_policy(oracle_cache_df, lambda_, "BEST_UNIFORM", seed)
    global_res = evaluate_policy(oracle_cache_df, lambda_, "GLOBAL_ONLY", seed)

    ditto_utils = []
    ditto_f1 = []
    ditto_cost = []
    for _, row in ditto_df.iterrows():
        u = compute_utility(row["macro_f1"], row["normalized_cost"], lambda_)
        ditto_utils.append(u)
        ditto_f1.append(row["macro_f1"])
        ditto_cost.append(row["normalized_cost"])

    import numpy as np

    arr = np.array(ditto_utils)
    return {
        "ORACLE": oracle_res,
        "BEST_UNIFORM": uniform_res,
        "GLOBAL_ONLY": global_res,
        "DITTO": {
            "policy": "DITTO",
            "lambda": lambda_,
            "mean_utility": float(arr.mean()),
            "std_utility": float(arr.std()),
            "mean_macro_f1": float(np.mean(ditto_f1)),
            "mean_cost": float(np.mean(ditto_cost)),
            "client_utilities": ditto_utils,
        },
    }
