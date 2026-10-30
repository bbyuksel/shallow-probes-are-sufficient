"""Live FL session: FedAvg + per-client level evaluation with timing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from fedpbg.config import CLIENT_TRAIN_RATIO, CLIENT_VAL_RATIO, PERSONALIZATION_LEVELS
from fedpbg.data.federated_split import build_client_splits
from fedpbg.data.registry import DatasetSpec, load_registered_dataset
from fedpbg.federated.client_trainer import train_and_evaluate_level
from fedpbg.federated.fedavg import run_fedavg
from fedpbg.hardware.profiles import HARDWARE_PROFILES, assign_hardware_profiles
from fedpbg.metrics.cost import build_cost_record, normalize_costs_across_levels
from fedpbg.metrics.heterogeneity import heterogeneity_scores
from fedpbg.models.factory import build_model, estimate_flops

PROBE_LEVELS = ["P0", "P1", "P2"]


@dataclass
class OnlineSession:
    config_id: str
    dataset: str
    resolved_dataset: str
    seed: int
    alpha: float
    num_clients: int
    client_level_df: pd.DataFrame
    timing: Dict[str, float] = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)


def config_id(dataset: str, seed: int, alpha: float, num_clients: int) -> str:
    return f"{dataset}_s{seed}_a{alpha}_n{num_clients}"


def run_online_fl_session(
    dataset_name: str,
    seed: int,
    alpha: float,
    num_clients: int = 20,
    global_rounds: int = 5,
    local_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    data_root: Optional[Path] = None,
    quick: bool = False,
    max_records: Optional[int] = None,
) -> OnlineSession:
    """FedAvg then exhaustive P0–P5 per client (oracle reference + cost norms)."""
    from fedpbg.config import PROJECT_ROOT

    data_root = data_root or PROJECT_ROOT / "data"
    device = device or torch.device("cpu")

    spec, dataset, labels, resolved = load_registered_dataset(
        dataset_name, data_root, quick=quick, max_records=max_records
    )

    patient_ids = getattr(dataset, "patient_ids", None)
    patient_split_meta = None
    if patient_ids is not None and str(resolved).startswith("ptbxl"):
        from fedpbg.data.patient_disjoint import build_patient_disjoint_client_splits

        pd_result = build_patient_disjoint_client_splits(
            labels=labels,
            patient_ids=patient_ids,
            num_clients=num_clients,
            alpha=alpha,
            train_ratio=CLIENT_TRAIN_RATIO,
            val_ratio=CLIENT_VAL_RATIO,
            random_seed=seed,
        )
        clients = pd_result.clients
        patient_split_meta = {
            "protocol": pd_result.protocol,  # ptbxl_patient_disjoint_v2
            "split_report_rows": pd_result.split_report_rows,
            "n_patients": int(len(pd_result.patient_table)),
            "n_records": int(len(labels)),
            "global_split_patients": {
                k: [str(x) for x in v.tolist()] for k, v in pd_result.global_split_patients.items()
            },
            "client_patients": {
                str(cid): [str(x) for x in pids.tolist()]
                for cid, pids in pd_result.client_patients.items()
            },
        }
    else:
        clients = build_client_splits(
            labels,
            num_clients=num_clients,
            alpha=alpha,
            train_ratio=CLIENT_TRAIN_RATIO,
            val_ratio=CLIENT_VAL_RATIO,
            random_seed=seed,
        )
    hardware_assignments = assign_hardware_profiles(num_clients, seed)

    model = build_model(spec.modality, spec.in_channels, spec.num_classes, spec.input_size).to(device)

    t0 = time.perf_counter()
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
    fedavg_sec = time.perf_counter() - t0

    het_map = heterogeneity_scores(
        global_model,
        dataset,
        {c.client_id: c.train_idx for c in clients},
        device,
        batch_size=batch_size,
    )
    flops_est = estimate_flops(global_model, spec.modality, spec.input_size)

    rows: List[Dict] = []
    probe_sec = 0.0
    oracle_sec = 0.0

    for client in clients:
        cid = client.client_id
        hw_type = hardware_assignments[cid]
        hw_profile = HARDWARE_PROFILES[hw_type]

        level_costs_raw = []
        level_results = []

        for level in PERSONALIZATION_LEVELS:
            t_level = time.perf_counter()
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
            elapsed = time.perf_counter() - t_level
            oracle_sec += elapsed
            if level in PROBE_LEVELS:
                probe_sec += elapsed

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

    cid_str = config_id(resolved, seed, alpha, num_clients)
    df = pd.DataFrame(rows)
    df["dataset"] = resolved
    df["seed"] = seed
    df["dirichlet_alpha"] = alpha
    df["num_clients"] = num_clients

    timing = {
        "fedavg_sec": fedavg_sec,
        "probe_sec": probe_sec,
        "oracle_levels_sec": oracle_sec,
        "total_sec": fedavg_sec + oracle_sec,
        "n_probe_trainings": num_clients * len(PROBE_LEVELS),
        "n_oracle_trainings": num_clients * len(PERSONALIZATION_LEVELS),
    }

    return OnlineSession(
        config_id=cid_str,
        dataset=dataset_name,
        resolved_dataset=resolved,
        seed=seed,
        alpha=alpha,
        num_clients=num_clients,
        client_level_df=df,
        timing=timing,
        meta={
            "global_rounds": global_rounds,
            "local_epochs": local_epochs,
            "modality": spec.modality,
            "patient_split": patient_split_meta,
            "protocol_version": "sinconf2026_methodology_v2",
        },
    )


def save_session(session: OnlineSession, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{session.config_id}_client_level.csv"
    session.client_level_df.to_csv(path, index=False)
    timing_path = out_dir / f"{session.config_id}_timing.json"
    import json

    timing_path.write_text(json.dumps(session.timing, indent=2), encoding="utf-8")
    return path


def load_session(cache_path: Path, timing_path: Optional[Path] = None) -> OnlineSession:
    import json
    import re

    df = pd.read_csv(cache_path)
    m = re.match(r"^(?P<ds>.+)_s(?P<seed>\d+)_a(?P<alpha>[\d.]+)_n(?P<n>\d+)_client_level\.csv$", cache_path.name)
    if not m:
        raise ValueError(f"Cannot parse config from {cache_path.name}")
    timing = {}
    tp = timing_path or cache_path.with_name(cache_path.stem.replace("_client_level", "") + "_timing.json")
    if tp.exists():
        timing = json.loads(tp.read_text(encoding="utf-8"))
    return OnlineSession(
        config_id=cache_path.stem.replace("_client_level", ""),
        dataset=m.group("ds"),
        resolved_dataset=str(df["dataset"].iloc[0]) if "dataset" in df.columns else m.group("ds"),
        seed=int(m.group("seed")),
        alpha=float(m.group("alpha")),
        num_clients=int(m.group("n")),
        client_level_df=df,
        timing=timing,
    )
