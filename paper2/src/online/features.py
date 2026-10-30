"""Build FedPBA feature rows from live FL session outcomes."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from fedpbg.hardware.profiles import HARDWARE_PROFILES, HardwareType
from paper2.src.config import LEVELS
from paper2.src.online.session import OnlineSession, PROBE_LEVELS


def _hardware_score(hw: str) -> float:
    try:
        profile = HARDWARE_PROFILES[HardwareType(hw)]
    except (KeyError, ValueError):
        return 0.0
    weights = {"compute": 0.35, "memory": 0.20, "bandwidth": 0.20, "latency": 0.15, "energy": 0.10}
    return float(
        weights["compute"] * profile.compute
        + weights["memory"] * profile.memory
        + weights["bandwidth"] * profile.bandwidth
        + weights["latency"] * profile.latency
        + weights["energy"] * profile.energy
    )


def build_client_feature_rows(session: OnlineSession, lambda_: float) -> pd.DataFrame:
    """One row per client for allocator prediction (context_probes features)."""
    df = session.client_level_df
    rows: List[Dict] = []

    for cid, group in df.groupby("client_id"):
        base = group.iloc[0]
        row: Dict = {
            "config_id": session.config_id,
            "dataset": session.resolved_dataset,
            "alpha": session.alpha,
            "lambda": lambda_,
            "seed": session.seed,
            "num_clients": session.num_clients,
            "client_id": int(cid),
            "hardware_profile": str(base["hardware"]),
            "hardware_score": _hardware_score(str(base["hardware"])),
            "hardware_multiplier": float(base["hardware_multiplier"]),
            "heterogeneity": float(base["heterogeneity"]),
        }
        for lv in PROBE_LEVELS:
            key = lv.lower()
            lv_row = group[group["level"] == lv]
            if lv_row.empty:
                row[f"{key}_macro_f1"] = np.nan
                row[f"cost_{key}"] = np.nan
            else:
                r = lv_row.iloc[0]
                row[f"{key}_macro_f1"] = float(r["macro_f1"])
                row[f"cost_{key}"] = float(r["normalized_cost"])
        rows.append(row)

    return pd.DataFrame(rows)
