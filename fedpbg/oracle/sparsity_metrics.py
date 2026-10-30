"""
Oracle sparsity metrics and policy comparisons.

Computes P0/deep ratios, uniform vs oracle gaps, and personalize-all penalty.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from fedpbg.config import PERSONALIZATION_LEVELS
from fedpbg.metrics.utility import compute_utility, oracle_level_for_client

DEEP_LEVELS = {"P3", "P4", "P5"}
SHALLOW_LEVELS = {"P1", "P2"}


def level_category(level: str) -> str:
    if level == "P0":
        return "P0"
    if level in SHALLOW_LEVELS:
        return "P1_P2"
    return "P3_P5"


def client_utilities(group: pd.DataFrame, lambda_: float) -> pd.Series:
    return group.apply(
        lambda r: compute_utility(r["macro_f1"], r["normalized_cost"], lambda_),
        axis=1,
    )


def best_uniform_policy(group: pd.DataFrame, lambda_: float) -> Dict:
    """Best single level assigned to all clients in this experiment group."""
    best_level, best_util = None, -np.inf
    for level in PERSONALIZATION_LEVELS:
        sub = group[group["level"] == level]
        if sub.empty:
            continue
        mean_u = client_utilities(sub, lambda_).mean()
        if mean_u > best_util:
            best_util = mean_u
            best_level = level
    return {"best_uniform_level": best_level, "best_uniform_utility": float(best_util)}


def personalize_all_utility(group: pd.DataFrame, lambda_: float) -> float:
    """Utility if every client uses P5."""
    sub = group[group["level"] == "P5"]
    if sub.empty:
        return float("nan")
    return float(client_utilities(sub, lambda_).mean())


def oracle_stats_for_experiment(
    client_df: pd.DataFrame,
    lambda_: float,
) -> Dict:
    """
    Compute oracle sparsity statistics for one experiment configuration.

    client_df must contain one row per (client_id, level) with macro_f1,
    normalized_cost, hardware, etc.
    """
    oracle_levels = []
    oracle_utils = []
    for cid, cg in client_df.groupby("client_id"):
        level, util = oracle_level_for_client(cg, lambda_)
        oracle_levels.append(level)
        oracle_utils.append(util)

    oracle_levels = np.array(oracle_levels)
    n = len(oracle_levels)
    p0_ratio = float(np.mean(oracle_levels == "P0"))
    p1_p2_ratio = float(np.mean(np.isin(oracle_levels, list(SHALLOW_LEVELS))))
    deep_ratio = float(np.mean(np.isin(oracle_levels, list(DEEP_LEVELS))))

    uniform = best_uniform_policy(client_df, lambda_)
    oracle_mean = float(np.mean(oracle_utils))
    uniform_mean = uniform["best_uniform_utility"]
    p5_mean = personalize_all_utility(client_df, lambda_)

    gap_uniform = oracle_mean - uniform_mean
    rel_gap_uniform = gap_uniform / max(abs(uniform_mean), 1e-8)

    gap_p5 = oracle_mean - p5_mean
    rel_gap_p5 = gap_p5 / max(abs(p5_mean), 1e-8)

    # Hardware-wise oracle distribution
    hw_records = []
    hw_breakdown: Dict[str, Dict] = {}
    if "hardware" in client_df.columns:
        cid_to_level = {}
        for cid, cg in client_df.groupby("client_id"):
            lvl, _ = oracle_level_for_client(cg, lambda_)
            cid_to_level[cid] = lvl
        for cid, cg in client_df.groupby("client_id"):
            hw = cg["hardware"].iloc[0]
            lvl = cid_to_level[cid]
            hw_records.append({"hardware": hw, "oracle_level": lvl})
        for hw in ["LOW", "MEDIUM", "HIGH"]:
            hw_lvls = [r["oracle_level"] for r in hw_records if r["hardware"] == hw]
            if hw_lvls:
                hw_arr = np.array(hw_lvls)
                hw_breakdown[hw] = {
                    "p0_ratio": float(np.mean(hw_arr == "P0")),
                    "deep_ratio": float(np.mean(np.isin(hw_arr, list(DEEP_LEVELS)))),
                    "mean_level": float(np.mean([int(x[1]) for x in hw_lvls])),
                }

    return {
        "lambda": lambda_,
        "num_clients": n,
        "p0_ratio": p0_ratio,
        "p1_p2_ratio": p1_p2_ratio,
        "deep_ratio": deep_ratio,
        "best_uniform_level": uniform["best_uniform_level"],
        "best_uniform_utility": uniform_mean,
        "oracle_utility": oracle_mean,
        "personalize_all_utility": p5_mean,
        "oracle_vs_uniform_gap": gap_uniform,
        "oracle_vs_uniform_rel_gap": rel_gap_uniform,
        "oracle_vs_personalize_all_gap": gap_p5,
        "oracle_vs_personalize_all_rel_gap": rel_gap_p5,
        "oracle_levels": oracle_levels.tolist(),
        "hardware_oracle": hw_records,
        "hardware_breakdown": hw_breakdown,
        "mean_oracle_level": float(np.mean([int(x[1]) for x in oracle_levels])),
    }


def build_oracle_client_rows(
    client_df: pd.DataFrame,
    lambda_values: List[float],
    experiment_meta: Dict,
) -> pd.DataFrame:
    """Expand per-client oracle assignments across lambda values."""
    rows = []
    for lam in lambda_values:
        stats = oracle_stats_for_experiment(client_df, lam)
        for cid, cg in client_df.groupby("client_id"):
            level, util = oracle_level_for_client(cg, lam)
            row = cg[cg["level"] == level].iloc[0]
            rows.append(
                {
                    **experiment_meta,
                    "client_id": cid,
                    "lambda": lam,
                    "oracle_level": level,
                    "oracle_utility": util,
                    "macro_f1": row["macro_f1"],
                    "normalized_cost": row["normalized_cost"],
                    "hardware": row["hardware"],
                }
            )
    return pd.DataFrame(rows)


def build_experiment_summary_row(
    client_df: pd.DataFrame,
    lambda_: float,
    experiment_meta: Dict,
) -> Dict:
    stats = oracle_stats_for_experiment(client_df, lambda_)
    row = {
        **experiment_meta,
        "lambda": lambda_,
        "p0_ratio": stats["p0_ratio"],
        "p1_p2_ratio": stats["p1_p2_ratio"],
        "deep_ratio": stats["deep_ratio"],
        "mean_oracle_level": stats["mean_oracle_level"],
        "best_uniform_level": stats["best_uniform_level"],
        "best_uniform_utility": stats["best_uniform_utility"],
        "oracle_utility": stats["oracle_utility"],
        "personalize_all_utility": stats["personalize_all_utility"],
        "oracle_vs_uniform_gap": stats["oracle_vs_uniform_gap"],
        "oracle_vs_uniform_rel_gap": stats["oracle_vs_uniform_rel_gap"],
        "oracle_vs_personalize_all_gap": stats["oracle_vs_personalize_all_gap"],
        "oracle_vs_personalize_all_rel_gap": stats["oracle_vs_personalize_all_rel_gap"],
    }
    for hw, hb in stats.get("hardware_breakdown", {}).items():
        row[f"p0_ratio_{hw}"] = hb["p0_ratio"]
        row[f"deep_ratio_{hw}"] = hb["deep_ratio"]
        row[f"mean_oracle_level_{hw}"] = hb["mean_level"]
    return row
