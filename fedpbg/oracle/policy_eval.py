"""
Fixed-policy evaluation from cached oracle client-level results.

Policies: ORACLE, RANDOM, BEST_UNIFORM, PERSONALIZE_ALL, GLOBAL_ONLY.
No learned allocator — regret study only.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from fedpbg.config import PERSONALIZATION_LEVELS
from fedpbg.metrics.utility import compute_utility, oracle_level_for_client

POLICIES = ["ORACLE", "RANDOM", "BEST_UNIFORM", "PERSONALIZE_ALL", "GLOBAL_ONLY"]


def _client_utility_row(group: pd.DataFrame, level: str, lambda_: float) -> Tuple[float, float, float]:
    row = group[group["level"] == level].iloc[0]
    u = compute_utility(row["macro_f1"], row["normalized_cost"], lambda_)
    return u, float(row["macro_f1"]), float(row["normalized_cost"])


def _oracle_per_client(client_df: pd.DataFrame, lambda_: float) -> Dict[int, Tuple[str, float, float, float]]:
    out = {}
    for cid, group in client_df.groupby("client_id"):
        level, u = oracle_level_for_client(group, lambda_)
        row = group[group["level"] == level].iloc[0]
        out[int(cid)] = (level, u, float(row["macro_f1"]), float(row["normalized_cost"]))
    return out


def _random_levels(num_clients: int, seed: int) -> Dict[int, str]:
    rng = np.random.default_rng(seed + 7919)
    return {i: rng.choice(PERSONALIZATION_LEVELS).item() for i in range(num_clients)}


def evaluate_policy(
    client_df: pd.DataFrame,
    lambda_: float,
    policy: str,
    seed: int,
) -> Dict:
    """
    Compute per-policy aggregate metrics for one experiment configuration.

    Returns mean/std utility, mean macro-F1, mean cost, and per-client utilities.
    """
    oracle_map = _oracle_per_client(client_df, lambda_)
    client_ids = sorted(oracle_map.keys())

    utilities: List[float] = []
    macro_f1s: List[float] = []
    costs: List[float] = []
    chosen_levels: List[str] = []

    if policy == "ORACLE":
        for cid in client_ids:
            lvl, u, f1, c = oracle_map[cid]
            chosen_levels.append(lvl)
            utilities.append(u)
            macro_f1s.append(f1)
            costs.append(c)

    elif policy == "GLOBAL_ONLY":
        for cid, group in client_df.groupby("client_id"):
            u, f1, c = _client_utility_row(group, "P0", lambda_)
            utilities.append(u)
            macro_f1s.append(f1)
            costs.append(c)
            chosen_levels.append("P0")

    elif policy == "PERSONALIZE_ALL":
        for cid, group in client_df.groupby("client_id"):
            u, f1, c = _client_utility_row(group, "P5", lambda_)
            utilities.append(u)
            macro_f1s.append(f1)
            costs.append(c)
            chosen_levels.append("P5")

    elif policy == "BEST_UNIFORM":
        best_level, best_mean = None, -np.inf
        for level in PERSONALIZATION_LEVELS:
            us = []
            for cid, group in client_df.groupby("client_id"):
                u, _, _ = _client_utility_row(group, level, lambda_)
                us.append(u)
            m = float(np.mean(us))
            if m > best_mean:
                best_mean = m
                best_level = level
        for cid, group in client_df.groupby("client_id"):
            u, f1, c = _client_utility_row(group, best_level, lambda_)
            utilities.append(u)
            macro_f1s.append(f1)
            costs.append(c)
            chosen_levels.append(best_level)

    elif policy == "RANDOM":
        rand_map = _random_levels(len(client_ids), seed)
        for i, cid in enumerate(client_ids):
            group = client_df[client_df["client_id"] == cid]
            lvl = str(rand_map[i])
            u, f1, c = _client_utility_row(group, lvl, lambda_)
            utilities.append(u)
            macro_f1s.append(f1)
            costs.append(c)
            chosen_levels.append(lvl)

    else:
        raise ValueError(f"Unknown policy: {policy}")

    arr = np.array(utilities, dtype=np.float64)
    return {
        "policy": policy,
        "lambda": lambda_,
        "mean_utility": float(arr.mean()),
        "std_utility": float(arr.std()),
        "mean_macro_f1": float(np.mean(macro_f1s)),
        "mean_cost": float(np.mean(costs)),
        "client_utilities": utilities,
        "chosen_levels": chosen_levels,
    }


def evaluate_all_policies(
    client_df: pd.DataFrame,
    lambda_: float,
    seed: int,
    experiment_meta: Dict,
) -> List[Dict]:
    """Evaluate all policies and compute regret vs ORACLE."""
    rows = []
    results = {}
    for policy in POLICIES:
        res = evaluate_policy(client_df, lambda_, policy, seed)
        results[policy] = res

    oracle_u = results["ORACLE"]["mean_utility"]
    for policy in POLICIES:
        res = results[policy]
        regret = oracle_u - res["mean_utility"]
        # Floor denominator to avoid blow-up when oracle utility ≈ 0
        rel_regret = regret / max(abs(oracle_u), 0.05, 1e-8)
        rows.append(
            {
                **experiment_meta,
                "lambda": lambda_,
                "policy": policy,
                "mean_utility": res["mean_utility"],
                "std_utility": res["std_utility"],
                "mean_macro_f1": res["mean_macro_f1"],
                "mean_cost": res["mean_cost"],
                "oracle_utility": oracle_u,
                "regret": regret,
                "relative_regret": rel_regret,
            }
        )
    return rows, results
