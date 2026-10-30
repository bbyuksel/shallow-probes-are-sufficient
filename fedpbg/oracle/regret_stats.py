"""
Statistical robustness analysis for oracle regret study.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASELINE_POLICIES = ["RANDOM", "BEST_UNIFORM", "PERSONALIZE_ALL", "GLOBAL_ONLY"]


def paired_oracle_test(
    oracle_utilities: List[float],
    policy_utilities: List[float],
) -> Tuple[float, float]:
    """
    Paired Wilcoxon signed-rank test (oracle vs policy, client-level).

    Returns (statistic, p_value). p=1.0 if insufficient variation.
    """
    o = np.asarray(oracle_utilities, dtype=np.float64)
    p = np.asarray(policy_utilities, dtype=np.float64)
    if len(o) < 5 or np.allclose(o, p):
        return 0.0, 1.0
    try:
        stat, pval = wilcoxon(o - p, alternative="greater", zero_method="wilcox")
        return float(stat), float(pval)
    except ValueError:
        return 0.0, 1.0


def robustness_by_dataset(
    regret_df: pd.DataFrame,
    policy: str,
) -> Dict:
    """Summarize how often oracle beats a baseline policy."""
    sub = regret_df[regret_df["policy"] == policy].copy()
    if sub.empty:
        return {}

    wins = (sub["regret"] > 1e-8).sum()
    n = len(sub)
    return {
        "policy": policy,
        "n_configs": n,
        "oracle_win_rate": float(wins / max(n, 1)),
        "mean_regret": float(sub["regret"].mean()),
        "median_regret": float(sub["regret"].median()),
        "mean_relative_regret_pct": float(100 * sub["relative_regret"].mean()),
        "std_regret": float(sub["regret"].std()),
    }


def build_regret_table(regret_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows = (dataset, alpha, lambda, seed), columns = policy regrets."""
    pivot = regret_df.pivot_table(
        index=["dataset", "dirichlet_alpha", "lambda", "seed"],
        columns="policy",
        values="regret",
        aggfunc="first",
    ).reset_index()
    return pivot


def aggregate_significance(
    all_policy_results: List[Dict],
) -> pd.DataFrame:
    """
    Aggregate Wilcoxon tests across configurations.

    all_policy_results: list of dicts with keys oracle_utilities, policy_utilities, meta
    """
    records = []
    for baseline in BASELINE_POLICIES:
        stats_list = []
        for item in all_policy_results:
            if baseline not in item["policy_utils"]:
                continue
            _, pval = paired_oracle_test(
                item["oracle_utils"],
                item["policy_utils"][baseline],
            )
            stats_list.append(pval)
        if stats_list:
            records.append(
                {
                    "baseline": baseline,
                    "n_tests": len(stats_list),
                    "fraction_p_lt_0.05": float(np.mean(np.array(stats_list) < 0.05)),
                    "median_p_value": float(np.median(stats_list)),
                }
            )
    return pd.DataFrame(records)
