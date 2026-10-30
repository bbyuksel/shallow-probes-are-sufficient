"""
Utility computation and oracle personalization level selection.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def compute_utility(macro_f1: float, normalized_cost: float, lambda_: float) -> float:
    """U = Macro-F1 - lambda * normalized_cost"""
    return macro_f1 - lambda_ * normalized_cost


def oracle_level_for_client(
    client_rows: pd.DataFrame,
    lambda_: float,
) -> Tuple[str, float]:
    """
    Return (best_level, best_utility) for one client at given lambda.

    client_rows must contain columns: level, macro_f1, normalized_cost
    """
    utilities = client_rows.apply(
        lambda r: compute_utility(r["macro_f1"], r["normalized_cost"], lambda_),
        axis=1,
    )
    best_idx = utilities.idxmax()
    return client_rows.loc[best_idx, "level"], float(utilities.loc[best_idx])


def build_oracle_table(client_df: pd.DataFrame, lambda_values: List[float]) -> pd.DataFrame:
    """Oracle budget P_i* for each client and lambda."""
    records = []
    for cid, group in client_df.groupby("client_id"):
        for lam in lambda_values:
            level, utility = oracle_level_for_client(group, lam)
            row = group[group["level"] == level].iloc[0]
            records.append(
                {
                    "client_id": cid,
                    "lambda": lam,
                    "oracle_level": level,
                    "oracle_utility": utility,
                    "macro_f1": row["macro_f1"],
                    "normalized_cost": row["normalized_cost"],
                    "heterogeneity": row["heterogeneity"],
                    "hardware": row["hardware"],
                }
            )
    return pd.DataFrame(records)
