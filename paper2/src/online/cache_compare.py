"""Compare live FedPBA with Phase-2 cache replay."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from paper2.src.config import RESULTS_DIR


def compare_with_cache(
    online_detail: pd.DataFrame,
    cache_detail_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Match live per-client FedPBA outcomes to Phase-2 cache replay."""
    cache_path = cache_detail_path or RESULTS_DIR / "phase2_policy_details.csv"
    if not cache_path.exists():
        return pd.DataFrame()

    cache = pd.read_csv(cache_path)
    cache_sub = cache[
        (cache.get("feature_set", pd.Series(["context_probes"] * len(cache))) == "context_probes")
        & (cache.get("model_name", pd.Series(["gradient_boosting"] * len(cache))) == "gradient_boosting")
    ].copy()
    if len(cache_sub) == 0:
        cache_sub = cache.copy()

    online = online_detail.copy()
    online["match_key"] = (
        online["dataset"].astype(str)
        + "_s"
        + online["seed"].astype(str)
        + "_a"
        + online["alpha"].astype(str)
        + "_n"
        + online["num_clients"].astype(str)
        + "_l"
        + online["lambda"].astype(str)
        + "_c"
        + online["client_id"].astype(str)
    )

    cache_sub["match_key"] = (
        cache["config_id"].astype(str)
        + "_l"
        + cache["lambda"].astype(str)
        + "_c"
        + cache["client_id"].astype(str)
    )

    merged = online.merge(
        cache_sub[["match_key", "fedpba_action", "fedpba_utility", "oracle_action", "oracle_utility"]].rename(
            columns={
                "fedpba_action": "cache_fedpba_action",
                "fedpba_utility": "cache_fedpba_utility",
                "oracle_action": "cache_oracle_action",
                "oracle_utility": "cache_oracle_utility",
            }
        ),
        on="match_key",
        how="left",
    )

    rows = []
    for (ds, lam), g in merged.groupby(["dataset", "lambda"]):
        matched = g["cache_fedpba_action"].notna()
        if matched.sum() == 0:
            continue
        sub = g.loc[matched]
        action_match = (sub["fedpba_action"] == sub["cache_fedpba_action"]).mean()
        regret_online = (sub["oracle_utility"] - sub["fedpba_utility"]).mean()
        regret_cache = (sub["cache_oracle_utility"] - sub["cache_fedpba_utility"]).mean()
        rows.append(
            {
                "dataset": ds,
                "lambda": lam,
                "n_matched": int(len(sub)),
                "action_agreement": float(action_match),
                "mean_regret_online": float(regret_online),
                "mean_regret_cache": float(regret_cache),
                "regret_delta": float(regret_online - regret_cache),
            }
        )
    return pd.DataFrame(rows)
