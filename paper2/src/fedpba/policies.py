"""Deployable policy evaluation for FedPBA Phase-2."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fedpbg.config import PROJECT_ROOT
from fedpbg.metrics.utility import compute_utility
from paper2.src.config import LEVELS
from paper2.src.fedpba.allocator import FedPBAAllocator
from paper2.src.predictors import best_uniform_predict, top2_accuracy

DITTO_CACHE = PROJECT_ROOT / "results" / "ditto" / "cache"
DITTO_LOOKUP: Optional[Dict] = None


def utility_for_action(row: pd.Series, action: str) -> float:
    col = f"utility_{action.lower()}"
    if col in row.index and pd.notna(row[col]):
        return float(row[col])
    f1 = row.get(f"{action.lower()}_macro_f1", np.nan)
    cost = row.get(f"cost_{action.lower()}", np.nan)
    if pd.notna(f1) and pd.notna(cost):
        return float(compute_utility(f1, cost, row["lambda"]))
    return np.nan


def macro_f1_for_action(row: pd.Series, action: str) -> float:
    col = f"{action.lower()}_macro_f1"
    return float(row[col]) if col in row.index and pd.notna(row[col]) else np.nan


def cost_for_action(row: pd.Series, action: str) -> float:
    col = f"cost_{action.lower()}"
    return float(row[col]) if col in row.index and pd.notna(row[col]) else np.nan


def get_ditto_lookup() -> Dict:
    global DITTO_LOOKUP
    if DITTO_LOOKUP is None:
        DITTO_LOOKUP = {}
        if DITTO_CACHE.exists():
            for path in DITTO_CACHE.glob("*_ditto.csv"):
                config_id = path.stem.replace("_ditto", "")
                ddf = pd.read_csv(path)
                for _, r in ddf.iterrows():
                    DITTO_LOOKUP[(config_id, int(r["client_id"]))] = r
    return DITTO_LOOKUP


def ditto_utility(row: pd.Series) -> float:
    d = get_ditto_lookup().get((row["config_id"], int(row["client_id"])))
    if d is None:
        return np.nan
    return float(compute_utility(d["macro_f1"], d["normalized_cost"], row["lambda"]))


def _action_for_policy(row: pd.Series, policy: str) -> Optional[str]:
    if policy == "oracle":
        return row["oracle_action"]
    if policy == "fedpba":
        return row["fedpba_action"]
    if policy == "best_uniform":
        return row["best_uniform_action"]
    if policy == "global_only":
        return "P0"
    if policy == "personalize_all":
        return "P5"
    if policy == "random":
        return row["random_action"]
    return None


def action_distribution(actions: np.ndarray) -> Dict[str, float]:
    counts = {lv: 0 for lv in LEVELS}
    for a in actions:
        if a in counts:
            counts[a] += 1
    n = max(len(actions), 1)
    return {k: v / n for k, v in counts.items()}


def evaluate_policies_on_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    allocator: FedPBAAllocator,
    split_name: str,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    test = test_df.copy().reset_index(drop=True)
    test["fedpba_action"] = allocator.predict(test)
    test["fedpba_utility"] = test.apply(lambda r: utility_for_action(r, r["fedpba_action"]), axis=1)

    bu_actions = best_uniform_predict(train_df, test)
    test["best_uniform_action"] = bu_actions
    test["best_uniform_utility"] = test.apply(
        lambda r: utility_for_action(r, r["best_uniform_action"]), axis=1
    )

    test["global_only_utility"] = test.apply(lambda r: utility_for_action(r, "P0"), axis=1)
    test["p5_all_utility"] = test.apply(lambda r: utility_for_action(r, "P5"), axis=1)
    test["random_action"] = rng.choice(LEVELS, size=len(test))
    test["random_utility"] = test.apply(lambda r: utility_for_action(r, r["random_action"]), axis=1)
    test["ditto_utility"] = test.apply(ditto_utility, axis=1)
    test["regret_fedpba"] = test["oracle_utility"] - test["fedpba_utility"]

    le = LabelEncoder()
    le.fit(LEVELS)
    proba = allocator.predict_proba(test)
    top2 = top2_accuracy(test["oracle_action"].values, proba, le) if proba is not None else np.nan

    fed_dist = action_distribution(test["fedpba_action"].values)
    oracle_dist = action_distribution(test["oracle_action"].values)

    policies = [
        ("oracle", "oracle_utility", None),
        ("fedpba", "fedpba_utility", "fedpba"),
        ("best_uniform", "best_uniform_utility", "best_uniform"),
        ("global_only", "global_only_utility", "global_only"),
        ("random", "random_utility", "random"),
        ("personalize_all", "p5_all_utility", "personalize_all"),
        ("ditto", "ditto_utility", "ditto"),
    ]

    bu_regret = (test["oracle_utility"] - test["best_uniform_utility"]).mean()
    go_regret = (test["oracle_utility"] - test["global_only_utility"]).mean()

    rows = []
    for policy, util_col, action_key in policies:
        mask = test[util_col].notna()
        if mask.sum() == 0:
            continue
        sub = test.loc[mask]
        u = sub[util_col].astype(float)
        oracle_u = sub["oracle_utility"].astype(float)
        regret = oracle_u - u
        f1s, costs = [], []
        for _, r in sub.iterrows():
            if policy == "ditto":
                d = get_ditto_lookup().get((r["config_id"], int(r["client_id"])))
                if d is not None:
                    f1s.append(float(d["macro_f1"]))
                    costs.append(float(d["normalized_cost"]))
            else:
                act = _action_for_policy(r, action_key or policy)
                f1s.append(macro_f1_for_action(r, act))
                costs.append(cost_for_action(r, act))

        row = {
            "split_name": split_name,
            "feature_set": allocator.feature_set,
            "model_name": allocator.model_name,
            "policy": policy,
            "mean_utility": float(u.mean()),
            "mean_regret": float(regret.mean()),
            "median_regret": float(regret.median()),
            "relative_regret": float((regret / np.maximum(np.abs(oracle_u), 0.05)).mean()),
            "mean_macro_f1": float(np.nanmean(f1s)) if f1s else np.nan,
            "mean_cost": float(np.nanmean(costs)) if costs else np.nan,
            "n_clients": int(len(sub)),
        }
        if policy == "fedpba":
            row["action_accuracy"] = float((sub["fedpba_action"] == sub["oracle_action"]).mean())
            row["top2_accuracy"] = float(top2) if not np.isnan(top2) else np.nan
            row["regret_reduction_vs_best_uniform"] = float(bu_regret - regret.mean())
            row["regret_reduction_vs_global_only"] = float(go_regret - regret.mean())
            for lv in LEVELS:
                row[f"action_frac_{lv}"] = fed_dist.get(lv, 0.0)
                row[f"oracle_frac_{lv}"] = oracle_dist.get(lv, 0.0)
        rows.append(row)

    return pd.DataFrame(rows), test
