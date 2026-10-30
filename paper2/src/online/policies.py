"""Evaluate deployable policies on live FL sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fedpbg.config import PROJECT_ROOT
from fedpbg.metrics.utility import compute_utility, oracle_level_for_client
from paper2.src.config import LEVELS
from paper2.src.fedpba.allocator import FedPBAAllocator
from paper2.src.online.features import build_client_feature_rows
from paper2.src.online.session import OnlineSession, PROBE_LEVELS
from paper2.src.predictors import top2_accuracy

DITTO_CACHE = PROJECT_ROOT / "results" / "ditto" / "cache"


def _level_lookup(group: pd.DataFrame, level: str) -> Tuple[float, float]:
    row = group[group["level"] == level]
    if row.empty:
        return np.nan, np.nan
    r = row.iloc[0]
    return float(r["macro_f1"]), float(r["normalized_cost"])


def utility_at_level(group: pd.DataFrame, level: str, lambda_: float) -> float:
    f1, cost = _level_lookup(group, level)
    if np.isnan(f1):
        return np.nan
    return float(compute_utility(f1, cost, lambda_))


def best_uniform_level(session: OnlineSession, lambda_: float) -> str:
    df = session.client_level_df
    best_level, best_mean = "P0", -np.inf
    for lv in LEVELS:
        utils = []
        for _, group in df.groupby("client_id"):
            u = utility_at_level(group, lv, lambda_)
            if not np.isnan(u):
                utils.append(u)
        if utils and np.mean(utils) > best_mean:
            best_mean = float(np.mean(utils))
            best_level = lv
    return best_level


def fedpba_extra_trainings(actions: np.ndarray) -> int:
    """Probe trainings are always 3 per client; extra if predicted depth > P2."""
    extra = 0
    for a in actions:
        if a in ("P3", "P4", "P5"):
            extra += 1
    return extra


def load_ditto_utilities(session: OnlineSession, lambda_: float) -> Dict[int, float]:
    path = DITTO_CACHE / f"{session.config_id}_ditto.csv"
    out: Dict[int, float] = {}
    if not path.exists():
        return out
    ddf = pd.read_csv(path)
    df = session.client_level_df
    for _, row in ddf.iterrows():
        cid = int(row["client_id"])
        p5 = df[(df["client_id"] == cid) & (df["level"] == "P5")]
        cost = float(p5.iloc[0]["normalized_cost"]) if len(p5) else 1.0
        out[cid] = float(compute_utility(row["macro_f1"], cost, lambda_))
    return out


def evaluate_session_policies(
    session: OnlineSession,
    lambda_: float,
    allocator: FedPBAAllocator,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Per-client policy outcomes for one session and lambda."""
    df = session.client_level_df
    feature_df = build_client_feature_rows(session, lambda_)
    fedpba_actions = allocator.predict(feature_df)
    feature_df["fedpba_action"] = fedpba_actions

    bu_level = best_uniform_level(session, lambda_)
    ditto_map = load_ditto_utilities(session, lambda_)

    rows: List[Dict] = []
    for cid, group in df.groupby("client_id"):
        feat_row = feature_df[feature_df["client_id"] == cid].iloc[0]
        oracle_level, oracle_u = oracle_level_for_client(group, lambda_)
        fed_action = feat_row["fedpba_action"]
        fed_u = utility_at_level(group, fed_action, lambda_)
        bu_u = utility_at_level(group, bu_level, lambda_)
        p0_u = utility_at_level(group, "P0", lambda_)
        p5_u = utility_at_level(group, "P5", lambda_)

        rows.append(
            {
                "config_id": session.config_id,
                "dataset": session.resolved_dataset,
                "alpha": session.alpha,
                "lambda": lambda_,
                "seed": session.seed,
                "num_clients": session.num_clients,
                "client_id": int(cid),
                "oracle_action": oracle_level,
                "oracle_utility": oracle_u,
                "fedpba_action": fed_action,
                "fedpba_utility": fed_u,
                "best_uniform_action": bu_level,
                "best_uniform_utility": bu_u,
                "global_only_utility": p0_u,
                "p5_all_utility": p5_u,
                "ditto_utility": ditto_map.get(int(cid), np.nan),
            }
        )

    detail = pd.DataFrame(rows)

    le = LabelEncoder()
    le.fit(LEVELS)
    proba = allocator.predict_proba(feature_df)
    top2 = top2_accuracy(detail["oracle_action"].values, proba, le) if proba is not None else np.nan

    extra_train = fedpba_extra_trainings(fedpba_actions)
    timing = session.timing
    n_clients = session.num_clients
    probe_trainings = timing.get("n_probe_trainings", n_clients * len(PROBE_LEVELS))
    oracle_trainings = timing.get("n_oracle_trainings", n_clients * len(LEVELS))

    policies = [
        ("oracle", "oracle_utility", "oracle_action"),
        ("fedpba", "fedpba_utility", "fedpba_action"),
        ("best_uniform", "best_uniform_utility", "best_uniform_action"),
        ("global_only", "global_only_utility", None),
        ("personalize_all", "p5_all_utility", None),
        ("ditto", "ditto_utility", None),
    ]

    summary_rows = []
    for policy, util_col, action_col in policies:
        mask = detail[util_col].notna()
        if mask.sum() == 0:
            continue
        sub = detail.loc[mask]
        u = sub[util_col].astype(float)
        oracle_u = sub["oracle_utility"].astype(float)
        regret = oracle_u - u
        summary_rows.append(
            {
                "config_id": session.config_id,
                "dataset": session.resolved_dataset,
                "alpha": session.alpha,
                "lambda": lambda_,
                "seed": session.seed,
                "policy": policy,
                "mean_utility": float(u.mean()),
                "mean_regret": float(regret.mean()),
                "relative_regret": float((regret / np.maximum(np.abs(oracle_u), 0.05)).mean()),
                "action_accuracy": float((sub[action_col] == sub["oracle_action"]).mean())
                if action_col and action_col in sub.columns
                else np.nan,
                "top2_accuracy": float(top2) if policy == "fedpba" and not np.isnan(top2) else np.nan,
                "n_clients": int(len(sub)),
                "fedavg_sec": timing.get("fedavg_sec", np.nan),
                "probe_sec": timing.get("probe_sec", np.nan),
                "oracle_levels_sec": timing.get("oracle_levels_sec", np.nan),
                "probe_trainings": probe_trainings,
                "oracle_trainings": oracle_trainings,
                "fedpba_extra_trainings": extra_train if policy == "fedpba" else np.nan,
                "probe_cost_ratio": probe_trainings / max(oracle_trainings, 1),
            }
        )

    meta = {
        "extra_trainings": extra_train,
        "probe_sec": timing.get("probe_sec"),
        "oracle_sec": timing.get("oracle_levels_sec"),
    }
    return pd.DataFrame(summary_rows), detail, meta
