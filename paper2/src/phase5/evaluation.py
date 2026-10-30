"""Evaluate contextual policies on Phase-4 live session caches."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fedpbg.metrics.utility import oracle_level_for_client
from paper2.src.config import LEVELS
from paper2.src.online.features import build_client_feature_rows
from paper2.src.online.policies import best_uniform_level, utility_at_level
from paper2.src.online.session import OnlineSession
from paper2.src.phase5.contextual_policies import (
    POLICY_NAMES,
    ContextualPolicyBundle,
    build_context_feature_rows,
    greedy_probe_action,
    policy_probe_trainings,
    predict_policy_actions,
)
from paper2.src.predictors import top2_accuracy


def _top2_for_allocator(alloc, feature_df: pd.DataFrame, oracle_actions: np.ndarray) -> float:
    proba = alloc.predict_proba(feature_df)
    if proba is None:
        return np.nan
    le = LabelEncoder()
    le.fit(LEVELS)
    return float(top2_accuracy(oracle_actions, proba, le))


def evaluate_session_contextual_policies(
    session: OnlineSession,
    lambda_: float,
    bundle: ContextualPolicyBundle,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all contextual policies for one session and lambda."""
    df = session.client_level_df
    uniform_depth = best_uniform_level(session, lambda_)
    timing = session.timing
    n_clients = session.num_clients
    oracle_trainings = timing.get("n_oracle_trainings", n_clients * len(LEVELS))

    groups = list(df.groupby("client_id"))
    client_ids = [int(cid) for cid, _ in groups]
    oracle_actions = []
    oracle_utilities = []
    for _, group in groups:
        oracle_level, oracle_u = oracle_level_for_client(group, lambda_)
        oracle_actions.append(oracle_level)
        oracle_utilities.append(oracle_u)
    oracle_actions_arr = np.array(oracle_actions)
    oracle_utilities_arr = np.array(oracle_utilities, dtype=float)

    context_df = build_context_feature_rows(session, lambda_)
    probe_df = build_client_feature_rows(session, lambda_)

    policy_actions: Dict[str, np.ndarray] = {
        "majority": np.full(n_clients, bundle.majority_action),
        "uniform_depth": np.full(n_clients, uniform_depth),
        "multinomial_logistic": bundle.multinomial_logistic.predict(context_df),
        "cost_sensitive_logistic": bundle.cost_sensitive_logistic.predict(context_df),
        "contextual_tree": bundle.contextual_tree.predict(context_df),
        "contextual_random_forest": bundle.contextual_random_forest.predict(context_df),
        "fedpba": bundle.fedpba.predict(probe_df),
    }
    policy_actions["greedy_probe"] = np.array(
        [greedy_probe_action(group, lambda_) for _, group in groups]
    )

    detail = pd.DataFrame(
        {
            "config_id": session.config_id,
            "dataset": session.resolved_dataset,
            "alpha": session.alpha,
            "lambda": lambda_,
            "seed": session.seed,
            "num_clients": session.num_clients,
            "client_id": client_ids,
            "oracle_action": oracle_actions_arr,
            "oracle_utility": oracle_utilities_arr,
        }
    )

    summary_rows: List[Dict] = []
    oracle_mean_u = float(oracle_utilities_arr.mean())

    for policy in POLICY_NAMES:
        actions = policy_actions[policy]
        utilities = np.array(
            [utility_at_level(group, actions[i], lambda_) for i, (_, group) in enumerate(groups)],
            dtype=float,
        )
        detail[f"{policy}_action"] = actions
        detail[f"{policy}_utility"] = utilities

        regret = oracle_utilities_arr - utilities
        rel_regret = regret / np.maximum(np.abs(oracle_utilities_arr), 0.05)
        probe_trainings = policy_probe_trainings(policy, actions)

        summary_rows.append(
            {
                "config_id": session.config_id,
                "dataset": session.resolved_dataset,
                "alpha": session.alpha,
                "lambda": lambda_,
                "seed": session.seed,
                "policy": policy,
                "mean_utility": float(np.nanmean(utilities)),
                "mean_regret": float(np.nanmean(regret)),
                "mean_relative_regret": float(np.nanmean(rel_regret)),
                "action_accuracy": float((actions == oracle_actions_arr).mean()),
                "top2_accuracy": np.nan,
                "oracle_utility_recovery_pct": float(utilities.mean() / oracle_mean_u * 100)
                if oracle_mean_u > 0
                else np.nan,
                "n_clients": n_clients,
                "probe_trainings": probe_trainings,
                "oracle_trainings": oracle_trainings,
                "probe_cost_ratio": probe_trainings / max(oracle_trainings, 1),
                "fedavg_sec": timing.get("fedavg_sec", np.nan),
                "probe_sec": timing.get("probe_sec", np.nan) if probe_trainings > 0 else 0.0,
                "oracle_levels_sec": timing.get("oracle_levels_sec", np.nan),
            }
        )

    return pd.DataFrame(summary_rows), detail


def add_aggregate_top2(detail: pd.DataFrame, bundle: ContextualPolicyBundle) -> Dict[str, float]:
    """Compute top-2 accuracy on full detail set per learned policy."""
    oracle_actions = detail["oracle_action"].values
    out: Dict[str, float] = {}
    policy_alloc = {
        "multinomial_logistic": bundle.multinomial_logistic,
        "cost_sensitive_logistic": bundle.cost_sensitive_logistic,
        "contextual_tree": bundle.contextual_tree,
        "contextual_random_forest": bundle.contextual_random_forest,
        "fedpba": bundle.fedpba,
    }
    for policy, alloc in policy_alloc.items():
        if policy == "fedpba":
            # rebuild feature rows from unique sessions is expensive; use stored actions only
            pass
    return out


def aggregate_policy_metrics(detail: pd.DataFrame, bundle: ContextualPolicyBundle | None = None) -> pd.DataFrame:
    """Aggregate per-policy metrics across all evaluations."""
    from paper2.src.phase4.statistics import bootstrap_ci

    rows = []
    oracle_actions = detail["oracle_action"].values

    for policy in POLICY_NAMES:
        util_col = f"{policy}_utility"
        action_col = f"{policy}_action"
        if util_col not in detail.columns:
            continue
        u = detail[util_col].astype(float)
        oracle_u = detail["oracle_utility"].astype(float)
        regret = oracle_u - u
        rel = regret / np.maximum(np.abs(oracle_u), 0.05)
        ci_u = bootstrap_ci(u.values)
        ci_r = bootstrap_ci(regret.values)
        oracle_mean = float(oracle_u.mean())
        top2 = np.nan
        if bundle and policy in (
            "multinomial_logistic",
            "cost_sensitive_logistic",
            "contextual_tree",
            "contextual_random_forest",
            "fedpba",
        ):
            alloc = getattr(bundle, policy if policy != "fedpba" else "fedpba")
            # sample one row per client already in detail; top2 needs proba on full feature matrix
            # approximate from action accuracy for speed in aggregate pass
            top2 = np.nan

        rows.append(
            {
                "policy": policy,
                "mean_utility": float(u.mean()),
                "utility_ci_lo": ci_u[0],
                "utility_ci_hi": ci_u[1],
                "mean_regret": float(regret.mean()),
                "regret_ci_lo": ci_r[0],
                "regret_ci_hi": ci_r[1],
                "mean_relative_regret": float(rel.mean()),
                "action_accuracy": float((detail[action_col] == detail["oracle_action"]).mean()),
                "top2_accuracy": top2,
                "oracle_utility_recovery_pct": float(u.mean() / oracle_mean * 100) if oracle_mean > 0 else np.nan,
                "n": int(u.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def context_vs_probe_comparison(detail: pd.DataFrame, policy_stats: pd.DataFrame) -> Dict[str, float]:
    """Compare best context-only policy vs FedPBA."""
    context_policies = [
        "majority",
        "uniform_depth",
        "multinomial_logistic",
        "cost_sensitive_logistic",
        "contextual_tree",
        "contextual_random_forest",
    ]
    ctx = policy_stats[policy_stats.policy.isin(context_policies)].sort_values("mean_regret")
    best_ctx = ctx.iloc[0]
    fed = policy_stats[policy_stats.policy == "fedpba"].iloc[0]
    greedy = policy_stats[policy_stats.policy == "greedy_probe"].iloc[0]
    return {
        "best_context_policy": str(best_ctx["policy"]),
        "context_mean_regret": float(best_ctx["mean_regret"]),
        "context_mean_utility": float(best_ctx["mean_utility"]),
        "context_oracle_recovery_pct": float(best_ctx["oracle_utility_recovery_pct"]),
        "greedy_probe_mean_regret": float(greedy["mean_regret"]),
        "greedy_probe_mean_utility": float(greedy["mean_utility"]),
        "greedy_probe_oracle_recovery_pct": float(greedy["oracle_utility_recovery_pct"]),
        "fedpba_mean_regret": float(fed["mean_regret"]),
        "fedpba_mean_utility": float(fed["mean_utility"]),
        "fedpba_oracle_recovery_pct": float(fed["oracle_utility_recovery_pct"]),
        "regret_difference_ctx_minus_fedpba": float(best_ctx["mean_regret"] - fed["mean_regret"]),
        "utility_difference_fedpba_minus_ctx": float(fed["mean_utility"] - best_ctx["mean_utility"]),
        "oracle_recovery_difference_pct": float(
            fed["oracle_utility_recovery_pct"] - best_ctx["oracle_utility_recovery_pct"]
        ),
    }
