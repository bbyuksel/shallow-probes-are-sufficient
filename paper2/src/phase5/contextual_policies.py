"""Contextual policy baselines for personalization depth allocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from paper2.src.config import LEVELS
from paper2.src.fedpba.allocator import FedPBAAllocator, train_allocator
from paper2.src.fedpba.allocator_io import load_allocator
from paper2.src.feature_sets import FEATURE_SETS
from paper2.src.online.features import build_client_feature_rows
from paper2.src.online.session import PROBE_LEVELS, OnlineSession
from paper2.src.predictors import encode_features

CONTEXT_ONLY_FEATURES = [
    "dataset",
    "alpha",
    "lambda",
    "num_clients",
    "hardware_profile",
    "hardware_score",
    "hardware_multiplier",
    "heterogeneity",
    "num_samples",
    "label_entropy",
    "majority_class_ratio",
    "class_count",
    "class_balance_score",
]

POLICY_NAMES = [
    "majority",
    "uniform_depth",
    "multinomial_logistic",
    "cost_sensitive_logistic",
    "greedy_probe",
    "contextual_tree",
    "contextual_random_forest",
    "fedpba",
]


@dataclass
class ContextualPolicyBundle:
    """Trained contextual policies for online evaluation."""

    majority_action: str
    multinomial_logistic: FedPBAAllocator
    cost_sensitive_logistic: FedPBAAllocator
    contextual_tree: FedPBAAllocator
    contextual_random_forest: FedPBAAllocator
    fedpba: FedPBAAllocator
    context_cols: List[str] = field(default_factory=list)


def _available_context_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in CONTEXT_ONLY_FEATURES if c in df.columns]


def _utility_class_weights(train_df: pd.DataFrame) -> Dict[str, float]:
    """Class weights proportional to mean oracle utility per action."""
    weights: Dict[str, float] = {}
    for action in LEVELS:
        sub = train_df[train_df["oracle_action"] == action]
        if len(sub) == 0:
            weights[action] = 1.0
        else:
            weights[action] = float(max(sub["oracle_utility"].mean(), 0.05))
    total = sum(weights.values())
    return {k: v / total * len(LEVELS) for k, v in weights.items()}


def _fit_sklearn_allocator(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    model: Any,
    model_name: str,
    sample_weight: Optional[np.ndarray] = None,
) -> FedPBAAllocator:
    X, used, enc, cats = encode_features(train_df, feature_cols)
    le = LabelEncoder()
    le.fit(LEVELS)
    y = le.transform(train_df["oracle_action"].values)
    if sample_weight is not None and hasattr(model, "named_steps"):
        model.fit(X, y, clf__sample_weight=sample_weight)
    elif sample_weight is not None:
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return FedPBAAllocator(model_name, "context_only", used, model, le, enc, cats)


def train_contextual_policies(
    train_df: pd.DataFrame,
    fedpba_path: Optional[str] = None,
) -> ContextualPolicyBundle:
    """Train contextual baselines on oracle-labeled dataset."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    context_cols = _available_context_cols(train_df)
    counts = train_df["oracle_action"].value_counts()
    majority_action = str(counts.idxmax())

    multinomial = _fit_sklearn_allocator(
        train_df,
        context_cols,
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs")),
            ]
        ),
        "multinomial_logistic",
    )

    class_w = _utility_class_weights(train_df)
    sample_w = train_df["oracle_action"].map(class_w).astype(float).values
    cost_sensitive = _fit_sklearn_allocator(
        train_df,
        context_cols,
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs")),
            ]
        ),
        "cost_sensitive_logistic",
        sample_weight=sample_w,
    )

    tree = _fit_sklearn_allocator(
        train_df,
        context_cols,
        DecisionTreeClassifier(max_depth=8, random_state=42),
        "contextual_tree",
    )
    rf = _fit_sklearn_allocator(
        train_df,
        context_cols,
        RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42, n_jobs=1),
        "contextual_random_forest",
    )

    if fedpba_path:
        from pathlib import Path

        fedpba = load_allocator(Path(fedpba_path))
    else:
        probe_cols = [c for c in FEATURE_SETS["context_probes"] if c in train_df.columns]
        fedpba = train_allocator(train_df, probe_cols, "context_probes", "gradient_boosting")

    return ContextualPolicyBundle(
        majority_action=majority_action,
        multinomial_logistic=multinomial,
        cost_sensitive_logistic=cost_sensitive,
        contextual_tree=tree,
        contextual_random_forest=rf,
        fedpba=fedpba,
        context_cols=context_cols,
    )


def build_context_feature_rows(session: OnlineSession, lambda_: float) -> pd.DataFrame:
    """Context-only features (no probe outcomes) for contextual baselines."""
    probe_df = build_client_feature_rows(session, lambda_)
    out = probe_df.copy()
    for col in CONTEXT_ONLY_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
    return out


def greedy_probe_action(group: pd.DataFrame, lambda_: float) -> str:
    """Select probe depth with highest observed utility among P0--P2."""
    from paper2.src.online.policies import utility_at_level

    best_level, best_u = "P0", -np.inf
    for lv in PROBE_LEVELS:
        u = utility_at_level(group, lv, lambda_)
        if not np.isnan(u) and u > best_u:
            best_u = u
            best_level = lv
    return best_level


def predict_policy_actions(
    policy: str,
    bundle: ContextualPolicyBundle,
    session: OnlineSession,
    lambda_: float,
    uniform_depth: str,
) -> np.ndarray:
    """Per-client predicted actions for one session."""
    df = session.client_level_df
    n_clients = session.num_clients
    context_df = build_context_feature_rows(session, lambda_)

    if policy == "majority":
        return np.full(n_clients, bundle.majority_action)
    if policy == "uniform_depth":
        return np.full(n_clients, uniform_depth)
    if policy == "greedy_probe":
        actions = []
        for cid, group in df.groupby("client_id"):
            actions.append(greedy_probe_action(group, lambda_))
        return np.array(actions)
    if policy == "multinomial_logistic":
        return bundle.multinomial_logistic.predict(context_df)
    if policy == "cost_sensitive_logistic":
        return bundle.cost_sensitive_logistic.predict(context_df)
    if policy == "contextual_tree":
        return bundle.contextual_tree.predict(context_df)
    if policy == "contextual_random_forest":
        return bundle.contextual_random_forest.predict(context_df)
    if policy == "fedpba":
        feature_df = build_client_feature_rows(session, lambda_)
        return bundle.fedpba.predict(feature_df)
    raise ValueError(f"Unknown policy: {policy}")


def policy_probe_trainings(policy: str, actions: np.ndarray) -> int:
    """Deployment-time probe trainings (P0--P2 evaluations per client)."""
    if policy in ("majority", "uniform_depth", "multinomial_logistic", "cost_sensitive_logistic", "contextual_tree", "contextual_random_forest"):
        return 0
    if policy == "greedy_probe":
        return len(actions) * len(PROBE_LEVELS)
    if policy == "fedpba":
        extra = sum(1 for a in actions if a in ("P3", "P4", "P5"))
        return len(actions) * len(PROBE_LEVELS) + extra
    return 0
