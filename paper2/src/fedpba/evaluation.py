"""FedPBA Phase-2 evaluation orchestration."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from paper2.src.config import LEVELS
from paper2.src.feature_sets import FEATURE_SETS
from paper2.src.fedpba.allocator import ALLOCATOR_MODELS, train_allocator
from paper2.src.fedpba.policies import evaluate_policies_on_split
from paper2.src.predictors import make_splits, top2_accuracy


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return np.nan, np.nan
    boots = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def _split_keys(splits: Dict, include_alpha: bool = True) -> List[str]:
    keys = ["random"]
    for k in sorted(splits):
        if k.startswith("loo_dataset_") or k.startswith("loo_lambda_"):
            keys.append(k)
        elif include_alpha and k.startswith("loo_alpha_"):
            keys.append(k)
    return keys


def run_fedpba_model_comparison(
    df: pd.DataFrame,
    feature_set: str = "context_probes",
    include_alpha: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train all allocator models; evaluate deployable policies on all splits."""
    feature_cols = [c for c in FEATURE_SETS[feature_set] if c in df.columns]
    splits = make_splits(df)
    split_keys = _split_keys(splits, include_alpha)

    all_summary, all_details = [], []
    for split_name in split_keys:
        train_idx, test_idx = splits[split_name]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        if len(test_df) == 0 or len(train_df) == 0:
            continue
        for model_name in ALLOCATOR_MODELS:
            alloc = train_allocator(train_df, feature_cols, feature_set, model_name)
            summary, detail = evaluate_policies_on_split(train_df, test_df, alloc, split_name)
            detail["split_name"] = split_name
            detail["model_name"] = model_name
            detail["feature_set"] = feature_set
            all_summary.append(summary)
            all_details.append(detail)

    return pd.concat(all_summary, ignore_index=True), pd.concat(all_details, ignore_index=True)


def run_ablation_variants(
    df: pd.DataFrame,
    model_name: str = "gradient_boosting",
    include_alpha: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ablation A–E across splits."""
    splits = make_splits(df)
    split_keys = _split_keys(splits, include_alpha)

    all_summary, all_details = [], []
    for fs_name, fs_cols in FEATURE_SETS.items():
        cols = [c for c in fs_cols if c in df.columns]
        for split_name in split_keys:
            train_idx, test_idx = splits[split_name]
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            if len(test_df) == 0 or len(train_df) == 0:
                continue
            alloc = train_allocator(train_df, cols, fs_name, model_name)
            summary, detail = evaluate_policies_on_split(train_df, test_df, alloc, split_name)
            detail["split_name"] = split_name
            detail["model_name"] = model_name
            detail["feature_set"] = fs_name
            all_summary.append(summary)
            all_details.append(detail)

    return pd.concat(all_summary, ignore_index=True), pd.concat(all_details, ignore_index=True)


def aggregate_loo(summary: pd.DataFrame, split_prefix: str, policy: str = "fedpba", model_name: Optional[str] = None) -> Dict:
    sub = summary[(summary.split_name.str.startswith(split_prefix)) & (summary.policy == policy)]
    if model_name:
        sub = sub[sub.model_name == model_name]
    if sub.empty:
        return {}
    return {
        "mean_regret": float(sub["mean_regret"].mean()),
        "mean_utility": float(sub["mean_utility"].mean()),
        "action_accuracy": float(sub["action_accuracy"].mean()) if "action_accuracy" in sub.columns else np.nan,
        "n_folds": len(sub),
    }


def pick_best_model(summary: pd.DataFrame, split: str = "random", feature_set: str = "context_probes") -> str:
    sub = summary[
        (summary.split_name == split)
        & (summary.feature_set == feature_set)
        & (summary.policy == "fedpba")
    ]
    if sub.empty:
        return "gradient_boosting"
    return sub.sort_values("mean_regret").iloc[0]["model_name"]


def confusion_for_split(detail: pd.DataFrame) -> np.ndarray:
    le = LabelEncoder()
    le.fit(LEVELS)
    y_true = le.transform(detail["oracle_action"])
    y_pred = le.transform(detail["fedpba_action"])
    return confusion_matrix(y_true, y_pred, labels=range(len(LEVELS)))
