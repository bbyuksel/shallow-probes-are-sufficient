"""Utility-aware evaluation metrics for Paper-2 Phase-0."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from paper2.src.config import LEVELS
from paper2.src.predictors import (
    SplitResult,
    aggregate_loo_results,
    best_uniform_predict,
    encode_features,
    get_sklearn_models,
    global_only_predict,
    majority_predict,
    make_splits,
    random_predict,
    top2_accuracy,
)


def _predicted_utility(y_pred: np.ndarray, utility_matrix: np.ndarray, le: LabelEncoder) -> np.ndarray:
    idx = le.transform(y_pred)
    return utility_matrix[np.arange(len(y_pred)), idx]


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        boots.append(sample.mean())
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def compute_regret_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    oracle_util: np.ndarray,
    utility_matrix: np.ndarray,
    le: LabelEncoder,
) -> Dict:
    pred_util = _predicted_utility(y_pred, utility_matrix, le)
    regret = oracle_util - pred_util
    rel_regret = regret / np.maximum(np.abs(oracle_util), 1e-8)
    lo, hi = bootstrap_ci(regret)
    return {
        "mean_regret": float(regret.mean()),
        "median_regret": float(np.median(regret)),
        "regret_ci_lo": lo,
        "regret_ci_hi": hi,
        "mean_relative_regret": float(rel_regret.mean()),
        "regret_per_client": regret,
        "pred_util": pred_util,
    }


def evaluate_baselines_on_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_name: str,
    feature_cols: List[str],
    seed: int = 42,
    fast: bool = False,
    loo: bool = False,
) -> List[Dict]:
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    le = LabelEncoder()
    le.fit(LEVELS)

    y_train = train_df["oracle_action"].values
    y_test = test_df["oracle_action"].values
    oracle_util = test_df["oracle_utility"].values
    util_mat = np.zeros((len(test_df), len(LEVELS)))
    for i, lv in enumerate(LEVELS):
        util_mat[:, i] = test_df[f"utility_{lv.lower()}"].values

    results: List[Dict] = []

    baseline_preds = {
        "majority": majority_predict(y_train, len(test_df)),
        "best_uniform": best_uniform_predict(train_df, test_df),
        "global_only": global_only_predict(len(test_df)),
        "random": random_predict(len(test_df), seed=seed),
    }

    for name, y_pred in baseline_preds.items():
        reg = compute_regret_metrics(y_test, y_pred, oracle_util, util_mat, le)
        results.append(
            {
                "split_name": split_name,
                "model_name": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "top2_accuracy": np.nan,
                **{k: v for k, v in reg.items() if k != "regret_per_client" and k != "pred_util"},
            }
        )

    # Oracle reference (zero regret)
    results.append(
        {
            "split_name": split_name,
            "model_name": "oracle",
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "top2_accuracy": 1.0,
            "mean_regret": 0.0,
            "median_regret": 0.0,
            "regret_ci_lo": 0.0,
            "regret_ci_hi": 0.0,
            "mean_relative_regret": 0.0,
        }
    )

    X_train, used_cols, cat_enc, cat_cols = encode_features(train_df, feature_cols)
    X_test, _, _, _ = encode_features(test_df, feature_cols, cat_encoder=cat_enc, cat_cols=cat_cols)

    for model_name, model in get_sklearn_models(seed, fast=fast, loo=loo).items():
        model.fit(X_train, le.transform(y_train))
        y_pred_idx = model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_idx)
        proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        reg = compute_regret_metrics(y_test, y_pred, oracle_util, util_mat, le)
        results.append(
            {
                "split_name": split_name,
                "model_name": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "top2_accuracy": top2_accuracy(y_test, proba, le) if proba is not None else np.nan,
                **{k: v for k, v in reg.items() if k != "regret_per_client" and k != "pred_util"},
                "y_pred": y_pred,
                "y_test": y_test,
                "used_cols": used_cols,
                "model": model,
            }
        )

    return results


def run_full_evaluation(
    cheap_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """Run all splits; return metrics table, raw results, best model info."""
    splits = make_splits(cheap_df)

    # Primary splits for reporting
    primary_splits = ["random"]
    primary_splits += [s for s in splits if s.startswith("loo_dataset_")]
    # Aggregated LOO summaries use all folds

    all_results: List[Dict] = []
    split_items = list(splits.items())
    for i, (split_name, (train_idx, test_idx)) in enumerate(split_items):
        if split_name not in primary_splits and not split_name.startswith(("loo_alpha_", "loo_lambda_")):
            continue
        fast = split_name != "random"
        loo = split_name != "random"
        print(f"    Evaluating split {i + 1}/{len(split_items)}: {split_name}", flush=True)
        res = evaluate_baselines_on_split(
            cheap_df, train_idx, test_idx, split_name, feature_cols, fast=fast, loo=loo
        )
        all_results.extend(res)

    metrics_df = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ("y_pred", "y_test", "used_cols", "model")}
            for r in all_results
        ]
    )

    agg = {
        "loo_dataset": aggregate_loo_results(all_results, "loo_dataset_", model_name="gradient_boosting"),
        "loo_alpha": aggregate_loo_results(all_results, "loo_alpha_", model_name="gradient_boosting"),
        "loo_lambda": aggregate_loo_results(all_results, "loo_lambda_", model_name="gradient_boosting"),
    }

    # Best tree model on random split by mean_regret (with importances for reporting)
    random_tree = [
        r
        for r in all_results
        if r["split_name"] == "random" and r["model_name"] in ("random_forest", "gradient_boosting", "xgboost")
    ]
    best = min(random_tree, key=lambda r: r["mean_regret"]) if random_tree else None
    best_fi = next((r for r in random_tree if r["model_name"] == "random_forest"), best)

    return metrics_df, all_results, {"best_tree": best, "best_tree_fi": best_fi, "aggregates": agg}


def confusion_matrix_for_result(result: Dict, le: LabelEncoder) -> np.ndarray:
    return confusion_matrix(result["y_test"], result["y_pred"], labels=le.classes_)
