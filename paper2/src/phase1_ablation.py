"""Phase-1 ablation evaluation across feature sets."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from paper2.src.evaluation import evaluate_baselines_on_split
from paper2.src.feature_sets import FEATURE_SETS
from paper2.src.predictors import aggregate_loo_results, make_splits


def run_ablation_for_feature_set(
    df: pd.DataFrame,
    feature_set_name: str,
    feature_cols: List[str],
    splits: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    splits = splits or make_splits(df)
    all_results: List[Dict] = []
    primary = ["random"] + [s for s in splits if s.startswith(("loo_dataset_", "loo_alpha_", "loo_lambda_"))]

    for split_name in primary:
        train_idx, test_idx = splits[split_name]
        loo = split_name != "random"
        res = evaluate_baselines_on_split(
            df, train_idx, test_idx, split_name, feature_cols, fast=not loo, loo=loo
        )
        for r in res:
            r["feature_set"] = feature_set_name
        all_results.extend(res)

    metrics = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("y_pred", "y_test", "used_cols", "model")} for r in all_results]
    )
    return metrics, all_results


def run_full_ablation(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    splits = make_splits(df)
    all_metrics = []
    all_results = []
    aggregates = {}

    for fs_name, cols in FEATURE_SETS.items():
        available = [c for c in cols if c in df.columns]
        print(f"  Feature set {fs_name}: {len(available)} features", flush=True)
        metrics, results = run_ablation_for_feature_set(df, fs_name, available, splits)
        all_metrics.append(metrics)
        all_results.extend(results)
        rand_gb = metrics[(metrics.split_name == "random") & (metrics.model_name == "gradient_boosting")]
        if len(rand_gb) == 0:
            rand_gb = metrics[(metrics.split_name == "random") & (metrics.model_name == "random_forest")]
        if len(rand_gb):
            aggregates[fs_name] = {
                "random": rand_gb.iloc[0].to_dict(),
                "loo_dataset": aggregate_loo_results(results, "loo_dataset_", "gradient_boosting"),
                "loo_lambda": aggregate_loo_results(results, "loo_lambda_", "gradient_boosting"),
            }

    return pd.concat(all_metrics, ignore_index=True), all_results, aggregates
