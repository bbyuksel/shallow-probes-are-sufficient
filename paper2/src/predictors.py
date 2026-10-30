"""Baseline predictors for oracle action classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from paper2.src.config import LEVELS

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


@dataclass
class SplitResult:
    split_name: str
    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]
    oracle_util: np.ndarray
    utility_matrix: np.ndarray  # (n, 6) utilities per level
    test_idx: np.ndarray


def encode_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    cat_encoder: Optional[OrdinalEncoder] = None,
    cat_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], OrdinalEncoder, List[str]]:
    """Encode mixed numeric/categorical features (unseen categories -> -1)."""
    used_cols: List[str] = []
    parts: List[np.ndarray] = []
    categorical = cat_cols or ["dataset", "hardware_profile"]
    enc = cat_encoder

    cat_present = [c for c in categorical if c in feature_cols and c in df.columns]
    if cat_present:
        cat_df = df[cat_present].astype(str).fillna("NA")
        if enc is None:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc.fit(cat_df)
        parts.append(enc.transform(cat_df).astype(float))
        used_cols.extend(cat_present)

    for col in feature_cols:
        if col not in df.columns or col in cat_present:
            continue
        parts.append(df[col].fillna(0).values.reshape(-1, 1).astype(float))
        used_cols.append(col)

    if not parts:
        raise ValueError("No usable feature columns.")
    X = np.hstack(parts)
    return X, used_cols, enc, cat_present


def _utility_matrix(df: pd.DataFrame) -> np.ndarray:
    mat = np.zeros((len(df), len(LEVELS)))
    for i, lv in enumerate(LEVELS):
        col = f"utility_{lv.lower()}"
        if col in df.columns:
            mat[:, i] = df[col].fillna(-np.inf).values
    return mat


def _oracle_idx(y: np.ndarray) -> np.ndarray:
    le = LabelEncoder()
    le.fit(LEVELS)
    return le.transform(y)


def majority_predict(train_y: np.ndarray, n_test: int) -> np.ndarray:
    vals, counts = np.unique(train_y, return_counts=True)
    majority = vals[np.argmax(counts)]
    return np.full(n_test, majority)


def best_uniform_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Best uniform level from training configs, applied per test config."""
    preds = []
    util_cols = [f"utility_{lv.lower()}" for lv in LEVELS]
    global_best = None
    global_mean = -np.inf

    for cfg in train_df.groupby("config_id"):
        _, g = cfg
        for lam, gl in g.groupby("lambda"):
            means = []
            for col in util_cols:
                means.append(gl[col].mean())
            m = max(means)
            if m > global_mean:
                global_mean = m
                global_best = LEVELS[means.index(m)]

    for _, test_g in test_df.groupby(["config_id", "lambda"]):
        match = train_df[
            (train_df["config_id"] == test_g["config_id"].iloc[0])
            & (train_df["lambda"] == test_g["lambda"].iloc[0])
        ]
        if len(match) > 0:
            means = [match[c].mean() for c in util_cols]
            best = LEVELS[int(np.argmax(means))]
        else:
            sub = train_df[train_df["lambda"] == test_g["lambda"].iloc[0]]
            if len(sub) > 0:
                means = [sub[c].mean() for c in util_cols]
                best = LEVELS[int(np.argmax(means))]
            else:
                best = global_best or "P0"
        preds.extend([best] * len(test_g))
    return np.array(preds)


def global_only_predict(n: int) -> np.ndarray:
    return np.full(n, "P0")


def random_predict(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(LEVELS, size=n)


def get_sklearn_models(seed: int = 42, fast: bool = False, loo: bool = False) -> Dict[str, Any]:
    n_rf = 60 if loo else (80 if fast else 200)
    n_gb = 40 if fast else 150
    models: Dict[str, Any] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=n_rf,
            max_depth=10,
            random_state=seed,
            n_jobs=-1,
        ),
    }
    if not loo:
        models["gradient_boosting"] = HistGradientBoostingClassifier(
            max_iter=n_gb,
            max_depth=6,
            random_state=seed,
            early_stopping=False,
        )
    if not fast and not loo:
        models["mlp"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=200,
                        random_state=seed,
                        early_stopping=True,
                        validation_fraction=0.1,
                    ),
                ),
            ]
        )
    if HAS_XGB and not loo:
        models["xgboost"] = XGBClassifier(
            n_estimators=100 if fast else 200,
            max_depth=5,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=len(LEVELS),
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    return models


def top2_accuracy(y_true: np.ndarray, y_proba: np.ndarray, label_encoder: LabelEncoder) -> float:
    if y_proba is None:
        return np.nan
    top2 = np.argsort(y_proba, axis=1)[:, -2:]
    true_idx = label_encoder.transform(y_true)
    hits = sum(true_idx[i] in top2[i] for i in range(len(y_true)))
    return hits / len(y_true)


def make_splits(df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    n = len(df)
    rng = np.random.default_rng(42)
    idx = np.arange(n)

    # A) Random 80/20 stratified by oracle_action
    from sklearn.model_selection import train_test_split

    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=df["oracle_action"]
    )
    splits = {"random": (train_idx, test_idx)}

    # B) Leave-one-dataset-out — use first dataset as example fold; run all in eval
    for ds in df["dataset"].unique():
        test_mask = df["dataset"] == ds
        splits[f"loo_dataset_{ds}"] = (idx[~test_mask], idx[test_mask])

    # C) Leave-one-alpha-out
    for alpha in sorted(df["alpha"].unique()):
        test_mask = df["alpha"] == alpha
        splits[f"loo_alpha_{alpha}"] = (idx[~test_mask], idx[test_mask])

    # D) Leave-one-lambda-out
    for lam in sorted(df["lambda"].unique()):
        test_mask = df["lambda"] == lam
        splits[f"loo_lambda_{lam}"] = (idx[~test_mask], idx[test_mask])

    return splits


def aggregate_loo_results(
    results: List[Dict],
    prefix: str,
    model_name: Optional[str] = None,
) -> Dict:
    """Aggregate leave-one-out folds (mean over folds, not model×fold rows)."""
    sub = [r for r in results if r["split_name"].startswith(prefix)]
    if not sub:
        return {}
    folds = sorted({r["split_name"] for r in sub})
    per_fold = []
    for fold in folds:
        fold_rows = [r for r in sub if r["split_name"] == fold]
        preferred = None
        for name in (model_name, "gradient_boosting", "random_forest", "logistic_regression"):
            if not name:
                continue
            preferred = next((r for r in fold_rows if r["model_name"] == name), None)
            if preferred:
                break
        if preferred:
            per_fold.append(preferred)
    if not per_fold:
        return {}
    return {
        "accuracy": float(np.mean([r["accuracy"] for r in per_fold])),
        "macro_f1": float(np.mean([r["macro_f1"] for r in per_fold])),
        "top2_accuracy": float(np.nanmean([r.get("top2_accuracy", np.nan) for r in per_fold])),
        "mean_regret": float(np.mean([r["mean_regret"] for r in per_fold])),
        "median_regret": float(np.median([r["median_regret"] for r in per_fold])),
        "n_folds": len(per_fold),
        "model_used": per_fold[0]["model_name"] if per_fold else None,
    }
