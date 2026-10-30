"""Grouped cross-validation for methodology v2 (session-level GroupKFold)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from paper2.src.methodology_v2.build_rows import (
    CONTEXT_ONLY_FEATURES,
    CONTEXT_PROBES_FEATURES,
    LEVELS,
)

# Prefer shared encoder; fall back to a minimal OrdinalEncoder wrapper.
try:
    from paper2.src.predictors import encode_features
except ImportError:  # pragma: no cover
    from sklearn.preprocessing import OrdinalEncoder

    def encode_features(  # type: ignore[misc]
        df: pd.DataFrame,
        feature_cols: List[str],
        cat_encoder=None,
        cat_cols: Optional[List[str]] = None,
    ):
        used: List[str] = []
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
            used.extend(cat_present)
        for col in feature_cols:
            if col not in df.columns or col in cat_present:
                continue
            parts.append(df[col].fillna(0).values.reshape(-1, 1).astype(float))
            used.append(col)
        if not parts:
            raise ValueError("No usable feature columns.")
        return np.hstack(parts), used, enc, cat_present


def _utility_of_action(df: pd.DataFrame, actions: np.ndarray) -> np.ndarray:
    """Look up ``utility_{level}`` for each predicted action (positional, safe)."""
    actions = np.asarray(actions).astype(str)
    out = np.full(len(df), np.nan, dtype=float)
    for lv in LEVELS:
        col = f"utility_{lv.lower()}"
        if col not in df.columns:
            continue
        mask = actions == lv
        if mask.any():
            vals = df[col].to_numpy(dtype=float, copy=False)
            out[mask] = vals[mask]
    return out


def _assert_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_name: str,
) -> Dict[str, Any]:
    train_sessions = set(train_df["session_id"].astype(str))
    test_sessions = set(test_df["session_id"].astype(str))
    train_clients = set(train_df["client_instance_id"].astype(str))
    test_clients = set(test_df["client_instance_id"].astype(str))

    overlap_sessions = sorted(train_sessions & test_sessions)
    overlap_clients = sorted(train_clients & test_clients)

    # Same client_instance must not span folds via λ copies (implied by client set).
    report = {
        "fold": fold_name,
        "train_sessions": len(train_sessions),
        "test_sessions": len(test_sessions),
        "overlapping_sessions": len(overlap_sessions),
        "overlapping_clients": len(overlap_clients),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "oracle_action_distribution_train": dict(
            Counter(train_df["oracle_action"].astype(str))
        ),
        "oracle_action_distribution_test": dict(
            Counter(test_df["oracle_action"].astype(str))
        ),
        "overlap_session_ids": overlap_sessions,
        "overlap_client_instance_ids": overlap_clients,
    }
    if overlap_sessions or overlap_clients:
        raise AssertionError(
            f"Leakage in fold {fold_name}: "
            f"sessions={overlap_sessions[:5]}… clients={overlap_clients[:5]}…"
        )
    return report


def _fit_hgb(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    random_state: int = 42,
) -> Tuple[Any, Any, LabelEncoder, List[str], List[str]]:
    """Fit HistGradientBoostingClassifier on train fold only."""
    cols = [c for c in feature_cols if c in train_df.columns]
    X, used, enc, cats = encode_features(train_df, cols)
    le = LabelEncoder()
    le.fit(LEVELS)
    y = le.transform(train_df["oracle_action"].values)
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=6,
        random_state=random_state,
        early_stopping=False,
    )
    clf.fit(X, y)
    return clf, enc, le, used, cats


def _fit_context_only(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    random_state: int = 42,
) -> Tuple[Any, Any, LabelEncoder, List[str], List[str]]:
    """Context-only learner (HistGradientBoosting; RF fallback if needed)."""
    cols = [c for c in feature_cols if c in train_df.columns]
    X, used, enc, cats = encode_features(train_df, cols)
    le = LabelEncoder()
    le.fit(LEVELS)
    y = le.transform(train_df["oracle_action"].values)
    try:
        clf = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=6,
            random_state=random_state,
            early_stopping=False,
        )
        clf.fit(X, y)
    except Exception:
        clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            random_state=random_state,
            n_jobs=1,
        )
        clf.fit(X, y)
    return clf, enc, le, used, cats


def _predict_clf(
    clf: Any,
    enc: Any,
    le: LabelEncoder,
    used: List[str],
    cats: List[str],
    test_df: pd.DataFrame,
) -> np.ndarray:
    X, _, _, _ = encode_features(test_df, used, cat_encoder=enc, cat_cols=cats)
    pred_idx = clf.predict(X)
    return le.inverse_transform(pred_idx)


def best_uniform_from_train(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """
    For each (dataset, alpha, lambda) pick depth maximizing mean utility on TRAIN
    only; apply that depth to matching test rows. Fallbacks: λ-only, then global.
    """
    util_cols = [f"utility_{lv.lower()}" for lv in LEVELS]
    # Stratum → best level
    stratum_best: Dict[Tuple, str] = {}
    for keys, g in train_df.groupby(["dataset", "alpha", "lambda"]):
        means = [float(g[c].mean()) for c in util_cols]
        stratum_best[keys] = LEVELS[int(np.argmax(means))]

    lam_best: Dict[float, str] = {}
    for lam, g in train_df.groupby("lambda"):
        means = [float(g[c].mean()) for c in util_cols]
        lam_best[float(lam)] = LEVELS[int(np.argmax(means))]

    global_means = [float(train_df[c].mean()) for c in util_cols]
    global_best = LEVELS[int(np.argmax(global_means))]

    preds = []
    for _, row in test_df.iterrows():
        key = (row["dataset"], row["alpha"], row["lambda"])
        if key in stratum_best:
            preds.append(stratum_best[key])
        elif float(row["lambda"]) in lam_best:
            preds.append(lam_best[float(row["lambda"])])
        else:
            preds.append(global_best)
    return np.array(preds)


def majority_from_train(train_df: pd.DataFrame, n_test: int) -> np.ndarray:
    mode = train_df["oracle_action"].mode(dropna=True)
    majority = str(mode.iloc[0]) if len(mode) else "P0"
    return np.full(n_test, majority)


def _policy_metrics(detail: pd.DataFrame, action_col: str, util_col: str) -> Dict[str, float]:
    if util_col not in detail.columns:
        return {}
    u = detail[util_col].astype(float)
    oracle_u = detail["oracle_utility"].astype(float)
    regret = oracle_u - u
    oracle_sum = float(oracle_u.sum())
    recovery = float(u.sum() / oracle_sum * 100.0) if oracle_sum != 0 else float("nan")
    deep = {"P3", "P4", "P5"}
    deep_mask = detail["oracle_action"].isin(deep)
    deep_recall = (
        float((detail.loc[deep_mask, action_col].isin(deep)).mean())
        if deep_mask.any() and action_col in detail.columns
        else float("nan")
    )
    p0_deep = 0
    if action_col in detail.columns:
        p0_deep = int(
            ((detail[action_col] == "P0") & detail["oracle_action"].isin(deep)).sum()
        )
    return {
        "mean_utility": float(u.mean()),
        "mean_regret": float(regret.mean()),
        "oracle_utility_recovery_pct": recovery,
        "action_accuracy": float((detail[action_col] == detail["oracle_action"]).mean())
        if action_col in detail.columns
        else float("nan"),
        "deep_oracle_recall": deep_recall,
        "p0_to_deep_error_count": p0_deep,
        "n": int(u.notna().sum()),
        "recovery_numerator": float(u.sum()),
        "recovery_denominator": oracle_sum,
    }


def _run_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_name: str,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    integrity = _assert_no_leakage(train_df, test_df, fold_name)

    # FEDPBA (context_probes)
    fed_clf, fed_enc, fed_le, fed_used, fed_cats = _fit_hgb(
        train_df, CONTEXT_PROBES_FEATURES, random_state=random_state
    )
    fed_actions = _predict_clf(fed_clf, fed_enc, fed_le, fed_used, fed_cats, test_df)

    # Context-only
    ctx_clf, ctx_enc, ctx_le, ctx_used, ctx_cats = _fit_context_only(
        train_df, CONTEXT_ONLY_FEATURES, random_state=random_state
    )
    ctx_actions = _predict_clf(ctx_clf, ctx_enc, ctx_le, ctx_used, ctx_cats, test_df)

    # Best uniform (train-only depth selection)
    bu_actions = best_uniform_from_train(train_df, test_df)

    # Majority
    maj_actions = majority_from_train(train_df, len(test_df))

    # Greedy: no fit — use precomputed greedy_probe_action on test
    greedy_actions = test_df["greedy_probe_action"].astype(str).values

    out = test_df.copy()
    out["fold"] = fold_name
    out["fedpba_action"] = fed_actions
    out["fedpba_utility"] = _utility_of_action(out, fed_actions)
    out["best_uniform_action"] = bu_actions
    out["best_uniform_utility"] = _utility_of_action(out, bu_actions)
    out["majority_action"] = maj_actions
    out["majority_utility"] = _utility_of_action(out, maj_actions)
    out["context_only_action"] = ctx_actions
    out["context_only_utility"] = _utility_of_action(out, ctx_actions)
    out["greedy_action"] = greedy_actions
    out["greedy_utility"] = _utility_of_action(out, greedy_actions)
    # Aliases matching phase5 naming
    out["greedy_probe_action"] = greedy_actions
    out["greedy_probe_utility"] = out["greedy_utility"]

    return out, integrity


def _aggregate_oof_metrics(oof: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    policies = {
        "fedpba": ("fedpba_action", "fedpba_utility"),
        "best_uniform": ("best_uniform_action", "best_uniform_utility"),
        "majority": ("majority_action", "majority_utility"),
        "context_only": ("context_only_action", "context_only_utility"),
        "greedy": ("greedy_action", "greedy_utility"),
        "oracle": ("oracle_action", "oracle_utility"),
    }
    return {name: _policy_metrics(oof, a, u) for name, (a, u) in policies.items()}


def run_group_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    out_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    GroupKFold by session_id. Returns (oof_detail, metrics, integrity_report).
    """
    groups = df["session_id"].astype(str).values
    gkf = GroupKFold(n_splits=n_splits)
    oof_parts: List[pd.DataFrame] = []
    integrity_rows: List[Dict[str, Any]] = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        fold_name = f"groupkfold_{fold_i}"
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        part, integrity = _run_fold(train_df, test_df, fold_name, random_state=random_state)
        oof_parts.append(part)
        integrity_rows.append(integrity)

    oof = pd.concat(oof_parts, axis=0).sort_index()
    # Ensure one prediction per original row
    if len(oof) != len(df):
        raise RuntimeError(f"OOF length mismatch: {len(oof)} vs {len(df)}")

    integrity_df = pd.DataFrame(integrity_rows)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Flatten nested dicts for CSV
        flat_rows = []
        for r in integrity_rows:
            flat = {
                k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                for k, v in r.items()
            }
            flat_rows.append(flat)
        pd.DataFrame(flat_rows).to_csv(out_dir / "split_integrity_report.csv", index=False)

    metrics = _aggregate_oof_metrics(oof)
    return oof, metrics, integrity_df


def run_leave_one_seed_out(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """5-fold leave-one-seed-out (one seed held out entirely)."""
    seeds = sorted(df["seed"].unique())
    parts: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, Any]] = []

    for seed in seeds:
        fold_name = f"loo_seed_{seed}"
        test_df = df[df["seed"] == seed]
        train_df = df[df["seed"] != seed]
        part, _ = _run_fold(train_df, test_df, fold_name, random_state=random_state)
        parts.append(part)
        m = _aggregate_oof_metrics(part)
        fold_metrics.append(
            {
                "fold": fold_name,
                "test_seed": int(seed),
                **{f"fedpba_{k}": v for k, v in m["fedpba"].items()},
                **{f"greedy_{k}": v for k, v in m["greedy"].items()},
            }
        )

    oof = pd.concat(parts, axis=0).sort_index()
    summary = {
        "protocol": "leave_one_seed_out",
        "n_folds": len(seeds),
        "per_fold": fold_metrics,
        "aggregate": _aggregate_oof_metrics(oof),
        "fedpba_regret_mean": float(np.mean([f["fedpba_mean_regret"] for f in fold_metrics])),
        "fedpba_regret_std": float(np.std([f["fedpba_mean_regret"] for f in fold_metrics], ddof=1))
        if len(fold_metrics) > 1
        else 0.0,
    }
    return oof, summary


def run_leave_one_dataset_out(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """3-fold leave-one-dataset-out."""
    datasets = sorted(df["dataset"].astype(str).unique())
    parts: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, Any]] = []

    for ds in datasets:
        fold_name = f"loo_dataset_{ds}"
        test_df = df[df["dataset"].astype(str) == ds]
        train_df = df[df["dataset"].astype(str) != ds]
        part, _ = _run_fold(train_df, test_df, fold_name, random_state=random_state)
        parts.append(part)
        m = _aggregate_oof_metrics(part)
        fold_metrics.append(
            {
                "fold": fold_name,
                "test_dataset": ds,
                **{f"fedpba_{k}": v for k, v in m["fedpba"].items()},
                **{f"greedy_{k}": v for k, v in m["greedy"].items()},
            }
        )

    oof = pd.concat(parts, axis=0).sort_index()
    summary = {
        "protocol": "leave_one_dataset_out",
        "n_folds": len(datasets),
        "per_fold": fold_metrics,
        "aggregate": _aggregate_oof_metrics(oof),
    }
    return oof, summary


def run_grouped_evaluation(
    df: pd.DataFrame,
    out_dir: Path | str,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Full methodology-v2 grouped evaluation.

    Writes ``split_integrity_report.csv`` and ``grouped_oof_predictions.csv``.
    Returns metrics dicts for GroupKFold, leave-one-seed-out, leave-one-dataset-out.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {"session_id", "client_instance_id", "oracle_action", "oracle_utility"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    oof, gkf_metrics, integrity = run_group_kfold(
        df, n_splits=n_splits, random_state=random_state, out_dir=out_dir
    )
    oof.to_csv(out_dir / "grouped_oof_predictions.csv", index=False)

    loo_seed_oof, loo_seed_summary = run_leave_one_seed_out(df, random_state=random_state)
    loo_seed_oof.to_csv(out_dir / "loo_seed_oof_predictions.csv", index=False)
    with open(out_dir / "loo_seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(loo_seed_summary), f, indent=2)

    loo_ds_oof, loo_ds_summary = run_leave_one_dataset_out(df, random_state=random_state)
    loo_ds_oof.to_csv(out_dir / "loo_dataset_oof_predictions.csv", index=False)
    with open(out_dir / "loo_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(loo_ds_summary), f, indent=2)

    # Flatten GroupKFold metrics for easy loading
    gkf_rows = [{"policy": k, **v} for k, v in gkf_metrics.items()]
    pd.DataFrame(gkf_rows).to_csv(out_dir / "grouped_cv_policy_metrics.csv", index=False)

    return {
        "group_kfold": gkf_metrics,
        "leave_one_seed_out": loo_seed_summary,
        "leave_one_dataset_out": loo_ds_summary,
        "oof": oof,
        "integrity": integrity,
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if np.isnan(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
