"""Deep-oracle correlation and policy comparison analyses (no fabricated numbers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

DEEP = {"P3", "P4", "P5"}
SHALLOW = {"P0", "P1", "P2"}
LEVEL_DEPTH = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def _pearson_r_only(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Pearson r without p-value (for bootstrap loops)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    sx = float(np.sqrt(np.dot(x, x)))
    sy = float(np.sqrt(np.dot(y, y)))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.dot(x, y) / (sx * sy))


def session_cluster_pearson_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    session_col: str = "session_id",
    n_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Session-cluster bootstrap 95% CI for Pearson r (resample sessions w/ replacement)."""
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    sessions = df[session_col].astype(str).to_numpy()
    uniq, inverse = np.unique(sessions, return_inverse=True)
    n_sess = len(uniq)
    if n_sess == 0:
        return float("nan"), float("nan"), float("nan")

    point_r = _pearson_r_only(x, y)
    # Row starts/counts via sorting inverse for contiguous blocks
    order = np.argsort(inverse, kind="mergesort")
    inv_sorted = inverse[order]
    x_s, y_s = x[order], y[order]
    counts = np.bincount(inv_sorted, minlength=n_sess)
    starts = np.zeros(n_sess, dtype=int)
    starts[1:] = np.cumsum(counts[:-1])

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        chosen = rng.integers(0, n_sess, n_sess)
        # Build concatenated sample indices into sorted arrays
        total = int(counts[chosen].sum())
        xb = np.empty(total, dtype=float)
        yb = np.empty(total, dtype=float)
        pos = 0
        for s in chosen:
            n = int(counts[s])
            st = int(starts[s])
            xb[pos : pos + n] = x_s[st : st + n]
            yb[pos : pos + n] = y_s[st : st + n]
            pos += n
        boots[b] = _pearson_r_only(xb, yb)
    return (
        float(point_r),
        float(np.nanpercentile(boots, 2.5)),
        float(np.nanpercentile(boots, 97.5)),
    )


def session_cluster_mean_ci(
    values: np.ndarray,
    sessions: np.ndarray,
    n_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Cluster bootstrap CI for the mean of row-level values (cluster = session)."""
    values = np.asarray(values, dtype=float)
    sessions = np.asarray(sessions).astype(str)
    mask = ~np.isnan(values)
    values, sessions = values[mask], sessions[mask]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(values))
    uniq, inverse = np.unique(sessions, return_inverse=True)
    n_sess = len(uniq)
    order = np.argsort(inverse, kind="mergesort")
    inv_sorted = inverse[order]
    vals_s = values[order]
    counts = np.bincount(inv_sorted, minlength=n_sess)
    starts = np.zeros(n_sess, dtype=int)
    starts[1:] = np.cumsum(counts[:-1])
    # Precompute per-session sums and counts for O(1) resample mean
    sess_sum = np.array(
        [vals_s[starts[i] : starts[i] + counts[i]].sum() for i in range(n_sess)],
        dtype=float,
    )
    sess_n = counts.astype(float)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        chosen = rng.integers(0, n_sess, n_sess)
        boots[b] = float(sess_sum[chosen].sum() / sess_n[chosen].sum())
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def enrich_deep_oracle_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Compute shallow-best, deep-gain, deltas, margin, and oracle_deep flag."""
    df = rows.copy()
    u0 = df["utility_p0"].astype(float)
    u1 = df["utility_p1"].astype(float)
    u2 = df["utility_p2"].astype(float)
    shallow = np.vstack([u0.to_numpy(), u1.to_numpy(), u2.to_numpy()])
    # best / second-best among P0–P2
    order = np.sort(shallow, axis=0)
    second_best = order[-2]
    best = order[-1]
    df["shallow_best_utility"] = best
    df["shallow_margin"] = best - second_best
    df["delta_01"] = u1 - u0
    df["delta_12"] = u2 - u1
    df["delta_02"] = u2 - u0
    df["shallow_slope_max"] = np.maximum(df["delta_01"].to_numpy(), df["delta_12"].to_numpy())
    df["deep_gain"] = df["oracle_utility"].astype(float) - df["shallow_best_utility"]
    df["oracle_deep"] = df["oracle_action"].astype(str).isin(DEEP)
    return df


def merge_rows_and_oof(
    policy_rows: pd.DataFrame,
    oof: pd.DataFrame,
) -> pd.DataFrame:
    """Merge policy-learning rows with OOF predictions on session/client/lambda."""
    keys = ["session_id", "client_id", "lambda"]
    enriched = enrich_deep_oracle_frame(policy_rows)
    enriched = enriched.copy()
    oof = oof.copy()
    enriched["client_id"] = enriched["client_id"].astype(int)
    oof["client_id"] = oof["client_id"].astype(int)
    enriched["session_id"] = enriched["session_id"].astype(str)
    oof["session_id"] = oof["session_id"].astype(str)
    enriched["lambda"] = enriched["lambda"].astype(float)
    oof["lambda"] = oof["lambda"].astype(float)

    pred_cols = [
        c
        for c in [
            "session_id",
            "client_id",
            "lambda",
            "fedpba_action",
            "fedpba_utility",
            "greedy_action",
            "greedy_utility",
            "best_uniform_action",
            "best_uniform_utility",
            "context_only_action",
            "context_only_utility",
        ]
        if c in oof.columns
    ]
    merged = enriched.merge(oof[pred_cols], on=keys, how="inner", suffixes=("", "_oof"))
    if len(merged) != len(enriched):
        raise RuntimeError(
            f"Merge length mismatch: enriched={len(enriched)} merged={len(merged)} "
            "(check session_id/client_id/lambda keys)"
        )
    # Prefer OOF greedy columns when present
    if "greedy_action" not in merged.columns and "greedy_probe_action" in merged.columns:
        merged["greedy_action"] = merged["greedy_probe_action"].astype(str)
    if "greedy_utility" not in merged.columns and "greedy_probe_utility" in merged.columns:
        merged["greedy_utility"] = merged["greedy_probe_utility"].astype(float)
    if "greedy_action" in merged.columns:
        merged["greedy_action"] = merged["greedy_action"].astype(str)
    if "fedpba_action" in merged.columns:
        merged["fedpba_action"] = merged["fedpba_action"].astype(str)
    return merged


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson + Spearman + session-cluster Pearson CI for A/B/C/D blocks."""
    records: List[Dict[str, Any]] = []

    def _add(
        subset: str,
        x_col: str,
        y_col: str,
        frame: pd.DataFrame,
    ) -> None:
        x = frame[x_col].to_numpy(float)
        y = frame[y_col].to_numpy(float)
        n = int(np.sum(~(np.isnan(x) | np.isnan(y))))
        pr, pp = _pearson_safe(x, y)
        sr, sp = _spearman_safe(x, y)
        _, ci_lo, ci_hi = session_cluster_pearson_ci(frame, x_col, y_col)
        records.append(
            {
                "subset": subset,
                "x": x_col,
                "y": y_col,
                "n": n,
                "n_sessions": int(frame["session_id"].nunique()),
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_rho": sr,
                "spearman_p": sp,
                "pearson_r_ci_lo": ci_lo,
                "pearson_r_ci_hi": ci_hi,
                "bootstrap_B": BOOTSTRAP_B,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )

    all_df = df
    shallow_df = df[~df["oracle_deep"]]
    deep_df = df[df["oracle_deep"]]

    _add("A_all_rows", "shallow_best_utility", "oracle_utility", all_df)
    _add("B_shallow_oracle", "shallow_best_utility", "oracle_utility", shallow_df)
    _add("C_deep_oracle", "shallow_best_utility", "oracle_utility", deep_df)
    for x_col in ["delta_01", "delta_12", "delta_02", "shallow_margin", "shallow_slope_max"]:
        _add(f"D_deep_oracle_{x_col}_vs_deep_gain", x_col, "deep_gain", deep_df)

    return pd.DataFrame(records)


def _depth(action: Any) -> float:
    return float(LEVEL_DEPTH.get(str(action), np.nan))


def _policy_deep_metrics(
    deep: pd.DataFrame,
    action_col: str,
    util_col: str,
    policy: str,
    stratum: str = "all",
) -> Dict[str, Any]:
    if deep.empty or action_col not in deep.columns or util_col not in deep.columns:
        return {
            "stratum": stratum,
            "policy": policy,
            "n": 0,
            "error": f"missing columns or empty deep set ({action_col}/{util_col})",
        }
    u = deep[util_col].astype(float)
    oracle_u = deep["oracle_utility"].astype(float)
    regret = oracle_u - u
    actions = deep[action_col].astype(str)
    oracle_a = deep["oracle_action"].astype(str)
    pred_deep = actions.isin(DEEP)
    mean_r, ci_lo, ci_hi = session_cluster_mean_ci(
        regret.to_numpy(), deep["session_id"].astype(str).to_numpy()
    )
    oracle_sum = float(oracle_u.sum())
    recovery = float(u.sum() / oracle_sum * 100.0) if oracle_sum != 0 else float("nan")
    return {
        "stratum": stratum,
        "policy": policy,
        "n": int(len(deep)),
        "n_sessions": int(deep["session_id"].nunique()),
        "mean_regret": mean_r,
        "median_regret": float(np.median(regret)),
        "regret_ci_lo": ci_lo,
        "regret_ci_hi": ci_hi,
        "regret_le_0_01_rate": float((regret <= 0.01).mean()),
        "regret_le_0_05_rate": float((regret <= 0.05).mean()),
        "regret_gt_0_05_count": int((regret > 0.05).sum()),
        "regret_gt_0_05_rate": float((regret > 0.05).mean()),
        "action_accuracy": float((actions == oracle_a).mean()),
        "shallow_vs_deep_accuracy": float((pred_deep == True).mean()),  # noqa: E712 — oracle all deep
        "predicted_deep_count": int(pred_deep.sum()),
        "p0_to_deep_count": int(((actions == "P0") & oracle_a.isin(DEEP)).sum()),
        "p1_to_deep_count": int(((actions == "P1") & oracle_a.isin(DEEP)).sum()),
        "p2_to_deep_count": int(((actions == "P2") & oracle_a.isin(DEEP)).sum()),
        "mean_deployed_depth": float(actions.map(_depth).mean()),
        "mean_oracle_depth": float(oracle_a.map(_depth).mean()),
        "mean_utility": float(u.mean()),
        "oracle_utility_recovery_pct": recovery,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def policy_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Deep-oracle policy metrics overall, by dataset, and by lambda."""
    deep = df[df["oracle_deep"]].copy()
    rows: List[Dict[str, Any]] = []

    policies = [
        ("greedy", "greedy_action", "greedy_utility"),
        ("fedpba", "fedpba_action", "fedpba_utility"),
    ]
    for name, a_col, u_col in policies:
        rows.append(_policy_deep_metrics(deep, a_col, u_col, name, "all"))

    if "dataset" in deep.columns:
        for ds, g in deep.groupby("dataset"):
            for name, a_col, u_col in policies:
                rows.append(_policy_deep_metrics(g, a_col, u_col, name, f"dataset={ds}"))

    if "lambda" in deep.columns:
        for lam, g in deep.groupby("lambda"):
            for name, a_col, u_col in policies:
                rows.append(
                    _policy_deep_metrics(g, a_col, u_col, name, f"lambda={float(lam)}")
                )

    return pd.DataFrame(rows)


def paired_deep_descriptive(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Paired greedy − fedpba utility on deep rows.

    If deep rows span >=10 sessions, also report session-level paired mean diff
    with cluster bootstrap CI (no fake p-values when coverage is thin).
    """
    deep = df[df["oracle_deep"]].copy()
    out: Dict[str, Any] = {
        "n_deep_rows": int(len(deep)),
        "n_deep_sessions": int(deep["session_id"].nunique()) if len(deep) else 0,
    }
    if deep.empty:
        out["note"] = "no deep-oracle rows"
        return out
    if "greedy_utility" not in deep.columns or "fedpba_utility" not in deep.columns:
        out["error"] = "missing greedy_utility or fedpba_utility"
        return out

    diff = deep["greedy_utility"].astype(float) - deep["fedpba_utility"].astype(float)
    out["row_mean_utility_diff_greedy_minus_fedpba"] = float(diff.mean())
    out["row_median_utility_diff_greedy_minus_fedpba"] = float(diff.median())

    n_sess = int(deep["session_id"].nunique())
    if n_sess >= 10:
        # session-level mean utilities, then paired diff
        sess = (
            deep.groupby("session_id")
            .agg(
                greedy_u=("greedy_utility", "mean"),
                fedpba_u=("fedpba_utility", "mean"),
            )
            .dropna()
        )
        sdiff = sess["greedy_u"].to_numpy(float) - sess["fedpba_u"].to_numpy(float)
        point = float(sdiff.mean())
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boots = [float(sdiff[rng.integers(0, len(sdiff), len(sdiff))].mean()) for _ in range(BOOTSTRAP_B)]
        out["session_paired_mean_diff"] = point
        out["session_paired_ci_lo"] = float(np.percentile(boots, 2.5))
        out["session_paired_ci_hi"] = float(np.percentile(boots, 97.5))
        out["session_paired_n"] = int(len(sdiff))
        out["inferential_test"] = "not_reported_thin_or_descriptive_only"
        out["note"] = (
            "Session-level paired mean difference with bootstrap CI; "
            "no p-value reported (descriptive / coverage-aware)."
        )
    else:
        out["note"] = (
            f"Only {n_sess} sessions contain deep-oracle rows (<10); "
            "reporting row-level descriptive diffs only (no session paired CI / p)."
        )
    return out


def run_deep_oracle_analysis(
    policy_rows_path: Path | str,
    oof_path: Path | str,
    out_dir: Path | str,
) -> Dict[str, Any]:
    """Write deep-oracle CSVs and return summary dict with real computed values."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(policy_rows_path)
    oof = pd.read_csv(oof_path)
    merged = merge_rows_and_oof(rows, oof)

    corr = correlation_analysis(merged)
    corr.to_csv(out_dir / "deep_oracle_correlation_analysis.csv", index=False)

    pol = policy_comparison(merged)
    pol.to_csv(out_dir / "deep_oracle_policy_comparison.csv", index=False)

    paired = paired_deep_descriptive(merged)
    paired_path = out_dir / "deep_oracle_paired_descriptive.json"
    import json

    with open(paired_path, "w", encoding="utf-8") as f:
        json.dump(paired, f, indent=2)

    # Enrichment snapshot for figures
    keep = [
        c
        for c in [
            "session_id",
            "client_id",
            "lambda",
            "dataset",
            "oracle_action",
            "oracle_utility",
            "shallow_best_utility",
            "deep_gain",
            "oracle_deep",
            "delta_01",
            "delta_12",
            "delta_02",
            "shallow_margin",
            "greedy_action",
            "greedy_utility",
            "fedpba_action",
            "fedpba_utility",
        ]
        if c in merged.columns
    ]
    merged[keep].to_csv(out_dir / "deep_oracle_enriched_rows.csv", index=False)

    a_row = corr[corr["subset"] == "A_all_rows"].iloc[0]
    c_row = corr[corr["subset"] == "C_deep_oracle"].iloc[0]
    greedy_all = pol[(pol["stratum"] == "all") & (pol["policy"] == "greedy")]
    fed_all = pol[(pol["stratum"] == "all") & (pol["policy"] == "fedpba")]

    return {
        "n_rows": int(len(merged)),
        "n_shallow": int((~merged["oracle_deep"]).sum()),
        "n_deep": int(merged["oracle_deep"].sum()),
        "aggregate_pearson_r": float(a_row["pearson_r"]),
        "deep_only_pearson_r": float(c_row["pearson_r"]),
        "deep_only_pearson_ci": (float(c_row["pearson_r_ci_lo"]), float(c_row["pearson_r_ci_hi"])),
        "deep_only_n": int(c_row["n"]),
        "greedy_deep_mean_regret": float(greedy_all["mean_regret"].iloc[0]) if len(greedy_all) else float("nan"),
        "fedpba_deep_mean_regret": float(fed_all["mean_regret"].iloc[0]) if len(fed_all) else float("nan"),
        "paired": paired,
        "corr": corr,
        "policy": pol,
        "merged": merged,
    }
