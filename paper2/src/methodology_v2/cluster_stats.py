"""Session-level cluster bootstrap, paired tests, and Holm correction."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from paper2.src.methodology_v2 import PROTOCOL

POLICIES = [
    "oracle",
    "fedpba",
    "best_uniform",
    "majority",
    "context_only",
    "greedy",
]

# Map policy name → (action_col, utility_col) on OOF / detail frames
POLICY_COLS = {
    "oracle": ("oracle_action", "oracle_utility"),
    "fedpba": ("fedpba_action", "fedpba_utility"),
    "best_uniform": ("best_uniform_action", "best_uniform_utility"),
    "majority": ("majority_action", "majority_utility"),
    "context_only": ("context_only_action", "context_only_utility"),
    "greedy": ("greedy_action", "greedy_utility"),
}

# Accept greedy_probe_* aliases
_ALIAS = {
    "greedy_action": ["greedy_probe_action"],
    "greedy_utility": ["greedy_probe_utility"],
}

PRIMARY_COMPARISONS = [
    ("greedy", "best_uniform", "greedy_vs_best_uniform"),
    ("fedpba", "best_uniform", "fedpba_vs_best_uniform"),
    ("greedy", "fedpba", "greedy_vs_fedpba"),
]

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 42


def _meta_cols() -> Dict[str, str]:
    return {
        "protocol_version": PROTOCOL,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _resolve_col(df: pd.DataFrame, primary: str) -> str:
    if primary in df.columns:
        return primary
    for alt in _ALIAS.get(primary, []):
        if alt in df.columns:
            return alt
    raise KeyError(f"Missing column {primary!r} (aliases tried: {_ALIAS.get(primary, [])})")


def aggregate_session_level(detail: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate client–λ rows → one row per session × policy.

    Recovery numerator/denominator are sums (not mean of ratios).
    """
    records: List[Dict] = []
    deep = {"P3", "P4", "P5"}

    for sid, g in detail.groupby("session_id"):
        base = {
            "session_id": sid,
            "dataset": g["dataset"].iloc[0],
            "alpha": float(g["alpha"].iloc[0]),
            "seed": int(g["seed"].iloc[0]),
            "n_rows": int(len(g)),
        }
        oracle_u = g["oracle_utility"].astype(float)
        oracle_sum = float(oracle_u.sum())

        for policy, (action_key, util_key) in POLICY_COLS.items():
            try:
                util_col = _resolve_col(g, util_key)
            except KeyError:
                continue
            u = g[util_col].astype(float)
            regret = oracle_u - u
            row = {
                **base,
                "policy": policy,
                "mean_utility": float(u.mean()),
                "mean_regret": float(regret.mean()),
                "recovery_numerator": float(u.sum()),
                "recovery_denominator": oracle_sum,
                "oracle_recovery_pct": float(u.sum() / oracle_sum * 100.0)
                if oracle_sum != 0
                else np.nan,
            }
            try:
                action_col = _resolve_col(g, action_key)
                row["action_accuracy"] = float(
                    (g[action_col].astype(str) == g["oracle_action"].astype(str)).mean()
                )
                deep_mask = g["oracle_action"].isin(deep)
                row["deep_oracle_recall"] = (
                    float(g.loc[deep_mask, action_col].isin(deep).mean())
                    if deep_mask.any()
                    else np.nan
                )
                row["p0_to_deep_error_count"] = int(
                    ((g[action_col] == "P0") & g["oracle_action"].isin(deep)).sum()
                )
            except KeyError:
                row["action_accuracy"] = np.nan
                row["deep_oracle_recall"] = np.nan
                row["p0_to_deep_error_count"] = np.nan
            records.append(row)

    return pd.DataFrame(records)


def _session_utility_vector(session_metrics: pd.DataFrame, policy: str) -> pd.Series:
    sub = session_metrics[session_metrics["policy"] == policy].set_index("session_id")
    return sub["mean_utility"].astype(float)


def holm_correction(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values (same order as input)."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        # rank 0 → multiply by m; rank 1 → m-1; ...
        raw = float(p_values[idx]) * (m - rank)
        running = max(running, raw)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def paired_session_test(
    session_metrics: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    n_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """Wilcoxon + Cohen dz + cluster-bootstrap CI on session mean-utility diffs (a − b)."""
    ua = _session_utility_vector(session_metrics, policy_a)
    ub = _session_utility_vector(session_metrics, policy_b)
    common = ua.index.intersection(ub.index)
    a = ua.loc[common].values.astype(float)
    b = ub.loc[common].values.astype(float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"n_sessions": int(len(a))}

    diff = a - b
    # Two-sided Wilcoxon on session-level differences
    try:
        w = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0

    rng = np.random.default_rng(seed)
    boots = []
    n = len(diff)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(float(diff[idx].mean()))
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))

    return {
        "n_sessions": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "std_diff": float(diff.std(ddof=1)),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "cohens_dz": dz,
        "diff_ci_lo": ci_lo,
        "diff_ci_hi": ci_hi,
        "win_rate": float((diff > 0).mean()),
    }


def cluster_bootstrap_summary(
    session_metrics: pd.DataFrame,
    policies: Optional[Sequence[str]] = None,
    n_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """
    Resample 45 sessions with replacement; compute mean regret and recovery per policy.

    Recovery = sum(recovery_numerator) / sum(recovery_denominator) over resampled sessions.
    """
    policies = list(policies) if policies is not None else [
        p for p in POLICIES if p != "oracle" and p in set(session_metrics["policy"])
    ]
    # Pivot to session × policy
    sessions = sorted(session_metrics["session_id"].unique())
    n_sess = len(sessions)
    sid_to_i = {s: i for i, s in enumerate(sessions)}

    regret_mat = np.full((n_sess, len(policies)), np.nan)
    num_mat = np.full((n_sess, len(policies)), np.nan)
    den_mat = np.full((n_sess, len(policies)), np.nan)

    for _, row in session_metrics.iterrows():
        if row["policy"] not in policies:
            continue
        i = sid_to_i[row["session_id"]]
        j = policies.index(row["policy"])
        regret_mat[i, j] = row["mean_regret"]
        num_mat[i, j] = row["recovery_numerator"]
        den_mat[i, j] = row["recovery_denominator"]

    rng = np.random.default_rng(seed)
    boot_regret = np.zeros((n_boot, len(policies)))
    boot_recovery = np.zeros((n_boot, len(policies)))

    for b in range(n_boot):
        idx = rng.integers(0, n_sess, n_sess)
        boot_regret[b] = np.nanmean(regret_mat[idx], axis=0)
        nums = np.nansum(num_mat[idx], axis=0)
        dens = np.nansum(den_mat[idx], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            boot_recovery[b] = np.where(dens != 0, nums / dens * 100.0, np.nan)

    rows = []
    meta = _meta_cols()
    for j, policy in enumerate(policies):
        point_regret = float(np.nanmean(regret_mat[:, j]))
        nums = float(np.nansum(num_mat[:, j]))
        dens = float(np.nansum(den_mat[:, j]))
        point_rec = nums / dens * 100.0 if dens != 0 else np.nan
        rows.append(
            {
                **meta,
                "policy": policy,
                "n_sessions": n_sess,
                "n_bootstrap": n_boot,
                "bootstrap_seed": seed,
                "mean_regret": point_regret,
                "regret_ci_lo": float(np.nanpercentile(boot_regret[:, j], 2.5)),
                "regret_ci_hi": float(np.nanpercentile(boot_regret[:, j], 97.5)),
                "oracle_recovery_pct": point_rec,
                "recovery_ci_lo": float(np.nanpercentile(boot_recovery[:, j], 2.5)),
                "recovery_ci_hi": float(np.nanpercentile(boot_recovery[:, j], 97.5)),
            }
        )
    return pd.DataFrame(rows)


def lambda_stratified(
    detail: pd.DataFrame,
    n_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Per-λ session aggregates + cluster bootstrap CI on mean regret."""
    rows = []
    meta = _meta_cols()
    for lam, g in detail.groupby("lambda"):
        sess = aggregate_session_level(g)
        boot = cluster_bootstrap_summary(sess, n_boot=n_boot, seed=seed)
        for _, r in boot.iterrows():
            rows.append({**meta, "lambda": float(lam), **r.to_dict()})
    # Drop duplicate meta keys from merge
    return pd.DataFrame(rows)


def dataset_stratified(
    detail: pd.DataFrame,
    n_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Per-dataset (15 sessions) session aggregates + cluster bootstrap."""
    rows = []
    meta = _meta_cols()
    for ds, g in detail.groupby("dataset"):
        sess = aggregate_session_level(g)
        boot = cluster_bootstrap_summary(sess, n_boot=n_boot, seed=seed)
        for _, r in boot.iterrows():
            rows.append({**meta, "dataset": str(ds), **r.to_dict()})
    return pd.DataFrame(rows)


def run_cluster_stats(
    detail: pd.DataFrame,
    out_dir: Path | str,
    n_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, pd.DataFrame]:
    """
    Write all session-level / cluster-aware CSVs under ``out_dir``.

    Files:
      - session_level_metrics.csv
      - cluster_bootstrap_summary.csv
      - paired_session_tests.csv
      - lambda_stratified_results.csv
      - dataset_stratified_results.csv
      - holm_corrected_primary_tests.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _meta_cols()

    session_metrics = aggregate_session_level(detail)
    for k, v in meta.items():
        session_metrics[k] = v
    session_metrics.to_csv(out_dir / "session_level_metrics.csv", index=False)

    boot = cluster_bootstrap_summary(session_metrics, n_boot=n_boot, seed=seed)
    boot.to_csv(out_dir / "cluster_bootstrap_summary.csv", index=False)

    # Paired tests (primary + exploratory)
    paired_rows = []
    comparisons = list(PRIMARY_COMPARISONS) + [
        ("fedpba", "majority", "fedpba_vs_majority"),
        ("greedy", "majority", "greedy_vs_majority"),
        ("fedpba", "context_only", "fedpba_vs_context_only"),
        ("greedy", "context_only", "greedy_vs_context_only"),
    ]
    primary_names = {c[2] for c in PRIMARY_COMPARISONS}
    for a, b, name in comparisons:
        if a not in set(session_metrics["policy"]) or b not in set(session_metrics["policy"]):
            continue
        res = paired_session_test(session_metrics, a, b, n_boot=n_boot, seed=seed)
        paired_rows.append(
            {
                **meta,
                "comparison": name,
                "policy_a": a,
                "policy_b": b,
                "role": "primary" if name in primary_names else "exploratory",
                **res,
            }
        )
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(out_dir / "paired_session_tests.csv", index=False)

    # Holm on primary only
    primary_df = paired_df[paired_df["role"] == "primary"].copy()
    if len(primary_df):
        # Stable order matching PRIMARY_COMPARISONS
        order = [c[2] for c in PRIMARY_COMPARISONS]
        primary_df["comparison"] = pd.Categorical(
            primary_df["comparison"], categories=order, ordered=True
        )
        primary_df = primary_df.sort_values("comparison")
        raw_p = primary_df["wilcoxon_p"].astype(float).tolist()
        adj = holm_correction(raw_p)
        primary_df["holm_adjusted_p"] = adj
        primary_df.to_csv(out_dir / "holm_corrected_primary_tests.csv", index=False)
    else:
        primary_df = pd.DataFrame()
        primary_df.to_csv(out_dir / "holm_corrected_primary_tests.csv", index=False)

    lam_df = lambda_stratified(detail, n_boot=n_boot, seed=seed)
    # Deduplicate meta keys that cluster_bootstrap already embeds
    lam_df.to_csv(out_dir / "lambda_stratified_results.csv", index=False)

    ds_df = dataset_stratified(detail, n_boot=n_boot, seed=seed)
    ds_df.to_csv(out_dir / "dataset_stratified_results.csv", index=False)

    return {
        "session_level_metrics": session_metrics,
        "cluster_bootstrap_summary": boot,
        "paired_session_tests": paired_df,
        "holm_corrected_primary_tests": primary_df,
        "lambda_stratified_results": lam_df,
        "dataset_stratified_results": ds_df,
    }
