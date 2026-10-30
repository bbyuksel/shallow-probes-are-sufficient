"""Phase-4 statistical analysis utilities."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def bootstrap_ci_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return np.nan, np.nan
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        diffs.append((a[idx] - b[idx]).mean())
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def paired_comparison(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Paired comparison: a vs b (e.g. FedPBA vs best uniform)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {}
    diff = a - b
    wilcoxon = stats.wilcoxon(a, b, alternative="greater", zero_method="wilcox")
    ttest = stats.ttest_rel(a, b, alternative="greater")
    dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
    ci = bootstrap_ci_mean_diff(a, b)
    return {
        "n_pairs": int(len(a)),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "win_rate": float((diff > 0).mean()),
        "wilcoxon_stat": float(wilcoxon.statistic),
        "wilcoxon_p": float(wilcoxon.pvalue),
        "ttest_stat": float(ttest.statistic),
        "ttest_p": float(ttest.pvalue),
        "cohens_dz": dz,
        "diff_ci_lo": ci[0],
        "diff_ci_hi": ci[1],
    }


def policy_metrics(detail: pd.DataFrame, policy: str) -> Dict[str, float]:
    util_col = {
        "oracle": "oracle_utility",
        "fedpba": "fedpba_utility",
        "best_uniform": "best_uniform_utility",
        "global_only": "global_only_utility",
        "personalize_all": "p5_all_utility",
        "ditto": "ditto_utility",
    }.get(policy)
    if not util_col or util_col not in detail.columns:
        return {}
    u = detail[util_col].astype(float)
    oracle_u = detail["oracle_utility"].astype(float)
    regret = oracle_u - u
    rel = regret / np.maximum(np.abs(oracle_u), 0.05)
    ci_u = bootstrap_ci(u.values)
    ci_r = bootstrap_ci(regret.values)
    out = {
        "mean_utility": float(u.mean()),
        "std_utility": float(u.std(ddof=1)),
        "utility_ci_lo": ci_u[0],
        "utility_ci_hi": ci_u[1],
        "mean_regret": float(regret.mean()),
        "std_regret": float(regret.std(ddof=1)),
        "regret_ci_lo": ci_r[0],
        "regret_ci_hi": ci_r[1],
        "mean_relative_regret": float(rel.mean()),
        "n": int(u.notna().sum()),
    }
    if policy == "fedpba":
        out["action_accuracy"] = float((detail["fedpba_action"] == detail["oracle_action"]).mean())
    return out


def enrich_detail_metrics(detail: pd.DataFrame, session_dir) -> pd.DataFrame:
    """Add macro_f1 and cost for FedPBA actions from session caches."""
    from pathlib import Path

    from paper2.src.online.policies import _level_lookup
    from paper2.src.online.session import load_session

    session_dir = Path(session_dir)
    rows = []
    for _, r in detail.iterrows():
        path = session_dir / f"{r['config_id']}_client_level.csv"
        if not path.exists():
            rows.append(r.to_dict())
            continue
        sess = load_session(path)
        grp = sess.client_level_df[sess.client_level_df["client_id"] == r["client_id"]]
        if grp.empty:
            rows.append(r.to_dict())
            continue
        f1, cost = _level_lookup(grp, r["fedpba_action"])
        row = r.to_dict()
        row["fedpba_macro_f1"] = f1
        row["fedpba_cost"] = cost
        rows.append(row)
    return pd.DataFrame(rows)
