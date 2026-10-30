"""
Headline confidence intervals and significance for Paper-1 freeze.

Uses completed CSV outputs and cached client-level files (post-hoc only).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from fedpbg.config import PROJECT_ROOT
from fedpbg.oracle.policy_eval import evaluate_all_policies
from fedpbg.oracle.regret_stats import BASELINE_POLICIES, paired_oracle_test

RESULTS = PROJECT_ROOT / "results"
STATS_DIR = RESULTS / "statistics"
TABLES_DIR = PROJECT_ROOT / "paper" / "tables"
LAMBDA_SPARSITY = 0.3
BOOTSTRAP_N = 10_000
RNG = np.random.default_rng(42)

CACHE_PATTERN = re.compile(
    r"^(?P<dataset>\w+)_s(?P<seed>\d+)_a(?P<alpha>[\d.]+)_n(?P<clients>\d+)_client_level\.csv$"
)


def t_confidence_interval(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float, float]:
    """Return mean, std, ci_low, ci_high (two-sided t interval)."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    if n < 2:
        return mean, std, mean, mean
    sem = std / np.sqrt(n)
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return mean, std, mean - tcrit * sem, mean + tcrit * sem


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(v[0]), float(v[0])
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    return float((centre - margin) / denom), float((centre + margin) / denom)


def summarize_metric(
    values: Iterable[float],
    metric: str,
    scope: str,
    unit: str = "proportion",
    as_percent: bool = False,
) -> Dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if as_percent:
        arr = 100 * arr
    mean, std, t_lo, t_hi = t_confidence_interval(arr)
    b_lo, b_hi = bootstrap_ci(arr)
    return {
        "metric": metric,
        "scope": scope,
        "n": len(arr),
        "mean": mean,
        "std": std,
        "ci_t_low": t_lo,
        "ci_t_high": t_hi,
        "ci_boot_low": b_lo,
        "ci_boot_high": b_hi,
        "unit": unit,
    }


def summarize_win_rate(regret_df: pd.DataFrame, policy: str, scope: str) -> Dict:
    sub = regret_df[regret_df["policy"] == policy]
    wins = (sub["regret"] > 1e-8).astype(int).to_numpy()
    n = len(wins)
    rate = float(np.mean(wins)) if n else float("nan")
    w_lo, w_hi = wilson_ci(int(wins.sum()), n)
    b_lo, b_hi = bootstrap_ci(wins.astype(np.float64))
    return {
        "metric": f"oracle_win_rate_vs_{policy.lower()}",
        "scope": scope,
        "n": n,
        "mean": 100 * rate,
        "std": 100 * float(np.std(wins, ddof=1)) if n > 1 else 0.0,
        "ci_t_low": 100 * w_lo,
        "ci_t_high": 100 * w_hi,
        "ci_boot_low": 100 * b_lo,
        "ci_boot_high": 100 * b_hi,
        "unit": "percent",
    }


def load_sparsity_df() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "oracle_sparsity_validation" / "oracle_sparsity_results.csv")


def load_regret_df() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "final_validation" / "oracle_regret_results.csv")


def build_headline_ci_records() -> pd.DataFrame:
    sp = load_sparsity_df()
    reg = load_regret_df()
    sp_l = sp[np.isclose(sp["lambda"], LAMBDA_SPARSITY)]

    records: List[Dict] = []
    # Sparsity headlines (λ=0.3)
    for col, name, pct in [
        ("p0_ratio", "p0_ratio", False),
        ("deep_ratio", "deep_ratio", False),
        ("oracle_vs_uniform_rel_gap", "oracle_vs_uniform_gap", True),
        ("oracle_vs_personalize_all_rel_gap", "oracle_vs_p5_all_gap", True),
    ]:
        records.append(summarize_metric(sp_l[col], name, "global_sparsity_l0.3", as_percent=pct))
        for ds in sorted(sp_l["dataset"].unique()):
            sub = sp_l[sp_l["dataset"] == ds]
            records.append(summarize_metric(sub[col], name, f"dataset_{ds}_l0.3", as_percent=pct))

    # Regret headlines
    policy_map = {
        "RANDOM": "relative_regret_vs_random",
        "BEST_UNIFORM": "relative_regret_vs_uniform",
        "GLOBAL_ONLY": "relative_regret_vs_global_only",
        "PERSONALIZE_ALL": "relative_regret_vs_personalize_all",
    }
    for pol, name in policy_map.items():
        sub = reg[reg["policy"] == pol]
        records.append(
            summarize_metric(sub["relative_regret"], name, "global_regret", as_percent=True)
        )
        records.append(summarize_win_rate(reg, pol, "global_regret"))
        for ds in sorted(sub["dataset"].unique()):
            dsub = sub[sub["dataset"] == ds]
            records.append(
                summarize_metric(dsub["relative_regret"], name, f"dataset_{ds}_regret", as_percent=True)
            )
            records.append(summarize_win_rate(dsub, pol, f"dataset_{ds}_regret"))

    return pd.DataFrame(records)


def discover_cache_files() -> List[Path]:
    files = []
    for sub in ("sparsity_phase_diagram/cache", "oracle_sparsity_validation/cache"):
        d = RESULTS / sub
        if d.exists():
            files.extend(sorted(d.glob("*_client_level.csv")))
    seen = set()
    unique = []
    for f in files:
        if f.name not in seen:
            seen.add(f.name)
            unique.append(f)
    return unique


def parse_cache_meta(path: Path, client_df: pd.DataFrame) -> dict:
    m = CACHE_PATTERN.match(path.name)
    if m:
        return {
            "dataset": client_df["dataset"].iloc[0] if "dataset" in client_df.columns else m.group("dataset"),
            "seed": int(m.group("seed")),
            "dirichlet_alpha": float(m.group("alpha")),
            "num_clients": int(m.group("clients")),
        }
    return {
        "dataset": client_df["dataset"].iloc[0],
        "seed": int(client_df["seed"].iloc[0]),
        "dirichlet_alpha": float(client_df["dirichlet_alpha"].iloc[0]),
        "num_clients": int(client_df["num_clients"].iloc[0]),
    }


def cohens_dz(oracle_utils: List[float], policy_utils: List[float]) -> float:
    diff = np.asarray(oracle_utils) - np.asarray(policy_utils)
    if len(diff) < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def compute_significance_package() -> Tuple[pd.DataFrame, List[Dict]]:
    """Wilcoxon + effect size per config cell from cached client-level CSVs."""
    lambdas = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
    cell_rows = []
    for path in discover_cache_files():
        client_df = pd.read_csv(path)
        meta = parse_cache_meta(path, client_df)
        for lam in lambdas:
            _, results = evaluate_all_policies(client_df, lam, meta["seed"], meta)
            oracle_u = results["ORACLE"]["client_utilities"]
            for baseline in BASELINE_POLICIES:
                policy_u = results[baseline]["client_utilities"]
                stat, pval = paired_oracle_test(oracle_u, policy_u)
                dz = cohens_dz(oracle_u, policy_u)
                cell_rows.append(
                    {
                        **meta,
                        "lambda": lam,
                        "baseline": baseline,
                        "wilcoxon_stat": stat,
                        "p_value": pval,
                        "cohens_dz": dz,
                        "significant_0.05": pval < 0.05,
                        "oracle_win": results["ORACLE"]["mean_utility"]
                        > results[baseline]["mean_utility"] + 1e-8,
                    }
                )
    cells = pd.DataFrame(cell_rows)
    return cells, _aggregate_significance(cells)


def _aggregate_significance(cells: pd.DataFrame) -> List[Dict]:
    agg = []
    for baseline in BASELINE_POLICIES:
        sub = cells[cells["baseline"] == baseline]
        pvals = sub["p_value"].to_numpy()
        dz = sub["cohens_dz"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        agg.append(
            {
                "baseline": baseline,
                "n_cells": len(sub),
                "n_significant_p05": int((pvals < 0.05).sum()),
                "fraction_significant": float((pvals < 0.05).mean()),
                "median_p_value": float(np.median(pvals)),
                "mean_cohens_dz": float(np.mean(dz)) if len(dz) else float("nan"),
                "median_cohens_dz": float(np.median(dz)) if len(dz) else float("nan"),
            }
        )
    return agg


def write_ci_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Headline Confidence Intervals",
        "",
        "Two-sided 95% intervals: **t-based** (across configuration cells) and **bootstrap** (10k resamples of cell means).",
        "Win rates use Wilson score (t columns) and bootstrap on binary win indicators.",
        "",
        "| Metric | Scope | n | Mean | Std | 95% CI (t) | 95% CI (boot) |",
        "|--------|-------|---|------|-----|------------|---------------|",
    ]
    global_rows = df[df["scope"].str.startswith("global")]
    for _, r in global_rows.iterrows():
        unit = "%" if r["unit"] == "percent" or "regret" in r["metric"] or "gap" in r["metric"] or "win" in r["metric"] else ""
        lines.append(
            f"| {r['metric']} | {r['scope']} | {int(r['n'])} | "
            f"{r['mean']:.2f}{unit} | {r['std']:.2f}{unit} | "
            f"[{r['ci_t_low']:.2f}, {r['ci_t_high']:.2f}] | "
            f"[{r['ci_boot_low']:.2f}, {r['ci_boot_high']:.2f}] |"
        )
    lines.extend(["", "## Per-dataset (λ=0.3 sparsity)", ""])
    ds_rows = df[df["scope"].str.contains("_l0.3") & ~df["scope"].str.startswith("global")]
    for _, r in ds_rows.iterrows():
        unit = "%" if "gap" in r["metric"] else ""
        lines.append(
            f"- **{r['scope']}** / {r['metric']}: {r['mean']:.3f}{unit} "
            f"(std {r['std']:.3f}, boot CI [{r['ci_boot_low']:.3f}, {r['ci_boot_high']:.3f}])"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_significance_markdown(agg: List[Dict], cells: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Significance Summary — Oracle vs Fixed Policies",
        "",
        "Per-cell paired **Wilcoxon signed-rank** test (client-level utilities, one-sided oracle > baseline).",
        f"Total cells per baseline: {agg[0]['n_cells']} (99 configs × 6 λ).",
        "",
        "| Baseline | Significant cells (p<0.05) | Fraction | Median p | Mean Cohen's d_z | Median Cohen's d_z |",
        "|----------|--------------------------|----------|----------|------------------|-------------------|",
    ]
    for row in agg:
        lines.append(
            f"| {row['baseline']} | {row['n_significant_p05']} / {row['n_cells']} | "
            f"{100*row['fraction_significant']:.1f}% | {row['median_p_value']:.4g} | "
            f"{row['mean_cohens_dz']:.3f} | {row['median_cohens_dz']:.3f} |"
        )
    lines.extend(["", "## Interpretation notes", ""])
    for row in agg:
        strength = "strong" if row["median_cohens_dz"] >= 0.8 else "moderate" if row["median_cohens_dz"] >= 0.5 else "small"
        lines.append(
            f"- **{row['baseline']}:** {strength} median effect (d_z={row['median_cohens_dz']:.2f}); "
            f"{100*row['fraction_significant']:.0f}% of cells significant at α=0.05."
        )
    # Per-dataset uniform
    lines.extend(["", "## Per-dataset — BEST_UNIFORM (fraction significant)", ""])
    uni = cells[cells["baseline"] == "BEST_UNIFORM"]
    for ds in sorted(uni["dataset"].unique()):
        sub = uni[uni["dataset"] == ds]
        frac = (sub["p_value"] < 0.05).mean()
        lines.append(f"- **{ds}:** {100*frac:.0f}% significant ({int((sub['p_value']<0.05).sum())}/{len(sub)} cells)")
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_pm(mean: float, std: float, pct: bool = True) -> str:
    if pct:
        return f"{mean:.1f} $\\pm$ {std:.1f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def _fmt_ci(lo: float, hi: float, pct: bool = True) -> str:
    if pct:
        return f"[{lo:.1f}, {hi:.1f}]"
    return f"[{lo:.3f}, {hi:.3f}]"


def write_latex_tables(ci_df: pd.DataFrame, reg: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Main regret table
    lines = [
        "% Auto-generated — headline regret with 95% bootstrap CI",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Oracle regret vs.\\ fixed policies (mean $\\pm$ std over configuration--$\\lambda$ cells; 95\\% bootstrap CI).}",
        "\\label{tab:main_regret}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Baseline & Rel.\\ regret (\\%) & Oracle win rate (\\%) & 95\\% CI (boot) \\\\",
        "\\midrule",
    ]
    pol_labels = {
        "relative_regret_vs_random": "Random",
        "relative_regret_vs_uniform": "Best uniform",
        "relative_regret_vs_global_only": "Global-only",
        "relative_regret_vs_personalize_all": "Personalize-all",
    }
    for key, label in pol_labels.items():
        r = ci_df[(ci_df["metric"] == key) & (ci_df["scope"] == "global_regret")].iloc[0]
        pol = {
            "relative_regret_vs_random": "RANDOM",
            "relative_regret_vs_uniform": "BEST_UNIFORM",
            "relative_regret_vs_global_only": "GLOBAL_ONLY",
            "relative_regret_vs_personalize_all": "PERSONALIZE_ALL",
        }[key]
        wr = ci_df[
            (ci_df["metric"] == f"oracle_win_rate_vs_{pol.lower()}") & (ci_df["scope"] == "global_regret")
        ].iloc[0]
        lines.append(
            f"{label} & {_fmt_pm(r['mean'], r['std'])} & {_fmt_pm(wr['mean'], wr['std'])} & "
            f"{_fmt_ci(r['ci_boot_low'], r['ci_boot_high'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (TABLES_DIR / "table_main_regret.tex").write_text("\n".join(lines), encoding="utf-8")

    # Dataset breakdown
    lines = [
        "% Auto-generated — per-dataset regret vs best uniform",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-dataset regret vs.\\ best uniform (mean $\\pm$ std; 95\\% bootstrap CI).}",
        "\\label{tab:dataset_breakdown}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Dataset & Rel.\\ regret (\\%) & Win rate (\\%) & 95\\% CI (boot) \\\\",
        "\\midrule",
    ]
    for ds in sorted(reg["dataset"].unique()):
        r = ci_df[
            (ci_df["metric"] == "relative_regret_vs_uniform")
            & (ci_df["scope"] == f"dataset_{ds}_regret")
        ]
        w = ci_df[
            (ci_df["metric"] == "oracle_win_rate_vs_best_uniform")
            & (ci_df["scope"] == f"dataset_{ds}_regret")
        ]
        if r.empty:
            continue
        r = r.iloc[0]
        w = w.iloc[0]
        label = ds.replace("_", "\\_")
        lines.append(
            f"{label} & {_fmt_pm(r['mean'], r['std'])} & {_fmt_pm(w['mean'], w['std'])} & "
            f"{_fmt_ci(r['ci_boot_low'], r['ci_boot_high'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (TABLES_DIR / "table_dataset_breakdown.tex").write_text("\n".join(lines), encoding="utf-8")

    # PTB-XL breakdown
    lines = [
        "% Auto-generated — real PTB-XL breakdown at lambda=0.3",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Real PTB-XL oracle metrics at $\\lambda=0.3$ (mean $\\pm$ std over seeds/$\\alpha$/client counts).}",
        "\\label{tab:ptbxl_breakdown}",
        "\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Metric & Mean $\\pm$ std & 95\\% CI (boot) \\\\",
        "\\midrule",
    ]
    ptb_metrics = [
        ("p0_ratio", "P0 ratio", False),
        ("deep_ratio", "Deep ratio", False),
        ("oracle_vs_uniform_gap", "Oracle vs uniform (rel \\%)", True),
        ("oracle_vs_p5_all_gap", "Oracle vs P5-all (rel \\%)", True),
    ]
    for metric_key, label, pct in ptb_metrics:
        r = ci_df[(ci_df["scope"] == "dataset_ptbxl_l0.3") & (ci_df["metric"] == metric_key)].iloc[0]
        if pct:
            val = f"{r['mean']:.1f} $\\pm$ {r['std']:.1f}"
        else:
            val = f"{r['mean']:.3f} $\\pm$ {r['std']:.3f}"
        ci = _fmt_ci(r["ci_boot_low"], r["ci_boot_high"], pct=pct)
        lines.append(f"{label} & {val} & {ci} \\\\")
    # regret row
    r = ci_df[
        (ci_df["metric"] == "relative_regret_vs_uniform") & (ci_df["scope"] == "dataset_ptbxl_regret")
    ].iloc[0]
    lines.append(
        f"Regret vs uniform (all $\\lambda$) & {_fmt_pm(r['mean'], r['std'])} & "
        f"{_fmt_ci(r['ci_boot_low'], r['ci_boot_high'])} \\\\"
    )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (TABLES_DIR / "table_ptbxl_breakdown.tex").write_text("\n".join(lines), encoding="utf-8")


def run_all() -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    ci_df = build_headline_ci_records()
    ci_df.to_csv(STATS_DIR / "headline_confidence_intervals.csv", index=False)
    write_ci_markdown(ci_df, STATS_DIR / "headline_confidence_intervals.md")

    cells, sig_agg = compute_significance_package()
    cells.to_csv(STATS_DIR / "significance_cells.csv", index=False)
    write_significance_markdown(sig_agg, cells, STATS_DIR / "significance_summary.md")

    reg = load_regret_df()
    write_latex_tables(ci_df, reg)
    print(f"[OK] {STATS_DIR}")
    print(f"[OK] {TABLES_DIR}")


if __name__ == "__main__":
    run_all()
