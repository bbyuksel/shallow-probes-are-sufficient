"""
Post-hoc Ditto regret analysis and positioning reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from fedpbg.oracle.regret_stats import paired_oracle_test

LAMBDAS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
BOOTSTRAP_N = 10_000
RNG = np.random.default_rng(42)


def bootstrap_ci(values: np.ndarray, n_boot: int = BOOTSTRAP_N) -> Tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n < 2:
        return float(v[0]), float(v[0])
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohens_dz_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if len(diff) < 2 or np.std(diff, ddof=1) < 1e-12:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def build_ditto_vs_oracle_rows(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per config x lambda with utilities and regrets."""
    rows = []
    for _, r in comparison_df.iterrows():
        oracle_u = r["oracle_utility"]
        for policy, col in [
            ("DITTO", "ditto_utility"),
            ("BEST_UNIFORM", "best_uniform_utility"),
            ("GLOBAL_ONLY", "global_only_utility"),
        ]:
            pu = r[col]
            regret = oracle_u - pu
            rel = regret / max(abs(oracle_u), 0.05, 1e-8)
            rows.append(
                {
                    **{k: r[k] for k in [
                        "dataset", "dataset_requested", "seed", "dirichlet_alpha",
                        "num_clients", "lambda",
                    ]},
                    "policy": policy,
                    "policy_utility": pu,
                    "oracle_utility": oracle_u,
                    "regret": regret,
                    "relative_regret": rel,
                    "oracle_wins": regret > 1e-8,
                }
            )
    return pd.DataFrame(rows)


def regret_summary_for_policy(
    regret_df: pd.DataFrame,
    policy: str,
) -> Dict:
    sub = regret_df[regret_df["policy"] == policy]
    rel = sub["relative_regret"].to_numpy()
    mean = float(np.mean(rel))
    std = float(np.std(rel, ddof=1)) if len(rel) > 1 else 0.0
    b_lo, b_hi = bootstrap_ci(rel)
    wins = sub["oracle_wins"].astype(int).to_numpy()
    win_rate = float(np.mean(wins)) if len(wins) else 0.0
    return {
        "policy": policy,
        "n_cells": len(sub),
        "mean_rel_regret_pct": 100 * mean,
        "std_rel_regret_pct": 100 * std,
        "ci_boot_low_pct": 100 * b_lo,
        "ci_boot_high_pct": 100 * b_hi,
        "oracle_win_rate_pct": 100 * win_rate,
    }


def wilcoxon_ditto_vs_oracle(comparison_df: pd.DataFrame) -> Dict:
    """Aggregate Wilcoxon: oracle vs Ditto across cells (cell-level mean utilities)."""
    pvals = []
    dzs = []
    for _, r in comparison_df.iterrows():
        # Cell-level: compare scalar oracle mean vs ditto mean
        o = r["oracle_utility"]
        d = r["ditto_utility"]
        if abs(o - d) > 1e-10:
            _, p = stats.wilcoxon([o], [d], alternative="greater")
            pvals.append(p)
    # Per-cell client-level from stored utilities if available
    return {"note": "cell-level scalar comparison", "n": len(comparison_df)}


def positioning_analysis(comparison_df: pd.DataFrame) -> Dict:
    """
    Answer: is Ditto closer to Oracle, Best Uniform, or Global-only?

    Uses mean L1 distance in utility space across all cells.
    """
    d_oracle = np.abs(comparison_df["ditto_utility"] - comparison_df["oracle_utility"])
    d_uniform = np.abs(comparison_df["ditto_utility"] - comparison_df["best_uniform_utility"])
    d_global = np.abs(comparison_df["ditto_utility"] - comparison_df["global_only_utility"])

    mean_d = {
        "oracle": float(d_oracle.mean()),
        "best_uniform": float(d_uniform.mean()),
        "global_only": float(d_global.mean()),
    }
    closest = min(mean_d, key=mean_d.get)

    # Win head-to-head vs baselines (fraction of cells where Ditto utility > baseline)
    vs_uniform = float((comparison_df["ditto_utility"] > comparison_df["best_uniform_utility"]).mean())
    vs_global = float((comparison_df["ditto_utility"] > comparison_df["global_only_utility"]).mean())
    vs_oracle = float((comparison_df["ditto_utility"] >= comparison_df["oracle_utility"] - 1e-8).mean())

    return {
        "mean_utility_distance": mean_d,
        "closest_baseline": closest,
        "ditto_beats_uniform_fraction": vs_uniform,
        "ditto_beats_global_fraction": vs_global,
        "ditto_matches_oracle_fraction": vs_oracle,
    }


def plot_ditto_positioning(comparison_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    policies = ["GLOBAL_ONLY", "BEST_UNIFORM", "DITTO", "ORACLE"]
    colors = ["#95a5a6", "#3498db", "#e74c3c", "#2ecc71"]
    for lam in [0.0, 0.3, 1.0]:
        sub = comparison_df[comparison_df["lambda"] == lam]
        rel = []
        for pol, col in [
            ("GLOBAL_ONLY", "global_only_utility"),
            ("BEST_UNIFORM", "best_uniform_utility"),
            ("DITTO", "ditto_utility"),
        ]:
            reg = (sub["oracle_utility"] - sub[col]) / sub["oracle_utility"].abs().clip(lower=0.05)
            rel.append(100 * reg.mean())
        rel.append(0.0)
        ax.plot(policies, rel, marker="o", label=f"λ={lam}")

    ax.set_ylabel("Mean relative regret vs oracle (%)")
    ax.set_title("Ditto vs fixed policies under U")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ditto_regret_comparison.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    lam03 = comparison_df[comparison_df["lambda"] == 0.3]
    ax2.scatter(
        lam03["best_uniform_utility"],
        lam03["ditto_utility"],
        alpha=0.5,
        s=20,
        label="Ditto vs Best Uniform",
    )
    lims = [
        min(lam03["best_uniform_utility"].min(), lam03["ditto_utility"].min()),
        max(lam03["best_uniform_utility"].max(), lam03["ditto_utility"].max()),
    ]
    ax2.plot(lims, lims, "k--", alpha=0.5)
    ax2.set_xlabel("Best uniform utility")
    ax2.set_ylabel("Ditto utility")
    ax2.set_title("λ=0.3: Ditto vs best uniform")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "ditto_vs_uniform_scatter.png", dpi=150)
    plt.close(fig2)


def write_regret_summary_md(
    regret_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    sig_records: List[Dict],
    path: Path,
) -> None:
    lines = [
        "# Ditto Regret Summary",
        "",
        "Ditto evaluated under $U = \\mathrm{Macro\\text{-}F1} - \\lambda \\cdot \\mathrm{Cost}$ "
        "(P5-normalized cost from oracle cache; Ditto trains full model).",
        "",
        "## Relative regret vs oracle (all λ cells)",
        "",
        "| Policy | Mean rel. regret (%) | Std (%) | 95% boot CI | Oracle win rate (%) |",
        "|--------|---------------------|---------|-------------|---------------------|",
    ]
    for policy in ["DITTO", "BEST_UNIFORM", "GLOBAL_ONLY", "RANDOM", "PERSONALIZE_ALL"]:
        if policy in ("RANDOM", "PERSONALIZE_ALL"):
            continue
        s = regret_summary_for_policy(regret_df, policy)
        lines.append(
            f"| {policy} | {s['mean_rel_regret_pct']:.1f} | {s['std_rel_regret_pct']:.1f} | "
            f"[{s['ci_boot_low_pct']:.1f}, {s['ci_boot_high_pct']:.1f}] | {s['oracle_win_rate_pct']:.1f} |"
        )

    lines.extend(["", "## Wilcoxon (oracle > policy, client-level per cell)", ""])
    if sig_records:
        lines.append("| Policy | Sig. cells | Fraction | Median p | Median d_z |")
        lines.append("|--------|------------|----------|----------|------------|")
        for rec in sig_records:
            lines.append(
                f"| {rec['policy']} | {rec['sig']}/{rec['n']} | {100*rec['frac']:.1f}% | "
                f"{rec['median_p']:.4g} | {rec['median_dz']:.3f} |"
            )

    pos = positioning_analysis(comparison_df)
    lines.extend([
        "",
        "## Positioning preview",
        "",
        f"- Closest baseline by mean |ΔU|: **{pos['closest_baseline']}**",
        f"- Ditto beats best uniform: {100*pos['ditto_beats_uniform_fraction']:.1f}% of cells",
        f"- Ditto beats global-only: {100*pos['ditto_beats_global_fraction']:.1f}% of cells",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_positioning_report_md(pos: Dict, comparison_df: pd.DataFrame, path: Path) -> None:
    closest = pos["closest_baseline"]
    label_map = {
        "oracle": "A) Oracle",
        "best_uniform": "B) Best Uniform",
        "global_only": "C) Global-only",
    }
    answer = label_map.get(closest, closest)

    lines = [
        "# Ditto Positioning Report",
        "",
        "## Key question",
        "",
        "Does Ditto behave closer to Oracle, Best Uniform, or Global-only under utility U?",
        "",
        f"## Answer: **{answer}**",
        "",
        "Evidence (mean absolute utility distance across all config–λ cells):",
        "",
        f"| Baseline | Mean |ΔU| vs Ditto |",
        "|----------|-------------------|",
        f"| Oracle | {pos['mean_utility_distance']['oracle']:.4f} |",
        f"| Best uniform | {pos['mean_utility_distance']['best_uniform']:.4f} |",
        f"| Global-only | {pos['mean_utility_distance']['global_only']:.4f} |",
        "",
        "## Head-to-head win rates (Ditto utility > baseline)",
        "",
        f"- vs best uniform: **{100*pos['ditto_beats_uniform_fraction']:.1f}%**",
        f"- vs global-only: **{100*pos['ditto_beats_global_fraction']:.1f}%**",
        f"- vs oracle (ties allowed): **{100*pos['ditto_matches_oracle_fraction']:.1f}%**",
        "",
        "## Per-dataset (λ=0.3)",
        "",
    ]
    lam03 = comparison_df[comparison_df["lambda"] == 0.3]
    for ds in sorted(lam03["dataset"].unique()):
        sub = lam03[lam03["dataset"] == ds]
        d_u = np.abs(sub["ditto_utility"] - sub["best_uniform_utility"]).mean()
        d_o = np.abs(sub["ditto_utility"] - sub["oracle_utility"]).mean()
        lines.append(f"- **{ds}**: |ΔU| to uniform {d_u:.4f}, to oracle {d_o:.4f}")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Ditto is a published PFL trainer (full local personalization with proximal regularization). "
        "Under Paper-1's utility U with P5-equivalent cost, positioning relative to the oracle ladder "
        "indicates whether 'how to personalize' (Ditto) aligns with the utility-optimal allocation "
        "identified by the oracle.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
