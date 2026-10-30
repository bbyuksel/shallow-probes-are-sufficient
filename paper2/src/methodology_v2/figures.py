"""Regenerate key figures from methodology-v2 results (no fabricated numbers)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper2.src.methodology_v2 import PROTOCOL

POLICY_LABELS = {
    "oracle": "Oracle",
    "fedpba": "FedPBA",
    "best_uniform": "Best uniform",
    "majority": "Majority",
    "context_only": "Context-only",
    "greedy": "Greedy probe",
    "greedy_probe": "Greedy probe",
}


def _label(p: str) -> str:
    return POLICY_LABELS.get(p, p)


def _submission_figures_dir() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "submission" / "figures"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_policy_regret_ci(
    boot: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Mean regret with session-cluster bootstrap 95% CI."""
    order = ["greedy", "fedpba", "best_uniform", "context_only", "majority"]
    sub = boot[boot["policy"].isin(order)].copy()
    sub["policy"] = pd.Categorical(sub["policy"], categories=order, ordered=True)
    sub = sub.dropna(subset=["mean_regret"]).sort_values("policy")
    if sub.empty:
        raise ValueError("No policies available for regret CI figure")

    names = [_label(p) for p in sub["policy"]]
    y = sub["mean_regret"].values.astype(float)
    lo = sub["regret_ci_lo"].values.astype(float)
    hi = sub["regret_ci_hi"].values.astype(float)
    err = np.vstack([y - lo, hi - y])

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(names))
    ax.bar(
        x,
        y,
        yerr=err,
        capsize=3,
        color="#4c78a8",
        edgecolor="black",
        linewidth=0.5,
        ecolor="black",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Mean regret vs. Oracle")
    ax.set_title(f"Session-cluster bootstrap CI ({PROTOCOL})")
    ax.axhline(0, color="gray", lw=0.8)
    _save(fig, out_path)
    return out_path


def fig_oracle_recovery(
    boot: pd.DataFrame,
    out_path: Path,
) -> Path:
    order = ["greedy", "fedpba", "best_uniform", "context_only"]
    sub = boot[boot["policy"].isin(order)].copy()
    sub["policy"] = pd.Categorical(sub["policy"], categories=order, ordered=True)
    sub = sub.dropna(subset=["oracle_recovery_pct"]).sort_values("policy")
    if sub.empty:
        raise ValueError("No policies available for recovery figure")

    names = [_label(p) for p in sub["policy"]]
    rec = sub["oracle_recovery_pct"].values.astype(float)
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.bar(names, rec, color="#54a24b", edgecolor="black", linewidth=0.5)
    ax.axhline(100, color="black", ls="--", lw=1, label="Oracle (100%)")
    ax.set_ylabel("Oracle utility recovery (%)")
    ax.set_title("Utility recovery (session-cluster)")
    ax.set_ylim(min(0, float(np.nanmin(rec)) - 5), 110)
    ax.legend(frameon=False)
    for i, v in enumerate(rec):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    _save(fig, out_path)
    return out_path


def fig_policy_class_regret(
    boot: pd.DataFrame,
    out_path: Path,
) -> Path:
    order = ["greedy", "fedpba", "best_uniform", "context_only", "majority"]
    sub = boot[boot["policy"].isin(order)].copy()
    sub["policy"] = pd.Categorical(sub["policy"], categories=order, ordered=True)
    sub = sub.dropna(subset=["mean_regret"]).sort_values("policy", ascending=False)
    names = [_label(p) for p in sub["policy"]]
    vals = sub["mean_regret"].values.astype(float)
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.barh(names, vals, color="#4c78a8", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Mean regret vs. Oracle")
    ax.set_title("Policy mean regret (grouped CV / session stats)")
    for y, v in enumerate(vals):
        ax.text(v + 0.002, y, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.2 if len(vals) else 1)
    _save(fig, out_path)
    return out_path


def fig_paired_effects(
    holm: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Primary paired Δutility with session-cluster CI."""
    if holm is None or holm.empty:
        raise ValueError("No Holm/primary tests for paired-effects figure")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    names = holm["comparison"].astype(str).tolist()
    y = holm["mean_diff"].astype(float).values
    lo = holm["diff_ci_lo"].astype(float).values
    hi = holm["diff_ci_hi"].astype(float).values
    err = np.vstack([y - lo, hi - y])
    x = np.arange(len(names))
    ax.bar(x, y, yerr=err, capsize=3, color="#e45756", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Mean session utility difference")
    ax.set_title("Primary paired comparisons (n=sessions)")
    _save(fig, out_path)
    return out_path


def generate_figures(
    cluster_tables: Dict[str, pd.DataFrame],
    out_dir: Path | str,
    also_copy_to_submission: bool = True,
) -> Dict[str, Path]:
    """
    Write PNGs under ``out_dir/figures`` and optionally copy to
    ``SINCONF/submission/figures``.
    """
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    boot = cluster_tables["cluster_bootstrap_summary"]
    paths: Dict[str, Path] = {}

    paths["regret_ci"] = fig_policy_regret_ci(
        boot, fig_dir / "phase4_regret_ci_by_policy.png"
    )
    paths["recovery"] = fig_oracle_recovery(
        boot, fig_dir / "oracle_utility_recovery.png"
    )
    paths["class_regret"] = fig_policy_class_regret(
        boot, fig_dir / "phase6_policy_class_regret.png"
    )

    holm = cluster_tables.get("holm_corrected_primary_tests")
    if holm is not None and len(holm):
        paths["paired"] = fig_paired_effects(
            holm, fig_dir / "session_paired_primary_comparisons.png"
        )

    # Also write oracle_recovery_comparison alias used in phase5
    paths["recovery_comparison"] = fig_oracle_recovery(
        boot, fig_dir / "oracle_recovery_comparison.png"
    )

    if also_copy_to_submission:
        dest = _submission_figures_dir()
        dest.mkdir(parents=True, exist_ok=True)
        for p in paths.values():
            shutil.copy2(p, dest / p.name)

    return paths
