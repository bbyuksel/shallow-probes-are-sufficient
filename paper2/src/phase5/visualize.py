"""Phase-5 figures for contextual policy comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper2.src.config import FIGURES_DIR
from paper2.src.phase5.contextual_policies import POLICY_NAMES


POLICY_LABELS = {
    "majority": "Majority",
    "uniform_depth": "Uniform depth",
    "multinomial_logistic": "Multinomial logit",
    "cost_sensitive_logistic": "Cost-sensitive logit",
    "greedy_probe": "Greedy probe",
    "contextual_tree": "Decision tree",
    "contextual_random_forest": "Random forest",
    "fedpba": "FedPBA",
}


def _label(p: str) -> str:
    return POLICY_LABELS.get(p, p)


def generate_phase5_figures(
    policy_stats: pd.DataFrame,
    ctx_probe: Dict[str, float],
    out_dir: Path | None = None,
) -> Dict[str, Path]:
    out_dir = out_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    stats = policy_stats.set_index("policy").reindex(POLICY_NAMES).dropna(subset=["mean_regret"])
    labels = [_label(p) for p in stats.index]
    regrets = stats["mean_regret"].values
    recovery = stats["oracle_utility_recovery_pct"].values

    # 1. Policy comparison (regret bar chart)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4c72b0" if p != "fedpba" else "#c44e52" for p in stats.index]
    ax.barh(labels, regrets, color=colors)
    ax.set_xlabel("Mean regret vs. oracle")
    ax.set_title("Contextual policy comparison (Phase-5)")
    ax.invert_yaxis()
    fig.tight_layout()
    p1 = out_dir / "contextual_policy_comparison.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths["comparison"] = p1

    # 2. Oracle recovery
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, recovery, color=colors)
    ax.axvline(100.0, color="gray", linestyle="--", linewidth=1, label="Oracle (100%)")
    ax.set_xlabel("Oracle utility recovery (%)")
    ax.set_title("Oracle utility recovery by policy")
    ax.invert_yaxis()
    fig.tight_layout()
    p2 = out_dir / "oracle_recovery_comparison.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths["recovery"] = p2

    # 3. Context vs probe value
    fig, ax = plt.subplots(figsize=(6, 4))
    best_ctx = ctx_probe["best_context_policy"]
    groups = ["Best context-only\n(" + _label(best_ctx) + ")", "Greedy probe", "FedPBA\n(context+probes)"]
    ctx_r = ctx_probe["context_mean_regret"]
    greedy_r = float(stats.loc["greedy_probe", "mean_regret"]) if "greedy_probe" in stats.index else np.nan
    fed_r = float(stats.loc["fedpba", "mean_regret"])
    vals = [ctx_r, greedy_r, fed_r]
    bar_colors = ["#4c72b0", "#dd8452", "#c44e52"]
    ax.bar(groups, vals, color=bar_colors)
    ax.set_ylabel("Mean regret vs. oracle")
    ax.set_title("Context-only vs. probe-informed policies")
    fig.tight_layout()
    p3 = out_dir / "context_vs_probe_value.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths["context_probe"] = p3

    # 4. Regret CI
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(stats))
    lo = stats["regret_ci_lo"].values
    hi = stats["regret_ci_hi"].values
    ax.barh(y, regrets, xerr=[regrets - lo, hi - regrets], color=colors, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean regret vs. oracle (95% bootstrap CI)")
    ax.set_title("Contextual policy regret with confidence intervals")
    ax.invert_yaxis()
    fig.tight_layout()
    p4 = out_dir / "contextual_policy_regret_ci.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    paths["regret_ci"] = p4

    return paths
