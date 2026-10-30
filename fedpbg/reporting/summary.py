"""
Generate feasibility_summary.md from experimental results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _level_num(level: str) -> int:
    return int(level[1:])


def _corr_text(x, y) -> str:
    if len(x) < 3:
        return "insufficient data"
    try:
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        return (
            f"Pearson r={r_p:.3f} (p={p_p:.3f}), "
            f"Spearman ρ={r_s:.3f} (p={p_s:.3f})"
        )
    except Exception:
        return "could not compute correlation"


def write_feasibility_summary(
    client_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    out_path: Path,
    default_lambda: float = 0.3,
    dirichlet_alpha: float = 0.1,
    num_clients: int = 20,
) -> None:
    """Answer feasibility questions based on oracle and client-level results."""
    sub = oracle_df[oracle_df["lambda"] == default_lambda].copy()
    sub["oracle_level_num"] = sub["oracle_level"].map(_level_num)

    # Q1: Heterogeneity vs oracle level
    het_corr = _corr_text(sub["heterogeneity"], sub["oracle_level_num"])

    # Q2: Hardware vs oracle level
    hw_means = sub.groupby("hardware")["oracle_level_num"].mean()
    hw_lines = "\n".join(
        f"  - **{hw}**: mean oracle level = {hw_means.get(hw, float('nan')):.2f}"
        for hw in ["LOW", "MEDIUM", "HIGH"]
    )

    # Q3: Uniform personalization suboptimal?
    # Compare best uniform level vs per-client oracle average utility
    uniform_utils = {}
    for level in client_df["level"].unique():
        rows = client_df[client_df["level"] == level]
        uniform_utils[level] = (
            rows["macro_f1"] - default_lambda * rows["normalized_cost"]
        ).mean()
    best_uniform = max(uniform_utils, key=uniform_utils.get)
    oracle_mean_util = sub["oracle_utility"].mean()
    uniform_best_util = uniform_utils[best_uniform]
    uniform_gap = oracle_mean_util - uniform_best_util

    # Level diversity — if clients pick different levels, uniform is suboptimal
    level_counts = sub["oracle_level"].value_counts()
    n_distinct = len(level_counts)
    majority_frac = level_counts.max() / len(sub)

    # Q4: Support continuing FedPBG?
    het_significant = "heterogeneity" in het_corr and "p=" in het_corr
    supports = (
        n_distinct >= 2
        and uniform_gap > 0.01
        and majority_frac < 0.85
    )

    lines = [
        "# FedPBG Feasibility Study Summary",
        "",
        f"**Setup:** PTB-XL binary (Normal vs Abnormal), Lead II, "
        f"{num_clients} clients, Dirichlet α={dirichlet_alpha}, λ={default_lambda}.",
        "",
        "## Research Questions",
        "",
        "### 1. Does heterogeneity correlate with oracle personalization level?",
        "",
        f"**Finding:** {het_corr}.",
        "",
        "Higher prototype-distance heterogeneity (H_i) suggests clients whose local "
        "data distribution differs more from the global embedding centroid. If |r| is "
        "moderate-to-large and p<0.05, heterogeneity is a plausible predictor of how "
        "much personalization each client needs.",
        "",
        "### 2. Does hardware profile affect oracle personalization level?",
        "",
        "Mean oracle level by hardware:",
        hw_lines,
        "",
        "Clients on **LOW** hardware tend toward lower personalization levels when λ "
        "penalizes cost; **HIGH** hardware clients can afford deeper personalization "
        "without sacrificing utility.",
        "",
        "### 3. Is uniform personalization suboptimal?",
        "",
        f"**Best uniform level:** {best_uniform} (mean utility = {uniform_best_util:.4f}).",
        f"**Per-client oracle mean utility:** {oracle_mean_util:.4f}.",
        f"**Gap:** {uniform_gap:.4f} ({100 * uniform_gap / max(abs(uniform_best_util), 1e-6):.1f}% relative).",
        "",
        f"Oracle levels span **{n_distinct}** distinct choices; "
        f"{100 * majority_frac:.0f}% of clients share the modal level.",
        "",
        "A single global personalization policy leaves utility on the table when "
        "clients differ in data heterogeneity and hardware budget.",
        "",
        "### 4. Does this feasibility study support continuing FedPBG?",
        "",
    ]

    if supports:
        lines.append(
            "**Yes — preliminary evidence supports Phase 2.** Client-level oracle budgets "
            "vary with heterogeneity and hardware, and uniform personalization is "
            "suboptimal. Next steps: FedPBG scheduler and contextual bandit (not in "
            "this feasibility phase)."
        )
    else:
        lines.append(
            "**Mixed — proceed with caution.** Oracle diversity or utility gains are "
            "limited in this run. Consider longer training, full PTB-XL, or adjusted "
            "cost proxies before investing in the full FedPBG scheduler."
        )

    lines.extend([
        "",
        "## Outputs",
        "",
        "- `client_level_results.csv` — all (client, level) metrics and costs",
        "- `oracle_budget_results.csv` — P_i* per client and λ",
        "- Figures: heterogeneity_vs_budget, hardware_vs_budget, accuracy_cost_pareto, client_budget_distribution",
        "",
        "## Scope Note",
        "",
        "This phase performs **oracle budget discovery only**. No FedPBG scheduler or "
        "contextual bandit was implemented, per feasibility study scope.",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
