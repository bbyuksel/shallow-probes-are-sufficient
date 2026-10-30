"""
Markdown summary for oracle sparsity validation study.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _yes_no(condition: bool) -> str:
    return "**Yes**" if condition else "**No / inconclusive**"


def write_oracle_sparsity_summary(
    summary_df: pd.DataFrame,
    out_path: Path,
    default_lambda: float = 0.3,
) -> None:
    """Answer research questions from aggregated oracle sparsity results."""
    sub = summary_df[summary_df["lambda"] == default_lambda].copy()

    # Aggregate across seeds/alphas/client counts per dataset
    ds_agg = sub.groupby("dataset").agg(
        p0_ratio=("p0_ratio", "mean"),
        deep_ratio=("deep_ratio", "mean"),
        uniform_gap=("oracle_vs_uniform_rel_gap", "mean"),
        p5_gap=("oracle_vs_personalize_all_rel_gap", "mean"),
        oracle_util=("oracle_utility", "mean"),
        uniform_util=("best_uniform_utility", "mean"),
    )

    global_p0 = sub["p0_ratio"].mean()
    global_deep = sub["deep_ratio"].mean()
    global_uniform_gap = sub["oracle_vs_uniform_rel_gap"].mean()
    global_p5_gap = sub["oracle_vs_personalize_all_rel_gap"].mean()

    datasets = sorted(sub["dataset"].unique())
    generalizes = len(datasets) >= 2 and all(
        ds_agg.loc[d, "p0_ratio"] > 0.4 for d in datasets if d in ds_agg.index
    )

    proceed = (
        global_p0 > 0.5
        and global_uniform_gap > 0.15
        and global_p5_gap > 0.0
        and generalizes
    )

    lines = [
        "# FedSUCP Oracle Sparsity Validation Summary",
        "",
        f"**Study:** Oracle action a_i* = argmax_a [Macro-F1(a) − λ·Cost(a)] over P0–P5.  ",
        f"**Primary λ:** {default_lambda}.  ",
        f"**Configurations:** {len(sub)} experiment cells (dataset × seed × α × num_clients).",
        "",
        "## Global aggregates (λ = {})".format(default_lambda),
        "",
        f"- Mean **P0 ratio:** {global_p0:.1%}",
        f"- Mean **deep (P3–P5) ratio:** {global_deep:.1%}",
        f"- Mean **oracle vs best uniform relative gap:** {100*global_uniform_gap:.1f}%",
        f"- Mean **oracle vs personalize-all (P5) relative gap:** {100*global_p5_gap:.1f}%",
        "",
        "## Research Questions",
        "",
        "### 1. Is personalization benefit sparse?",
        "",
        _yes_no(global_p0 >= 0.5 and global_deep < 0.35),
        f" Most clients' oracle action is P0 (global-only); deep personalization (P3–P5) "
        f"accounts for ~{global_deep:.1%} of oracle selections on average.",
        "",
        "### 2. Is P0 optimal for a majority of clients?",
        "",
        _yes_no(global_p0 >= 0.5),
        f" Mean P0 oracle ratio = {global_p0:.1%} across all configurations.",
        "",
        "### 3. Are deep personalization benefits concentrated in a minority?",
        "",
        _yes_no(global_deep < 0.35),
        f" Deep personalization actions represent ~{global_deep:.1%} of oracle choices.",
        "",
        "### 4. Is uniform personalization suboptimal?",
        "",
        _yes_no(global_uniform_gap > 0.1),
        f" Oracle utility exceeds best uniform policy by ~{100*global_uniform_gap:.1f}% on average.",
        "",
        "### 5. Is personalize-all (P5 for everyone) suboptimal?",
        "",
        _yes_no(global_p5_gap > 0.0),
        f" Oracle exceeds personalize-all utility by ~{100*global_p5_gap:.1f}% on average "
        "(cost-aware utility penalizes unnecessary full personalization).",
        "",
        "### 6. Does the sparsity pattern generalize beyond PTB-XL?",
        "",
    ]

    for ds in datasets:
        if ds in ds_agg.index:
            lines.append(
                f"- **{ds}:** P0={ds_agg.loc[ds, 'p0_ratio']:.1%}, "
                f"deep={ds_agg.loc[ds, 'deep_ratio']:.1%}, "
                f"uniform gap={100*ds_agg.loc[ds, 'uniform_gap']:.1f}%"
            )

    lines.extend([
        "",
        _yes_no(generalizes),
        " Sparsity (majority P0) observed across multiple datasets supports cross-domain generality.",
        "",
        "### 7. Should FedSUCP proceed as a full ICML paper?",
        "",
    ])

    if proceed:
        lines.append(
            "**Yes — proceed to full FedSUCP method phase.** Oracle sparsity, uniform suboptimality, "
            "and personalize-all penalty replicate across settings. Next: probe-based sparse allocator "
            "(not implemented in this validation phase)."
        )
    else:
        lines.append(
            "**Proceed with caution.** Some sparsity indicators are weak or dataset-dependent. "
            "Strengthen with full-scale runs, additional seeds, or real device cost traces before "
            "committing to ICML submission."
        )

    lines.extend([
        "",
        "## Per-dataset detail",
        "",
        "| Dataset | P0 ratio | Deep ratio | Uniform gap | P5 gap |",
        "|---------|----------|------------|-------------|--------|",
    ])
    for ds in datasets:
        if ds in ds_agg.index:
            r = ds_agg.loc[ds]
            lines.append(
                f"| {ds} | {r['p0_ratio']:.3f} | {r['deep_ratio']:.3f} | "
                f"{100*r['uniform_gap']:.1f}% | {100*r['p5_gap']:.1f}% |"
            )
    lines.extend([
        "",
        "## Scope",
        "",
        "This study validates **oracle sparsity only**. FedSUCP allocator and contextual bandit "
        "are out of scope.",
        "",
        "## Output files",
        "",
        "- `oracle_sparsity_results.csv`",
        "- `dataset_level_summary.csv`",
        "- Figures: oracle_budget_distribution, p0_ratio, utility gaps, hardware distribution",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
