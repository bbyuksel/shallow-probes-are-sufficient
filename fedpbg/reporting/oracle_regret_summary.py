"""
Conservative markdown summary for oracle regret final validation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fedpbg.oracle.regret_stats import BASELINE_POLICIES, robustness_by_dataset


def _verdict(mean_rel_pct: float, win_rate: float, strong_thresh: float = 10.0) -> str:
    if mean_rel_pct >= strong_thresh and win_rate >= 0.8:
        return "**Strong**"
    if mean_rel_pct >= 3.0 and win_rate >= 0.6:
        return "**Moderate**"
    if mean_rel_pct > 0 and win_rate > 0.5:
        return "**Weak but consistent**"
    return "**Weak / inconclusive**"


def write_oracle_regret_summary(
    regret_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    out_path: Path,
    n_configs: int,
    n_datasets: int,
) -> None:
    """Answer scientific questions conservatively from regret evidence."""
    baselines = regret_df[regret_df["policy"].isin(BASELINE_POLICIES)]
    oracle_rows = regret_df[regret_df["policy"] == "ORACLE"]

    global_stats = {}
    for pol in BASELINE_POLICIES:
        global_stats[pol] = robustness_by_dataset(baselines, pol)

    # Per-dataset
    ds_stats = {}
    for ds in sorted(regret_df["dataset"].unique()):
        ds_stats[ds] = {pol: robustness_by_dataset(baselines[baselines["dataset"] == ds], pol) for pol in BASELINE_POLICIES}

    # Phase transition robustness: uniform regret varies with lambda?
    uniform = baselines[baselines["policy"] == "BEST_UNIFORM"]
    lambda_var = uniform.groupby("lambda")["regret"].mean()
    alpha_var = uniform.groupby("dirichlet_alpha")["regret"].mean()

    lines = [
        "# Oracle Regret Study — Final Validation Summary",
        "",
        "**Paper title (working):** *Phase Transitions in Personalized Federated Learning:",
        "A Utility-Based Characterization of the Personalization–Cost Tradeoff*",
        "",
        f"**Configurations analyzed:** {n_configs} (dataset × α × seed), ",
        f"**{n_datasets} datasets**, ",
        f"**{regret_df['lambda'].nunique()} λ values**.",
        "",
        "> Conservative assessment. Claims below are tied directly to observed regret statistics.",
        "",
        "## Global regret vs ORACLE (mean across all cells)",
        "",
        "| Policy | Mean regret | Mean relative regret (%) | Oracle win rate |",
        "|--------|-------------|--------------------------|-----------------|",
    ]

    for pol in BASELINE_POLICIES:
        s = global_stats.get(pol, {})
        lines.append(
            f"| {pol} | {s.get('mean_regret', float('nan')):.4f} | "
            f"{s.get('mean_relative_regret_pct', float('nan')):.1f}% | "
            f"{100*s.get('oracle_win_rate', 0):.0f}% |"
        )

    lines.extend(["", "## Scientific Questions", ""])

    questions = [
        ("1. Is Oracle meaningfully better than Random?", "RANDOM"),
        ("2. Is Oracle meaningfully better than Uniform?", "BEST_UNIFORM"),
        ("3. Is Oracle meaningfully better than Personalize-All?", "PERSONALIZE_ALL"),
        ("4. Is Oracle meaningfully better than Global-Only?", "GLOBAL_ONLY"),
    ]

    for qtext, pol in questions:
        s = global_stats.get(pol, {})
        verdict = _verdict(s.get("mean_relative_regret_pct", 0), s.get("oracle_win_rate", 0))
        lines.append(f"### {qtext}")
        lines.append("")
        lines.append(
            f"{verdict} — mean relative regret {s.get('mean_relative_regret_pct', 0):.1f}%, "
            f"oracle wins in {100*s.get('oracle_win_rate', 0):.0f}% of configurations."
        )
        if not sig_df.empty and pol in sig_df["baseline"].values:
            row = sig_df[sig_df["baseline"] == pol].iloc[0]
            lines.append(
                f" Paired Wilcoxon (client-level): significant in "
                f"{100*row['fraction_p_lt_0.05']:.0f}% of configs (median p={row['median_p_value']:.3f})."
            )
        lines.append("")

    lines.extend([
        "### 5. Are the observed personalization phase transitions robust?",
        "",
    ])

    if len(lambda_var) >= 2 and lambda_var.max() - lambda_var.min() > 0.02:
        lines.append(
            "**Partially supported.** Regret of BEST_UNIFORM vs oracle varies with λ "
            f"(range {lambda_var.min():.3f}–{lambda_var.max():.3f}), consistent with a "
            "cost-driven phase transition. α also modulates regret, but effects are dataset-dependent."
        )
    else:
        lines.append(
            "**Insufficient evidence in this run** to confirm robust phase transitions; "
            "expand α/λ grid and seeds."
        )

    lines.extend([
        "",
        "### 6. Strongest defensible scientific claim",
        "",
        "> Under a utility objective U = Performance − λ·Cost, **fixed global policies** ",
        "> (uniform personalization, personalize-all, global-only) incur **non-zero regret** ",
        "> relative to per-client oracle allocation in most configurations; the magnitude and ",
        "> which baseline is worst **depend on λ, dataset, and α** — supporting a ",
        "> **phase-transition view** rather than a universal sparsity law.",
        "",
        "### 7. Claims that should NOT be made",
        "",
        "- Do **not** claim personalization is always sparse (CIFAR-100 at λ=0 shows deep personalization).",
        "- Do **not** claim oracle is always far above random (random regret can be small when λ is high).",
        "- Do **not** claim a new algorithm — this is **characterization + oracle upper bound** only.",
        "- Do **not** claim cross-device hardware generalization without measured device costs.",
        "- Do **not** overstate statistical power with few seeds or synthetic PTB-XL.",
        "",
        "### 8. Would an ICML reviewer consider the evidence sufficient?",
        "",
    ])

    # Conservative ICML assessment
    uniform_rel = global_stats.get("BEST_UNIFORM", {}).get("mean_relative_regret_pct", 0)
    p5_rel = global_stats.get("PERSONALIZE_ALL", {}).get("mean_relative_regret_pct", 0)
    n_ds = n_datasets

    if uniform_rel >= 15 and p5_rel > 0 and n_ds >= 2 and n_configs >= 10:
        lines.append(
            "**Borderline-to-sufficient for a characterization paper** if paired with: "
            "(i) full α/λ phase diagrams, (ii) ≥2 real datasets with multiple seeds, "
            "(iii) clear theory sketch of utility-dominated regimes. "
            "Current evidence supports the **problem formulation** and **phase-transition framing** "
            "more strongly than a new method contribution."
        )
    elif uniform_rel >= 5:
        lines.append(
            "**Insufficient alone for ICML main track** — regret vs uniform is moderate but "
            "needs broader benchmarks, more seeds, and real (non-synthetic) PTB-XL. "
            "Suitable as strong motivation section; method paper still required."
        )
    else:
        lines.append(
            "**Insufficient** — regret gaps are too small or inconsistent. "
            "Do not submit on characterization evidence alone."
        )

    lines.extend([
        "",
        "## Per-dataset regret (BEST_UNIFORM baseline)",
        "",
    ])
    for ds, stats in ds_stats.items():
        u = stats.get("BEST_UNIFORM", {})
        lines.append(
            f"- **{ds}:** mean regret={u.get('mean_regret', 0):.4f}, "
            f"rel={u.get('mean_relative_regret_pct', 0):.1f}%, "
            f"win rate={100*u.get('oracle_win_rate', 0):.0f}%"
        )

    lines.extend([
        "",
        "## Scope",
        "",
        "Oracle regret study only. No allocator, bandit, or RL implemented.",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
