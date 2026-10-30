"""Phase-4 report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from paper2.src.config import REPORTS_DIR
from paper2.src.phase4.statistics import bootstrap_ci, paired_comparison, policy_metrics


def _fmt(x, d=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.{d}f}"


def build_policy_stats_table(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in ["oracle", "fedpba", "best_uniform", "global_only", "personalize_all", "ditto"]:
        m = policy_metrics(detail, p)
        if m:
            m["policy"] = p
            rows.append(m)
    return pd.DataFrame(rows)


def build_dataset_stats(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in detail["dataset"].unique():
        sub = detail[detail["dataset"] == ds]
        bu = policy_metrics(sub, "best_uniform")
        fed = policy_metrics(sub, "fedpba")
        for p in ["fedpba", "best_uniform", "oracle"]:
            m = policy_metrics(sub, p)
            if not m:
                continue
            m["dataset"] = ds
            m["policy"] = p
            if p == "fedpba" and bu and fed:
                m["regret_reduction_vs_bu"] = bu["mean_regret"] - fed["mean_regret"]
                m["utility_improvement_vs_bu"] = fed["mean_utility"] - bu["mean_utility"]
                wrong = sub[sub["fedpba_action"] != sub["oracle_action"]]
                m["failure_rate"] = len(wrong) / max(len(sub), 1)
                hi = wrong["oracle_utility"] - wrong["fedpba_utility"]
                m["high_regret_wrong"] = int((hi > 0.05).sum())
            rows.append(m)
    return pd.DataFrame(rows)


def build_lambda_stats(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lam in sorted(detail["lambda"].unique()):
        sub = detail[detail["lambda"] == lam]
        for p in ["fedpba", "best_uniform"]:
            m = policy_metrics(sub, p)
            if m:
                m["lambda"] = lam
                m["policy"] = p
                rows.append(m)
    return pd.DataFrame(rows)


def write_statistical_validation_report(
    policy_stats: pd.DataFrame,
    comparisons: Dict[str, Dict],
    detail: pd.DataFrame,
    n_seeds: int,
    out_path: Optional[Path] = None,
) -> Path:
    out_path = out_path or REPORTS_DIR / "phase4_statistical_validation_report.md"
    fed = policy_stats[policy_stats.policy == "fedpba"].iloc[0]
    bu = policy_stats[policy_stats.policy == "best_uniform"].iloc[0]
    cmp_bu = comparisons.get("fedpba_vs_best_uniform", {})

    lines = [
        "# Phase-4 Statistical Validation Report",
        "",
        f"**Seeds:** {n_seeds} | **Client evaluations:** {len(detail):,}",
        "",
        "## Policy summary (95% bootstrap CI)",
        "",
        "| Policy | Mean utility [CI] | Mean regret [CI] |",
        "|--------|-------------------|-------------------|",
    ]
    for _, r in policy_stats.iterrows():
        lines.append(
            f"| {r['policy']} | {_fmt(r['mean_utility'])} [{_fmt(r['utility_ci_lo'])}, {_fmt(r['utility_ci_hi'])}] | "
            f"{_fmt(r['mean_regret'])} [{_fmt(r['regret_ci_lo'])}, {_fmt(r['regret_ci_hi'])}] |"
        )

    lines.extend(
        [
            "",
            "## FedPBA vs Best Uniform (paired)",
            "",
            f"- Mean utility difference: {_fmt(cmp_bu.get('mean_diff'))} [{_fmt(cmp_bu.get('diff_ci_lo'))}, {_fmt(cmp_bu.get('diff_ci_hi'))}]",
            f"- Win rate: {_fmt(cmp_bu.get('win_rate'))}",
            f"- Cohen's d_z: {_fmt(cmp_bu.get('cohens_dz'))}",
            f"- Wilcoxon p (FedPBA > BU): {_fmt(cmp_bu.get('wilcoxon_p'), 6)}",
            f"- Paired t-test p: {_fmt(cmp_bu.get('ttest_p'), 6)}",
            f"- n pairs: {cmp_bu.get('n_pairs', '—')}",
            "",
            f"**FedPBA action accuracy:** {_fmt(fed.get('action_accuracy'))}",
            "",
        ]
    )
    for key in ["fedpba_vs_global_only", "fedpba_vs_personalize_all"]:
        c = comparisons.get(key, {})
        if c:
            label = key.replace("_", " ")
            lines.append(f"### {label}")
            lines.append(f"- Mean diff: {_fmt(c.get('mean_diff'))}, win rate: {_fmt(c.get('win_rate'))}, p: {_fmt(c.get('wilcoxon_p'), 6)}")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_dataset_breakdown(ds_stats: pd.DataFrame, detail: pd.DataFrame, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or REPORTS_DIR / "phase4_dataset_breakdown.md"
    lines = ["# Phase-4 Dataset Breakdown", ""]
    for ds in sorted(detail["dataset"].unique()):
        sub = detail[detail["dataset"] == ds]
        fed = policy_metrics(sub, "fedpba")
        bu = policy_metrics(sub, "best_uniform")
        wrong = sub[sub["fedpba_action"] != sub["oracle_action"]]
        lines.extend(
            [
                f"## {ds}",
                "",
                f"- FedPBA regret: {_fmt(fed.get('mean_regret'))} [{_fmt(fed.get('regret_ci_lo'))}, {_fmt(fed.get('regret_ci_hi'))}]",
                f"- Best uniform regret: {_fmt(bu.get('mean_regret'))}",
                f"- Regret reduction: {_fmt(bu.get('mean_regret', 0) - fed.get('mean_regret', 0))}",
                f"- Action accuracy: {_fmt(fed.get('action_accuracy'))}",
                f"- Failure rate: {100 * len(wrong) / len(sub):.1f}%",
                f"- High-regret wrong: {int((wrong['oracle_utility'] - wrong['fedpba_utility'] > 0.05).sum())}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_lambda_breakdown(detail: pd.DataFrame, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or REPORTS_DIR / "phase4_lambda_breakdown.md"
    lines = ["# Phase-4 Lambda Breakdown", ""]
    for lam in sorted(detail["lambda"].unique()):
        sub = detail[detail["lambda"] == lam]
        fed = policy_metrics(sub, "fedpba")
        bu = policy_metrics(sub, "best_uniform")
        cmp = paired_comparison(sub["fedpba_utility"].values, sub["best_uniform_utility"].values)
        beats = fed.get("mean_regret", 1) < bu.get("mean_regret", 1)
        lines.extend(
            [
                f"## λ = {lam}",
                "",
                f"- FedPBA regret: {_fmt(fed.get('mean_regret'))}",
                f"- Best uniform regret: {_fmt(bu.get('mean_regret'))}",
                f"- FedPBA beats BU: **{'yes' if beats else 'no'}**",
                f"- Paired win rate: {_fmt(cmp.get('win_rate'))}, p={_fmt(cmp.get('wilcoxon_p'), 6)}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_submission_readiness_audit(
    policy_stats: pd.DataFrame,
    comparisons: Dict,
    detail: pd.DataFrame,
    runtime_stats: Dict,
    n_seeds: int,
    estimated_10_seed_hours: Optional[float] = None,
    out_path: Optional[Path] = None,
) -> Path:
    out_path = out_path or REPORTS_DIR / "phase4_submission_readiness_audit.md"
    fed = policy_stats[policy_stats.policy == "fedpba"].iloc[0]
    bu = policy_stats[policy_stats.policy == "best_uniform"].iloc[0]
    cmp_bu = comparisons.get("fedpba_vs_best_uniform", {})
    ptbxl = detail[detail["dataset"].str.contains("ptbxl", case=False, na=False)]
    ptbxl_fed = policy_metrics(ptbxl, "fedpba") if len(ptbxl) else {}
    ptbxl_bu = policy_metrics(ptbxl, "best_uniform") if len(ptbxl) else {}

    sig = cmp_bu.get("wilcoxon_p", 1) < 0.05 and fed["regret_ci_hi"] < bu["regret_ci_lo"]
    overlap = fed["regret_ci_hi"] > bu["regret_ci_lo"]

    seed_regrets = detail.groupby("seed").apply(lambda g: (g["oracle_utility"] - g["fedpba_utility"]).mean())
    seed_stable = seed_regrets.std() < 0.02 if len(seed_regrets) > 1 else True

    lines = [
        "# Phase-4 Submission Readiness Audit",
        "",
        f"**Seeds analyzed:** {n_seeds}",
        "",
        "## Questions",
        "",
        f"1. **Is FedPBA statistically better than best uniform?** "
        f"{'Yes' if cmp_bu.get('wilcoxon_p', 1) < 0.05 else 'Unclear'} "
        f"(Wilcoxon p={_fmt(cmp_bu.get('wilcoxon_p'), 6)}, CI overlap: {'yes' if overlap else 'no'}).",
        "",
        f"2. **Is the improvement practically meaningful?** "
        f"Mean regret reduction {_fmt(bu['mean_regret'] - fed['mean_regret'])}; "
        f"Cohen's d_z={_fmt(cmp_bu.get('cohens_dz'))}.",
        "",
        f"3. **Does PTB-XL remain a weakness?** "
        f"{'Yes' if ptbxl_fed.get('mean_regret', 0) > 0.015 else 'Moderate'} "
        f"(PTB-XL FedPBA regret {_fmt(ptbxl_fed.get('mean_regret'))} vs BU {_fmt(ptbxl_bu.get('mean_regret'))}).",
        "",
        f"4. **Stable across seeds?** "
        f"{'Yes' if seed_stable else 'Some variance'} "
        f"(cross-seed regret std: {_fmt(seed_regrets.std() if len(seed_regrets) > 1 else 0)}).",
        "",
        "5. **Venue readiness:**",
        f"   - **ECML/AISTATS:** {'Borderline-ready' if sig and not overlap else 'Needs more seeds/significance'}",
        "   - **TMLR:** Plausible with full reproducibility package",
        "   - **ICML/NeurIPS:** Not yet — need 10 seeds, real PTB-XL, theory",
        "",
        f"6. **Recommendation:** "
        f"{'**Continue writing**' if sig and fed['mean_regret'] < bu['mean_regret'] else '**Continue cautiously or redesign**'}",
        "",
        "## Runtime",
        "",
        f"- Median probe/oracle time ratio: {_fmt(runtime_stats.get('median_time_ratio'))}",
        f"- Estimated speedup: {_fmt(runtime_stats.get('median_speedup'), 1)}×",
    ]
    if estimated_10_seed_hours:
        lines.append(f"- Estimated runtime for 10 seeds: ~{estimated_10_seed_hours:.1f} GPU-hours")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
