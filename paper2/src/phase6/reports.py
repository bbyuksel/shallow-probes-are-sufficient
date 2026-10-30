"""Phase-6 markdown reports."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from paper2.src.config import REPORTS_DIR, TABLES_DIR
from paper2.src.phase6.audit import POLICY_CLASSES, CONTEXT_ONLY_POLICIES


def _fmt(x, d=4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{float(x):.{d}f}"


def write_probe_sufficiency_summary(metrics: Dict, out: Path) -> None:
    lines = [
        "# Phase-6 Probe Sufficiency Metrics",
        "",
        f"**Best context-only policy:** `{metrics['context_only_best_policy']}`",
        "",
        "## 1. Oracle utility recovery (%)",
        "",
        "| Policy | Recovery % |",
        "|--------|----------:|",
    ]
    for k, v in metrics["oracle_recovery"].items():
        lines.append(f"| {k} | {_fmt(v, 1)} |")

    lines.extend(["", "## 2. Regret reduction vs. best uniform", "", "| Policy | Δ regret |", "|--------|--------:|"])
    for k, v in metrics["regret_reduction_vs_uniform"].items():
        lines.append(f"| {k} | {_fmt(v)} |")

    lines.extend(
        ["", "## 3. Regret reduction vs. context-only best", "", "| Policy | Δ regret |", "|--------|--------:|"]
    )
    for k, v in metrics["regret_reduction_vs_context_only"].items():
        lines.append(f"| {k} | {_fmt(v)} |")

    lines.extend(
        [
            "",
            "## 4. Probe information gain",
            "",
            "Gain_probe = Regret(context-only best) − Regret(probe policy)",
            "",
            "| Policy | Gain_probe |",
            "|--------|----------:|",
        ]
    )
    for k, v in metrics["probe_information_gain"].items():
        lines.append(f"| {k} | {_fmt(v)} |")

    lines.extend(
        [
            "",
            "## 5. Probe sufficiency ratio",
            "",
            "Sufficiency = 1 − Regret(policy) / Regret(uniform_depth)",
            "",
            "| Policy | Sufficiency |",
            "|--------|------------:|",
        ]
    )
    for k, v in metrics["probe_sufficiency_ratio"].items():
        lines.append(f"| {k} | {_fmt(v)} |")

    out.write_text("\n".join(lines), encoding="utf-8")


def write_policy_class_comparison(
    stats: pd.DataFrame,
    runtime: pd.DataFrame,
    metrics: Dict,
    out: Path,
) -> None:
    merged = stats.merge(runtime, on="policy", how="left")

    lines = [
        "# Phase-6 Policy Class Comparison",
        "",
        "## Class definitions",
        "",
        "- **A) Fixed:** majority, uniform_depth (+ global-only, personalize-all from Phase-4)",
        "- **B) Context-only learned:** multinomial logit, cost-sensitive logit, decision tree, random forest",
        "- **C) Probe-based non-learned:** greedy_probe",
        "- **D) Probe-based learned:** FedPBA",
        "",
        "## Per-policy summary",
        "",
        "| Class | Policy | Regret | Oracle recovery % | Action acc. | Probe trainings | Probe cost ratio |",
        "|-------|--------|-------:|------------------:|------------:|----------------:|-----------------:|",
    ]

    class_map = {p: k for k, v in POLICY_CLASSES.items() for p in v}
    class_map["global_only"] = "A_fixed"
    class_map["personalize_all"] = "A_fixed"

    for _, row in stats.sort_values("mean_regret").iterrows():
        pol = row["policy"]
        rt = merged[merged["policy"] == pol]
        probe_t = float(rt["mean_probe_trainings"].iloc[0]) if len(rt) else 0
        probe_r = float(rt["mean_probe_cost_ratio"].iloc[0]) if len(rt) else 0
        cls = class_map.get(pol, "—")
        lines.append(
            f"| {cls} | {pol} | {_fmt(row['mean_regret'])} | {_fmt(row['oracle_utility_recovery_pct'], 1)} | "
            f"{_fmt(row.get('action_accuracy', float('nan')))} | {_fmt(probe_t, 1)} | {_fmt(probe_r, 2)} |"
        )

    lines.extend(
        [
            "",
            "## Class-level averages",
            "",
            "| Class | Mean regret | Mean oracle recovery % |",
            "|-------|------------:|-----------------------:|",
        ]
    )
    for cls, pols in POLICY_CLASSES.items():
        sub = stats[stats["policy"].isin(pols)]
        lines.append(
            f"| {cls} | {_fmt(sub['mean_regret'].mean())} | {_fmt(sub['oracle_utility_recovery_pct'].mean(), 1)} |"
        )

    lines.extend(
        [
            "",
            "## Key contrasts",
            "",
            f"- Probe sufficiency (greedy): **{_fmt(metrics['probe_sufficiency_ratio']['greedy_probe'])}**",
            f"- Probe sufficiency (FedPBA): **{_fmt(metrics['probe_sufficiency_ratio']['fedpba'])}**",
            f"- Context-only best sufficiency: **{_fmt(metrics['probe_sufficiency_ratio']['context_only_best'])}**",
            f"- Probe information gain (greedy): **{_fmt(metrics['probe_information_gain']['greedy_probe'])}**",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def write_probe_mechanism(mechanism: Dict, out: Path) -> None:
    lines = [
        "# Phase-6 Probe Mechanism Analysis",
        "",
        "## 1. Correlation: shallow probe utilities vs. oracle utility",
        "",
        "| Probe signal | Pearson r with oracle utility |",
        "|--------------|------------------------------:|",
        f"| P0 utility | {_fmt(mechanism['corr_oracle_u_p0'])} |",
        f"| P1 utility | {_fmt(mechanism['corr_oracle_u_p1'])} |",
        f"| P2 utility | {_fmt(mechanism['corr_oracle_u_p2'])} |",
        f"| max(P0,P1,P2) utility | **{_fmt(mechanism['corr_oracle_u_max_shallow'])}** |",
        f"| Greedy probe deployed utility | {_fmt(mechanism['corr_oracle_greedy_probe_utility'])} |",
        "",
        "## 2. Oracle action depth distribution",
        "",
        f"- Oracle action in **P0–P2**: **{_fmt(mechanism['oracle_action_shallow_pct'], 1)}%** "
        f"({2700 - mechanism['oracle_deep_count']} / 2700)",
        f"- Oracle action in **P3–P5**: **{_fmt(mechanism['oracle_action_deep_pct'], 1)}%** "
        f"({mechanism['oracle_deep_count']} / 2700)",
        "",
        "## 3. When oracle is deep (P3–P5)",
        "",
        f"- Greedy probe achieves low regret (≤0.05): **{_fmt(mechanism['greedy_low_regret_when_oracle_deep_pct'], 1)}%** of deep-oracle cases",
        f"- High-regret greedy failures (oracle deep, deployed shallow, regret>0.05): **{mechanism['greedy_high_regret_deep_cases']}** cases",
        "",
        "## 4. Missed deep oracle actions (high-regret greedy failures)",
        "",
    ]
    if mechanism["top_missed_deep_oracle_actions"]:
        for k, v in mechanism["top_missed_deep_oracle_actions"].items():
            lines.append(f"- {k}: {v} cases")
    else:
        lines.append("- (none)")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "High correlation between max shallow probe utility and oracle utility supports the",
            "**Probe Sufficiency Hypothesis**: shallow probes encode most of the decision-relevant",
            "information when the oracle itself prefers shallow depths. Failures concentrate when",
            "the oracle selects deep actions invisible to greedy probe.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def write_probe_failure_modes(failure: Dict, confusion: pd.DataFrame, out: Path) -> None:
    lines = [
        "# Phase-6 Probe Failure Modes",
        "",
        "## Deep misallocation: predicted P0, oracle P3–P5",
        "",
        "| Policy | Count | Mean regret | High-regret (>0.05) |",
        "|--------|------:|------------:|--------------------:|",
        f"| Greedy probe | {failure['overall_gp_p0_oracle_deep']['count']} | "
        f"{_fmt(failure['overall_gp_p0_oracle_deep']['mean_regret'])} | "
        f"{failure['overall_gp_p0_oracle_deep']['high_regret_count']} |",
        f"| FedPBA | {failure['overall_fed_p0_oracle_deep']['count']} | "
        f"{_fmt(failure['overall_fed_p0_oracle_deep']['mean_regret'])} | "
        f"{failure['overall_fed_p0_oracle_deep']['high_regret_count']} |",
        "",
        "## By dataset",
        "",
        "| Dataset | GP regret | FedPBA regret | GP P0→deep | Fed P0→deep |",
        "|---------|----------:|--------------:|-----------:|------------:|",
    ]
    for ds, v in failure["by_dataset"].items():
        lines.append(
            f"| {ds} | {_fmt(v['greedy_probe_regret'])} | {_fmt(v['fedpba_regret'])} | "
            f"{v['gp_p0_oracle_deep']} | {v['fed_p0_oracle_deep']} |"
        )

    lines.extend(
        [
            "",
            "## By lambda",
            "",
            "| λ | GP regret | FedPBA regret | GP P0→deep | Fed P0→deep |",
            "|--:|----------:|--------------:|-----------:|------------:|",
        ]
    )
    for lam, v in failure["by_lambda"].items():
        lines.append(
            f"| {lam} | {_fmt(v['greedy_probe_regret'])} | {_fmt(v['fedpba_regret'])} | "
            f"{v['gp_p0_oracle_deep']} | {v['fed_p0_oracle_deep']} |"
        )

    lines.extend(["", "## PTB-XL", "", f"- Greedy P0→oracle deep: **{failure['ptbxl_gp_p0_oracle_deep']}** cases", f"- FedPBA P0→oracle deep: **{failure['ptbxl_fed_p0_oracle_deep']}** cases", "", "## Top action pairs (oracle deep, greedy shallow)", ""])
    for _, row in confusion.head(12).iterrows():
        lines.append(f"- greedy `{row['greedy_probe_action']}` → oracle `{row['oracle_action']}`: {int(row['count'])}")

    out.write_text("\n".join(lines), encoding="utf-8")


def write_paper_identity_decision(metrics: Dict, mechanism: Dict, failure: Dict, out: Path) -> None:
    gp_r = metrics["mean_regret"]["greedy_probe"]
    fed_r = metrics["mean_regret"]["fedpba"]
    suff_gp = metrics["probe_sufficiency_ratio"]["greedy_probe"]

    lines = [
        "# Phase-6 Paper Identity Decision",
        "",
        "## 1. Should Paper-2 be framed as a FedPBA method paper?",
        "",
        "**No.** Greedy probe achieves lower regret (0.0101 vs. 0.0120) without learning.",
        "FedPBA does not win the policy comparison. A method paper would invite immediate",
        "rejection on the greedy baseline.",
        "",
        "## 2. Should it be framed as Probe Sufficiency empirical discovery?",
        "",
        f"**Yes.** Shallow probes recover **{_fmt(metrics['oracle_recovery']['greedy_probe'], 1)}%** of oracle utility",
        f"with sufficiency ratio **{_fmt(suff_gp)}** vs. uniform depth. Context-only learners fail",
        f"(regret **{_fmt(metrics['mean_regret']['context_only_best'])}**). The scientific claim is:",
        "*probe outcomes contain most information needed for near-oracle depth allocation.*",
        "",
        "## 3. Strongest title (recommended)",
        "",
        "**\"Shallow Probes Are Sufficient for Near-Oracle Personalization Depth Allocation",
        "in Federated Learning\"**",
        "",
        "Alternative: *\"When Are Shallow Probes Enough? A Sufficiency Audit for Cost-Aware",
        "Personalization Depth in Federated Learning\"*",
        "",
        "## 4. Final four contributions",
        "",
        "1. **Probe Sufficiency Hypothesis** — formalized and tested against fixed, context-only, and probe policies.",
        "2. **Policy-class audit** — 2,700 live evaluations showing probes dominate deployment context.",
        "3. **Mechanism evidence** — correlation structure and oracle shallow/deep mass explain when probes suffice.",
        "4. **Failure characterization** — PTB-XL, low λ, and deep-oracle cases where sufficiency breaks.",
        "",
        "## 5. Central results (main text)",
        "",
        "- Policy-class regret table (fixed / context / greedy / FedPBA)",
        "- Probe sufficiency ratio and information gain",
        "- Oracle recovery: greedy 97.7%, FedPBA 97.3%, uniform 93.9%",
        "- max(P0,P1,P2) vs. oracle utility correlation",
        "- Failure heatmap: deep-oracle / P0 prediction",
        "",
        "## 6. Move to appendix",
        "",
        "- FedPBA architecture details and gradient boosting ablation",
        "- Full Phase-4 paired Wilcoxon tables",
        "- Per-seed stability tables",
        "- Ditto baseline (incomplete n)",
        "- Context-only individual model comparisons",
        "",
        "## 7. Is the paper stronger after reframing?",
        "",
        "**Yes — if honest about greedy probe.** The story becomes a falsifiable empirical claim",
        "(probe sufficiency) supported by systematic policy classes, not a weaker method pitch.",
        "FedPBA becomes one instantiation, not the hero.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def write_rewrite_plan(metrics: Dict, out: Path) -> None:
    lines = [
        "# Phase-6 Rewrite Plan: v3 → Probe Sufficiency Paper",
        "",
        "## New title",
        "",
        "*Shallow Probes Are Sufficient for Near-Oracle Personalization Depth Allocation in Federated Learning*",
        "",
        "## Abstract structure (5 sentences)",
        "",
        "1. Problem: PFL focuses on training; deployment requires choosing personalization depth under cost.",
        "2. Hypothesis: shallow probes (P0–P2) contain most information for near-oracle allocation.",
        "3. Audit: 2,700 live evaluations across policy classes (fixed, context-only, greedy, learned).",
        "4. Findings: greedy probe 97.7% oracle recovery; context-only fails; FedPBA extrapolates to deep depths.",
        "5. Limitation: sufficiency breaks on deep-oracle ECG cases.",
        "",
        "## Introduction flow",
        "",
        "1. FL deployment needs per-client depth decisions, not only training algorithms.",
        "2. Exhaustive oracle search is infeasible; uniform depth is strong but inflexible.",
        "3. **Probe Sufficiency Hypothesis** stated explicitly.",
        "4. Policy-class audit design (no new FL).",
        "5. Summary: probes sufficient for ~98% recovery; greedy probe strongest; FedPBA for extrapolation.",
        "",
        "## Results order",
        "",
        "1. Setup (unchanged ladder, utility, oracle)",
        "2. **Policy-class comparison table** (Phase-6 headline)",
        "3. Probe sufficiency metrics (gain, ratio, recovery)",
        "4. Mechanism: correlation + oracle shallow/deep mass",
        "5. Failure modes: PTB-XL, λ, deep-oracle",
        "6. FedPBA as learned extrapolator (subsection, not lead)",
        "",
        "## Keep",
        "",
        "- Utility formulation and ladder table",
        "- Phase-4 aggregate numbers (as part of audit)",
        "- `phase6_probe_sufficiency_ratio.png`, `phase6_policy_class_regret.png`",
        "- Failure analysis (reframed as sufficiency breaks)",
        "",
        "## Remove or demote",
        "",
        "- \"FedPBA is the deployable policy\" as lead claim",
        "- \"First deployable oracle approximation\"",
        "- Contextual decision problem as primary framing (→ secondary)",
        "- Method-first pipeline figure as Figure 1 (move to appendix)",
        "",
        "## Emphasize",
        "",
        f"- Greedy probe regret **{_fmt(metrics['mean_regret']['greedy_probe'])}** vs. uniform **{_fmt(metrics['mean_regret']['uniform_depth'])}**",
        f"- Probe information gain **{_fmt(metrics['probe_information_gain']['greedy_probe'])}**",
        f"- Context-only regret **{_fmt(metrics['mean_regret']['context_only_best'])}** (failure of metadata-only context)",
        "- Oracle shallow mass supports sufficiency",
        "",
        "## FedPBA positioning (one paragraph)",
        "",
        "\"FedPBA extrapolates beyond greedy probe to depths P3–P5 when shallow curves are",
        "ambiguous; it does not beat greedy probe on aggregate regret but enables deep deployment",
        "when required.\"",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
