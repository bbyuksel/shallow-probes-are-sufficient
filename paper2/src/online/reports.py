"""Phase-3 online FedPBA report."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from paper2.src.config import REPORTS_DIR


def _fmt(x, d=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.{d}f}"


def write_phase3_report(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    cache_compare: Optional[pd.DataFrame] = None,
    out_path: Optional[Path] = None,
) -> Path:
    out_path = out_path or REPORTS_DIR / "phase3_online_fedpba_report.md"

    def agg_policy(p):
        sub = summary[summary.policy == p]
        if sub.empty:
            return {}
        return {
            "mean_utility": sub["mean_utility"].mean(),
            "mean_regret": sub["mean_regret"].mean(),
            "action_accuracy": sub["action_accuracy"].mean() if "action_accuracy" in sub else np.nan,
        }

    fed = agg_policy("fedpba")
    bu = agg_policy("best_uniform")
    oracle = agg_policy("oracle")
    go = agg_policy("global_only")

    beats_bu = fed.get("mean_regret", 999) < bu.get("mean_regret", 999)
    regret_vs_oracle = fed.get("mean_regret", np.nan)
    oracle_gap = regret_vs_oracle

    fed_rows = summary[summary.policy == "fedpba"]
    probe_ratio = fed_rows["probe_cost_ratio"].mean() if "probe_cost_ratio" in fed_rows else np.nan
    probe_sec = fed_rows["probe_sec"].mean() if "probe_sec" in fed_rows else np.nan
    oracle_sec = fed_rows["oracle_levels_sec"].mean() if "oracle_levels_sec" in fed_rows else np.nan

    ds_fed = fed_rows.groupby("dataset")["mean_regret"].mean() if len(fed_rows) else pd.Series()
    ds_bu = summary[summary.policy == "best_uniform"].groupby("dataset")["mean_regret"].mean()

    works_online = regret_vs_oracle < 0.08 and beats_bu
    ptbxl_hard = float(ds_fed.get("ptbxl", 0)) > 0.015 if len(ds_fed) else False
    if works_online and regret_vs_oracle < 0.03 and not ptbxl_hard:
        rec = "**continue** — Live FedPBA approximates oracle with large gains over best uniform; paper is a real allocation study."
    elif works_online:
        rec = "**continue cautiously** — Live FedPBA works on CIFAR but PTB-XL regret is higher; validate on real ECG data before submission."
    elif beats_bu:
        rec = "**redesign** — Beats best uniform but oracle gap too large; improve probes or allocator."
    else:
        rec = "**stop or merge** — Live FedPBA does not beat best uniform; offline gains do not transfer."

    lines = [
        "# Phase-3 Online FedPBA Report",
        "",
        "**Title:** FedPBA: Online Probe-Based Personalization Budget Allocation for Federated Learning",
        "",
        "Live evaluation: FedAvg → P0/P1/P2 probes → allocator → apply action (utility from full level eval for scoring).",
        "",
        "## Aggregated policy comparison",
        "",
        "| Policy | Mean utility | Mean regret | Action accuracy |",
        "|--------|-------------:|------------:|----------------:|",
    ]
    for p in ["oracle", "fedpba", "best_uniform", "global_only", "ditto", "personalize_all"]:
        a = agg_policy(p)
        if not a:
            continue
        lines.append(
            f"| {p} | {_fmt(a.get('mean_utility'))} | {_fmt(a.get('mean_regret'))} | {_fmt(a.get('action_accuracy'))} |"
        )

    lines.extend(
        [
            "",
            "## Per-dataset regret (FedPBA vs best uniform)",
            "",
            "| Dataset | FedPBA regret | Best uniform regret |",
            "|---------|--------------:|--------------------:|",
        ]
    )
    for ds in sorted(set(ds_fed.index) | set(ds_bu.index)):
        lines.append(f"| {ds} | {_fmt(ds_fed.get(ds, np.nan))} | {_fmt(ds_bu.get(ds, np.nan))} |")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- **Scoring uses same-session level outcomes:** Oracle runs full P0–P5 for reference; FedPBA deploy path uses 3 probes + allocator; final utility is read from the evaluated level (no second training).",
            "- **PTB-XL sessions are fast (~50s)** — likely synthetic/fallback ECG if WFDB data absent; real PTB-XL may differ.",
            "- **Ditto underperforms** on live CIFAR (high normalized cost proxy); compare cautiously.",
            "- **Cache vs live:** Action agreement 26–93% depending on dataset/λ; regret magnitudes are similar.",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Runtime / probe overhead",
            "",
            f"- Mean probe wall time (P0–P2): {_fmt(probe_sec, 1)}s per session",
            f"- Mean oracle wall time (P0–P5): {_fmt(oracle_sec, 1)}s per session",
            f"- Probe trainings / oracle trainings: {_fmt(probe_ratio, 2)} ratio",
            "",
        ]
    )

    if cache_compare is not None and len(cache_compare):
        lines.extend(["## Online vs cache replay", "", "| Dataset | λ | Action agreement | Regret online | Regret cache |", "|---------|---|----------------:|--------------:|-------------:|"])
        for _, r in cache_compare.iterrows():
            lines.append(
                f"| {r['dataset']} | {r['lambda']} | {_fmt(r['action_agreement'])} | "
                f"{_fmt(r['mean_regret_online'])} | {_fmt(r['mean_regret_cache'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Honest answers",
            "",
            f"1. **Does FedPBA work in live FL evaluation?** "
            f"{'Yes — low regret vs oracle on live runs.' if regret_vs_oracle < 0.05 else ('Partially.' if regret_vs_oracle < 0.1 else 'No — regret too high.')}",
            "",
            f"2. **Does it beat best uniform?** {'Yes' if beats_bu else 'No'} "
            f"(FedPBA regret {_fmt(fed.get('mean_regret'))}, best uniform {_fmt(bu.get('mean_regret'))}).",
            "",
            f"3. **Does it approximate oracle sufficiently?** "
            f"Mean regret vs oracle: {_fmt(oracle_gap)} "
            f"(action accuracy {_fmt(fed.get('action_accuracy'))}).",
            "",
            f"4. **How much cheaper than exhaustive oracle search?** "
            f"~{100 * (1 - probe_ratio):.0f}% fewer probe trainings (3 vs 6 levels per client); "
            f"wall time ratio ~{probe_sec / max(oracle_sec, 1e-6):.2f} for probes vs full oracle.",
            "",
            "5. **ECML/AISTATS?** "
            f"{'Plausible if live results hold across datasets.' if works_online else 'Not yet — live transfer is the blocker.'}",
            "",
            "6. **ICML/NeurIPS?** Requires multi-seed live FL, broader datasets, and causal analysis of probe failures.",
            "",
            f"7. **Recommendation:** {rec}",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
