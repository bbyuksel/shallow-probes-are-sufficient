"""Phase-3 online failure analysis."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from paper2.src.config import LEVELS, REPORTS_DIR


def write_online_failure_analysis(
    detail: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> Path:
    out_path = out_path or REPORTS_DIR / "phase3_online_failure_analysis.md"
    fed = detail.copy()
    fed["correct"] = fed["fedpba_action"] == fed["oracle_action"]
    fed["regret"] = fed["oracle_utility"] - fed["fedpba_utility"]
    wrong = fed[~fed["correct"]]

    lines = [
        "# Phase-3 Online FedPBA Failure Analysis",
        "",
        f"- Total client-configs: {len(fed):,}",
        f"- Wrong action: {len(wrong):,} ({100 * len(wrong) / max(len(fed), 1):.1f}%)",
        f"- Wrong low regret (≤0.01): {int((wrong['regret'] <= 0.01).sum())}",
        f"- Wrong high regret (>0.05): {int((wrong['regret'] > 0.05).sum())}",
        "",
    ]

    by_ds = (
        wrong.groupby("dataset")
        .agg(n_wrong=("regret", "count"), mean_regret=("regret", "mean"))
        .sort_values("mean_regret", ascending=False)
    )
    lines.extend(["## Dataset-specific failures", "", "| Dataset | # wrong | Mean regret |", "|---------|--------:|------------:|"])
    for ds, r in by_ds.iterrows():
        lines.append(f"| {ds} | {int(r['n_wrong'])} | {r['mean_regret']:.4f} |")
    lines.append("")

    ptbxl = wrong[wrong["dataset"].str.contains("ptbxl", case=False, na=False)]
    if len(ptbxl):
        lines.extend(
            [
                "## PTB-XL failures",
                "",
                f"- Wrong predictions: {len(ptbxl)}",
                f"- Mean regret when wrong: {ptbxl['regret'].mean():.4f}",
                "",
            ]
        )

    pairs = []
    for _, r in wrong.iterrows():
        pairs.append((r["oracle_action"], r["fedpba_action"], r["regret"]))
    if pairs:
        pdf = pd.DataFrame(pairs, columns=["oracle", "predicted", "regret"])
        agg = (
            pdf.groupby(["oracle", "predicted"])
            .agg(count=("regret", "count"), mean_regret=("regret", "mean"))
            .reset_index()
            .sort_values("mean_regret", ascending=False)
        )
        p0p2p5 = agg[
            agg["oracle"].isin(["P0", "P2", "P5"]) | agg["predicted"].isin(["P0", "P2", "P5"])
        ]
        lines.extend(["## P0/P2/P5 confusion (wrong predictions)", ""])
        lines.append("| Oracle | Predicted | Count | Mean regret |")
        lines.append("|--------|-----------|------:|------------:|")
        for _, r in p0p2p5.head(12).iterrows():
            lines.append(f"| {r['oracle']} | {r['predicted']} | {int(r['count'])} | {r['mean_regret']:.4f} |")
        lines.append("")

        deep_miss = agg[(agg["oracle"].isin(["P3", "P4", "P5"])) & (agg["predicted"] == "P0")]
        if len(deep_miss):
            lines.extend(["## Deep personalization missed as P0", ""])
            lines.append("| Oracle | Count | Mean regret |")
            lines.append("|--------|------:|------------:|")
            for oracle, g in deep_miss.groupby("oracle"):
                lines.append(f"| {oracle} | {int(g['count'].sum())} | {g['mean_regret'].mean():.4f} |")
            lines.append("")

    by_lam = (
        wrong.groupby("lambda")
        .agg(n_wrong=("regret", "count"), mean_regret=("regret", "mean"))
        .sort_values("mean_regret", ascending=False)
    )
    lines.extend(["## Lambda-specific failures", "", "| λ | # wrong | Mean regret |", "|---|--------:|------------:|"])
    for lam, r in by_lam.iterrows():
        lines.append(f"| {lam} | {int(r['n_wrong'])} | {r['mean_regret']:.4f} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
