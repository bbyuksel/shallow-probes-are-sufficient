"""Phase-4 failure stability analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from paper2.src.config import LEVELS, REPORTS_DIR


def _fmt_mr(x):
    return f"{x:.4f}" if x == x else "—"


def write_failure_stability_report(detail: pd.DataFrame, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or REPORTS_DIR / "phase4_failure_stability_report.md"
    d = detail.copy()
    d["correct"] = d["fedpba_action"] == d["oracle_action"]
    d["regret"] = d["oracle_utility"] - d["fedpba_utility"]
    wrong = d[~d["correct"]]

    lines = [
        "# Phase-4 Failure Stability Report",
        "",
        f"Across **{d['seed'].nunique()}** seeds, **{len(d):,}** client evaluations.",
        "",
        f"- Wrong action rate: {100 * len(wrong) / max(len(d), 1):.1f}%",
        f"- High-regret wrong (>0.05): {int((wrong['regret'] > 0.05).sum())}",
        "",
        "## By dataset",
        "",
        "| Dataset | Wrong rate | Mean regret (all) | Mean regret (wrong) |",
        "|---------|----------:|--------------------:|--------------------:|",
    ]

    for ds, g in d.groupby("dataset"):
        w = g[~g["correct"]]
        wr = 100 * len(w) / len(g)
        mr_wrong = w["regret"].mean() if len(w) else float("nan")
        lines.append(
            f"| {ds} | {wr:.1f}% | {g['regret'].mean():.4f} | {_fmt_mr(mr_wrong)} |"
        )

    lines.extend(["", "## By seed (stability)", "", "| Seed | Wrong rate | Mean regret |", "|------|----------:|------------:|"])
    for seed, g in d.groupby("seed"):
        w = g[~g["correct"]]
        lines.append(f"| {seed} | {100 * len(w) / len(g):.1f}% | {g['regret'].mean():.4f} |")

    ptbxl = d[d["dataset"].str.contains("ptbxl", case=False, na=False)]
    if len(ptbxl):
        lines.extend(
            [
                "",
                "## PTB-XL failures",
                "",
                f"- Mean regret: {ptbxl['regret'].mean():.4f}",
                f"- Wrong rate: {100 * (1 - ptbxl['correct'].mean()):.1f}%",
            ]
        )

    deep_oracle = {"P3", "P4", "P5"}
    deep_miss = wrong[wrong["oracle_action"].isin(deep_oracle) & (wrong["fedpba_action"] == "P0")]
    p0_as_deep = wrong[(wrong["oracle_action"] == "P0") & wrong["fedpba_action"].isin(deep_oracle)]
    deep_pred_p0 = wrong[wrong["fedpba_action"] == "P0"]

    lines.extend(
        [
            "",
            "## Deep personalization errors",
            "",
            f"- Oracle P3/P4/P5 predicted as P0: **{len(deep_miss)}** (mean regret {deep_miss['regret'].mean():.4f})" if len(deep_miss) else "- Oracle P3/P4/P5 predicted as P0: 0",
            f"- Oracle P0 predicted as P3/P4/P5: **{len(p0_as_deep)}**" if len(p0_as_deep) else "- Oracle P0 predicted as P3/P4/P5: 0",
            "",
            "## Lambda-specific high-regret errors (>0.05)",
            "",
            "| λ | Count | Mean regret |",
            "|---|------:|------------:|",
        ]
    )
    hi = wrong[wrong["regret"] > 0.05]
    for lam, g in hi.groupby("lambda"):
        lines.append(f"| {lam} | {len(g)} | {g['regret'].mean():.4f} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
