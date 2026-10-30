"""FedPBA failure analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from paper2.src.config import LEVELS, REPORTS_DIR


def _regret_bucket(regret: float) -> str:
    if regret <= 0.01:
        return "low_regret_wrong_action"
    if regret <= 0.05:
        return "moderate_regret_wrong_action"
    return "high_regret_wrong_action"


def analyze_failures(detail: pd.DataFrame, split_name: str = "random") -> Dict:
    sub = detail[detail["split_name"] == split_name].copy()
    if sub.empty:
        return {}

    sub["correct"] = sub["fedpba_action"] == sub["oracle_action"]
    sub["regret"] = sub["oracle_utility"] - sub["fedpba_utility"]
    wrong = sub[~sub["correct"]].copy()

    failure_cases = {
        "wrong_low_regret": int((wrong["regret"] <= 0.01).sum()),
        "wrong_moderate_regret": int(((wrong["regret"] > 0.01) & (wrong["regret"] <= 0.05)).sum()),
        "wrong_high_regret": int((wrong["regret"] > 0.05).sum()),
        "total_wrong": int(len(wrong)),
        "total_clients": int(len(sub)),
    }

    by_dataset = (
        wrong.groupby("dataset")
        .agg(n_wrong=("regret", "count"), mean_regret=("regret", "mean"))
        .sort_values("mean_regret", ascending=False)
        .reset_index()
    )

    by_lambda = (
        wrong.groupby("lambda")
        .agg(n_wrong=("regret", "count"), mean_regret=("regret", "mean"))
        .sort_values("mean_regret", ascending=False)
        .reset_index()
    )

    confusion_pairs = []
    for true_a in LEVELS:
        for pred_a in LEVELS:
            if true_a == pred_a:
                continue
            mask = (sub["oracle_action"] == true_a) & (sub["fedpba_action"] == pred_a)
            if mask.sum() == 0:
                continue
            confusion_pairs.append(
                {
                    "oracle": true_a,
                    "predicted": pred_a,
                    "count": int(mask.sum()),
                    "mean_regret": float(sub.loc[mask, "regret"].mean()),
                }
            )
    confusion_df = pd.DataFrame(confusion_pairs).sort_values("mean_regret", ascending=False)

    p0_p2_p5 = confusion_df[
        confusion_df["oracle"].isin(["P0", "P2", "P5"]) | confusion_df["predicted"].isin(["P0", "P2", "P5"])
    ]

    return {
        "split_name": split_name,
        "failure_cases": failure_cases,
        "by_dataset": by_dataset,
        "by_lambda": by_lambda,
        "confusion_pairs": confusion_df,
        "p0_p2_p5_confusion": p0_p2_p5,
    }


def write_failure_analysis(
    details: pd.DataFrame,
    out_path: Path | None = None,
    splits: List[str] | None = None,
) -> Path:
    out_path = out_path or REPORTS_DIR / "fedpba_failure_analysis.md"
    splits = splits or ["random", "loo_dataset_cifar10", "loo_dataset_ptbxl", "loo_lambda_0.5"]

    lines = [
        "# FedPBA Failure Analysis",
        "",
        "Cases where FedPBA predicts a different action than the oracle, grouped by regret severity.",
        "",
    ]

    for split in splits:
        avail = details["split_name"].unique()
        match = next((s for s in avail if s == split or split in s), None)
        if match is None:
            continue
        sub_detail = details[details["split_name"] == match]
        if sub_detail.empty:
            continue
        # Use primary model if multiple
        if "model_name" in sub_detail.columns:
            model = sub_detail["model_name"].value_counts().index[0]
            sub_detail = sub_detail[sub_detail["model_name"] == model]
        if "feature_set" in sub_detail.columns:
            fs = sub_detail["feature_set"].value_counts().index[0]
            sub_detail = sub_detail[sub_detail["feature_set"] == fs]

        res = analyze_failures(sub_detail, match)
        if not res:
            continue

        fc = res["failure_cases"]
        lines.extend(
            [
                f"## Split: `{match}`",
                "",
                f"- Total clients: {fc['total_clients']:,}",
                f"- Wrong action: {fc['total_wrong']:,} ({100 * fc['total_wrong'] / max(fc['total_clients'], 1):.1f}%)",
                f"- Wrong but low regret (≤0.01): {fc['wrong_low_regret']:,}",
                f"- Wrong moderate regret (0.01–0.05): {fc['wrong_moderate_regret']:,}",
                f"- Wrong high regret (>0.05): {fc['wrong_high_regret']:,}",
                "",
                "### Dataset-specific failures (wrong predictions)",
                "",
            ]
        )
        bd = res["by_dataset"]
        if len(bd):
            lines.append("| Dataset | # wrong | Mean regret |")
            lines.append("|---------|--------:|------------:|")
            for _, r in bd.head(8).iterrows():
                lines.append(f"| {r['dataset']} | {int(r['n_wrong'])} | {r['mean_regret']:.4f} |")
            lines.append("")

        bl = res["by_lambda"]
        if len(bl):
            lines.extend(["### Lambda-specific failures", "", "| λ | # wrong | Mean regret |", "|---|--------:|------------:|"])
            for _, r in bl.iterrows():
                lines.append(f"| {r['lambda']} | {int(r['n_wrong'])} | {r['mean_regret']:.4f} |")
            lines.append("")

        cp = res["p0_p2_p5_confusion"]
        if len(cp):
            lines.extend(["### P0/P2/P5 confusion (top pairs by regret)", ""])
            lines.append("| Oracle | Predicted | Count | Mean regret |")
            lines.append("|--------|-----------|------:|------------:|")
            for _, r in cp.head(10).iterrows():
                lines.append(f"| {r['oracle']} | {r['predicted']} | {int(r['count'])} | {r['mean_regret']:.4f} |")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
