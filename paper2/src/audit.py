"""Dataset audit report for Paper-2 Phase-0."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from paper2.src.config import REPORTS_DIR


def _pct_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c in df.columns:
            miss = df[c].isna().mean() * 100
            rows.append({"feature": c, "missing_pct": round(miss, 2)})
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


def write_dataset_audit(df: pd.DataFrame, out_path: Optional[Path] = None) -> Path:
    out = out_path or REPORTS_DIR / "oracle_dataset_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(df)
    n_datasets = df["dataset"].nunique()
    n_clients = df.groupby(["config_id", "client_id"]).ngroups
    n_configs = df["config_id"].nunique()

    action_dist = df["oracle_action"].value_counts(normalize=True).sort_index()
    action_counts = df["oracle_action"].value_counts().sort_index()

    feature_cols = [c for c in df.columns if c not in ("oracle_action", "oracle_utility", "config_id")]
    missing = _pct_missing(df, feature_cols)

    ds_action = pd.crosstab(df["dataset"], df["oracle_action"], normalize="index") * 100
    lam_action = pd.crosstab(df["lambda"], df["oracle_action"], normalize="index") * 100

    max_frac = action_counts.max() / action_counts.sum()
    imbalance = "yes" if max_frac > 0.5 else "moderate" if max_frac > 0.35 else "no"

    lines = [
        "# Oracle Decision Dataset Audit",
        "",
        "## Overview",
        "",
        f"- **Rows:** {n_rows:,}",
        f"- **Configurations:** {n_configs}",
        f"- **Unique clients (per config):** {n_clients:,}",
        f"- **Datasets:** {n_datasets} — {', '.join(sorted(df['dataset'].unique()))}",
        f"- **Lambda values:** {sorted(df['lambda'].unique())}",
        f"- **Alpha values:** {sorted(df['alpha'].unique())}",
        "",
        "## Oracle action distribution",
        "",
        "| Action | Count | Fraction |",
        "|--------|------:|---------:|",
    ]
    for act in action_counts.index:
        lines.append(f"| {act} | {action_counts[act]:,} | {action_dist[act]:.3f} |")

    lines.extend(
        [
            "",
            f"**Target imbalanced?** {imbalance} (majority class fraction = {max_frac:.3f})",
            "",
            "## Missing feature rates",
            "",
            "| Feature | Missing % |",
            "|---------|----------:|",
        ]
    )
    for _, r in missing.iterrows():
        lines.append(f"| {r['feature']} | {r['missing_pct']:.1f} |")

    lines.extend(["", "## Oracle action by dataset (row %)", "", ds_action.round(1).to_markdown()])

    lines.extend(["", "## Oracle action by lambda (row %)", "", lam_action.round(1).to_markdown()])

    geom_cols = [
        "gradient_norm",
        "gradient_variance",
        "cosine_to_global_update",
        "representation_distance",
        "prototype_distance",
    ]
    geom_missing = missing[missing["feature"].isin(geom_cols)]["missing_pct"].mean()
    lines.extend(
        [
            "",
            "## Geometry status",
            "",
            f"- `heterogeneity` / `client_drift_score`: available (Paper-1 proxy)",
            f"- Other geometry features: **not implemented** (mean missing = {geom_missing:.1f}%)",
            "",
            "## Client data features",
            "",
            "- `num_samples`, `label_entropy`, `majority_class_ratio`, `class_count`, `class_balance_score`: "
            "**not in Paper-1 caches** (all NaN placeholders)",
        ]
    )

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
