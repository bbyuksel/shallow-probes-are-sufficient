"""Report writers for Paper-2 Phase-0."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from paper2.src.config import REPORTS_DIR


def write_feature_importance_summary(
    best_result: Optional[Dict],
    out_path: Optional[Path] = None,
) -> Path:
    out = out_path or REPORTS_DIR / "feature_importance_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Feature Importance Summary",
        "",
        "Best tree-based model on random split (selected by lowest mean utility regret).",
        "",
    ]

    if not best_result or "model" not in best_result:
        lines.append("No tree model result available.")
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    model = best_result["model"]
    inner = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    names = best_result.get("used_cols", [])
    lines.append(f"**Model:** {best_result['model_name']}")
    lines.append("")

    if not hasattr(inner, "feature_importances_"):
        lines.append("Selected model does not expose feature importances.")
    else:
        imp = inner.feature_importances_
        ranking = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)
        lines.extend(["## Top features", "", "| Rank | Feature | Importance |", "|-----:|---------|----------:|"])
        for i, (n, v) in enumerate(ranking[:15], 1):
            lines.append(f"| {i} | {n} | {v:.4f} |")

        groups = {
            "lambda": ["lambda"],
            "dataset": ["dataset"],
            "alpha": ["alpha"],
            "label entropy / client data": [
                "label_entropy",
                "num_samples",
                "majority_class_ratio",
                "class_count",
                "class_balance_score",
            ],
            "hardware": ["hardware_profile", "hardware_score", "hardware_multiplier"],
            "p0 probe": ["p0_macro_f1", "cost_p0"],
            "p1/p2 probes": ["p1_macro_f1", "p2_macro_f1", "cost_p1", "cost_p2"],
            "geometry": [
                "heterogeneity",
                "client_drift_score",
                "gradient_norm",
                "gradient_variance",
                "cosine_to_global_update",
                "representation_distance",
                "prototype_distance",
            ],
        }
        group_imp = {}
        for g, feats in groups.items():
            group_imp[g] = sum(v for n, v in ranking if n in feats)

        lines.extend(["", "## Feature group totals", ""])
        for g, total in sorted(group_imp.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{g}**: {total:.4f}")

        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                _interpret_groups(group_imp, ranking),
            ]
        )

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _interpret_groups(group_imp: Dict[str, float], ranking: List) -> str:
    top_feat = ranking[0][0] if ranking else "unknown"
    top_group = max(group_imp, key=group_imp.get) if group_imp else "unknown"
    geom_total = group_imp.get("geometry", 0)
    client_data = group_imp.get("label entropy / client data", 0)
    parts = [
        f"- Strongest single feature: `{top_feat}`.",
        f"- Highest-importance group: **{top_group}**.",
    ]
    if client_data < 0.01:
        parts.append("- Client-data features are absent (NaN); they contribute nothing in Phase-0.")
    if geom_total > 0 and ranking[0][0] in ("heterogeneity", "client_drift_score"):
        parts.append("- Only available geometry proxy is heterogeneity / client drift; dedicated geometry features are not yet implemented.")
    elif geom_total < 0.05:
        parts.append("- Geometry features (beyond heterogeneity) are not yet available.")
    return "\n".join(parts)


def write_feasibility_report(
    metrics_df: pd.DataFrame,
    agg: Dict,
    df: pd.DataFrame,
    best_result: Optional[Dict],
    out_path: Optional[Path] = None,
) -> Path:
    out = out_path or REPORTS_DIR / "phase0_feasibility_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    rand = metrics_df[metrics_df["split_name"] == "random"]
    maj = rand[rand["model_name"] == "majority"].iloc[0]
    bun = rand[rand["model_name"] == "best_uniform"].iloc[0]
    glo = rand[rand["model_name"] == "global_only"].iloc[0]
    rnd = rand[rand["model_name"] == "random"].iloc[0]
    orc = rand[rand["model_name"] == "oracle"].iloc[0]

    ml_models = rand[~rand["model_name"].isin(["majority", "best_uniform", "global_only", "random", "oracle"])]
    best_ml = ml_models.loc[ml_models["mean_regret"].idxmin()]
    best_acc_ml = ml_models.loc[ml_models["accuracy"].idxmax()]

    loo_ds = agg.get("loo_dataset", {})
    loo_lam = agg.get("loo_lambda", {})
    loo_alpha = agg.get("loo_alpha", {})

    q1 = best_ml["accuracy"] > maj["accuracy"]
    q2 = best_ml["mean_regret"] < bun["mean_regret"]
    loo_q3 = loo_ds.get("mean_regret", np.inf) < bun["mean_regret"]
    loo_q4 = loo_lam.get("mean_regret", np.inf) < bun["mean_regret"]

    # Cheap features only - geometry mostly NaN except heterogeneity
    geom_avail = df["gradient_norm"].isna().mean() < 0.5
    q5 = best_ml["mean_regret"] < bun["mean_regret"]
    q6 = not geom_avail

    strong = q2 and q1 and loo_q3
    if q2 and loo_q3 and loo_q4:
        rec = "continue"
        rec_detail = "Proceed to Phase-1 learned allocator with cheap probes; add geometry when implemented."
    elif q2 and not loo_q3:
        rec = "add geometry"
        rec_detail = "Utility regret improves in-distribution but cross-dataset generalization is weak; geometry/client-data features may be needed."
    elif best_ml["accuracy"] < 0.35 and best_ml["mean_regret"] > bun["mean_regret"]:
        rec = "redesign target"
        rec_detail = "Neither action accuracy nor regret beats strong baselines; consider coarser action space or regression on utility."
    elif best_ml["mean_regret"] > glo["mean_regret"]:
        rec = "stop"
        rec_detail = "Learned predictor does not beat global-only; deployable policy may not be feasible with current features."
    else:
        rec = "continue"
        rec_detail = "Mixed signals — proceed cautiously with expanded features and utility-aware training."

    lines = [
        "# Phase-0 Feasibility Report",
        "",
        "**Paper-2 Phase-0** — Oracle decision prediction from cheap client features.",
        "",
        "## Executive summary",
        "",
        f"- Best ML model (random split, by regret): **{best_ml['model_name']}**",
        f"- Action accuracy: {best_ml['accuracy']:.3f} (majority baseline: {maj['accuracy']:.3f})",
        f"- Mean utility regret: {best_ml['mean_regret']:.4f} (best uniform: {bun['mean_regret']:.4f}, global-only: {glo['mean_regret']:.4f})",
        f"- 95% bootstrap CI: [{best_ml['regret_ci_lo']:.4f}, {best_ml['regret_ci_hi']:.4f}]",
        "",
        "## Random-split metrics (key models)",
        "",
        "| Model | Accuracy | Macro-F1 | Top-2 Acc | Mean regret | Median regret |",
        "|-------|---------:|---------:|----------:|------------:|--------------:|",
    ]
    for _, r in rand.sort_values("mean_regret").iterrows():
        t2 = f"{r['top2_accuracy']:.3f}" if not pd.isna(r.get("top2_accuracy")) else "—"
        lines.append(
            f"| {r['model_name']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {t2} | "
            f"{r['mean_regret']:.4f} | {r['median_regret']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Feasibility questions",
            "",
            f"1. **Can oracle actions be predicted better than majority baseline?** "
            f"{'Yes' if q1 else 'No'} — best ML accuracy {best_ml['accuracy']:.3f} vs majority {maj['accuracy']:.3f}.",
            f"2. **Does the predictor reduce utility regret vs best uniform?** "
            f"{'Yes' if q2 else 'No'} — {best_ml['mean_regret']:.4f} vs {bun['mean_regret']:.4f}.",
            f"3. **Does it generalize across datasets?** "
            f"{'Yes' if loo_q3 else 'No'} — LOO-dataset mean regret {loo_ds.get('mean_regret', float('nan')):.4f} "
            f"(accuracy {loo_ds.get('accuracy', float('nan')):.3f}, model {loo_ds.get('model_used', 'n/a')}, n_folds={loo_ds.get('n_folds', 0)}).",
            f"4. **Does it generalize across lambda?** "
            f"{'Yes' if loo_q4 else 'No'} — LOO-lambda mean regret {loo_lam.get('mean_regret', float('nan')):.4f} "
            f"(model {loo_lam.get('model_used', 'n/a')}).",
            f"5. **Are cheap features sufficient?** "
            f"{'Promising' if q5 else 'Insufficient so far'} — regret vs best uniform and global-only are the decisive checks.",
            f"6. **Are geometry features needed?** "
            f"{'Likely yes' if q6 else 'Unclear'} — only heterogeneity proxy available; other geometry columns are NaN.",
            f"7. **Strong enough to justify Paper-2?** "
            f"{'Preliminary yes' if strong else 'Not yet'} — utility regret is the primary criterion.",
            "",
            f"8. **Recommended next step:** **{rec}** — {rec_detail}",
            "",
            "## Conservative notes",
            "",
            "- Action accuracy can remain low while regret is acceptable if mispredictions stay near oracle in utility.",
            "- Phase-0 uses frozen Paper-1 caches only; no new FL training was performed.",
            "- Client-data and most geometry features are placeholders (NaN).",
            f"- Highest-accuracy model: {best_acc_ml['model_name']} ({best_acc_ml['accuracy']:.3f}); "
            f"may differ from lowest-regret model.",
            "",
            "## LOO aggregate summaries",
            "",
            f"- **Leave-one-dataset-out:** {loo_ds}",
            f"- **Leave-one-alpha-out:** {loo_alpha}",
            f"- **Leave-one-lambda-out:** {loo_lam}",
        ]
    )

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
