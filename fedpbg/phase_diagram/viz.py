"""
Paper-ready heatmaps for the personalization-cost phase diagram.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LEVEL_TO_NUM = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}


def _aggregate_phase_df(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds for each (dataset, alpha, lambda) cell."""
    keys = ["dataset", "dirichlet_alpha", "lambda"]
    numeric = df.groupby(keys).mean(numeric_only=True)
    if "best_uniform_level" in df.columns:
        mode_level = df.groupby(keys)["best_uniform_level"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
        )
        numeric["best_uniform_level"] = mode_level
    return numeric.reset_index()


def _pivot_heat(
    agg: pd.DataFrame,
    dataset: str,
    value_col: str,
    alphas: List[float],
    lambdas: List[float],
) -> pd.DataFrame:
    sub = agg[agg["dataset"] == dataset]
    pivot = sub.pivot_table(
        index="lambda",
        columns="dirichlet_alpha",
        values=value_col,
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=lambdas, columns=alphas)
    return pivot


def _plot_metric_heatmaps(
    agg: pd.DataFrame,
    datasets: List[str],
    alphas: List[float],
    lambdas: List[float],
    value_col: str,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    fmt: str = ".2f",
    as_percent: bool = False,
) -> None:
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    vmin, vmax = None, None
    if as_percent:
        vmin, vmax = 0, 1

    for ax, ds in zip(axes[0], datasets):
        pivot = _pivot_heat(agg, ds, value_col, alphas, lambdas)
        data = pivot.values
        if as_percent:
            annot = pivot.map(lambda x: f"{100*x:.0f}%" if pd.notna(x) else "")
        else:
            annot = pivot.map(lambda x: format(x, fmt) if pd.notna(x) else "")

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=annot,
            fmt="",
            cbar_kws={"label": value_col},
            linewidths=0.5,
        )
        ax.set_title(ds)
        ax.set_xlabel("Dirichlet α")
        ax.set_ylabel("λ (cost weight)")

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_p0_ratio_heatmap(agg: pd.DataFrame, datasets: List[str], alphas, lambdas, out_path: Path):
    _plot_metric_heatmaps(
        agg, datasets, alphas, lambdas, "p0_ratio",
        "P0 (Global-Only) Oracle Ratio — Phase Diagram",
        out_path, cmap="Blues", as_percent=True,
    )


def plot_deep_ratio_heatmap(agg: pd.DataFrame, datasets: List[str], alphas, lambdas, out_path: Path):
    _plot_metric_heatmaps(
        agg, datasets, alphas, lambdas, "deep_ratio",
        "Deep Personalization (P3–P5) Oracle Ratio — Phase Diagram",
        out_path, cmap="Oranges", as_percent=True,
    )


def plot_oracle_uniform_gap_heatmap(agg: pd.DataFrame, datasets: List[str], alphas, lambdas, out_path: Path):
    agg = agg.copy()
    agg["uniform_gap_pct"] = 100 * agg["oracle_vs_uniform_rel_gap"]
    _plot_metric_heatmaps(
        agg, datasets, alphas, lambdas, "uniform_gap_pct",
        "Oracle vs Best Uniform — Relative Utility Gap (%)",
        out_path, cmap="YlOrRd", fmt=".1f",
    )


def plot_personalize_all_penalty_heatmap(agg: pd.DataFrame, datasets: List[str], alphas, lambdas, out_path: Path):
    agg = agg.copy()
    agg["p5_gap_pct"] = 100 * agg["oracle_vs_personalize_all_rel_gap"]
    # Clip extreme values for readable heatmap
    agg["p5_gap_pct"] = agg["p5_gap_pct"].clip(-200, 200)
    _plot_metric_heatmaps(
        agg, datasets, alphas, lambdas, "p5_gap_pct",
        "Oracle vs Personalize-All (P5) — Relative Utility Gap (%)",
        out_path, cmap="RdPu", fmt=".0f",
    )


def plot_best_uniform_level_heatmap(agg: pd.DataFrame, datasets: List[str], alphas, lambdas, out_path: Path):
    """Categorical heatmap: best uniform level as numeric 0–5."""
    agg = agg.copy()
    agg["best_uniform_num"] = agg["best_uniform_level"].map(LEVEL_TO_NUM)

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        pivot = _pivot_heat(agg, ds, "best_uniform_num", alphas, lambdas)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="tab10",
            vmin=0,
            vmax=5,
            annot=pivot.map(lambda x: f"P{int(x)}" if pd.notna(x) else ""),
            fmt="",
            cbar_kws={"label": "Best uniform level"},
            linewidths=0.5,
        )
        ax.set_title(ds)
        ax.set_xlabel("Dirichlet α")
        ax.set_ylabel("λ")

    fig.suptitle("Best Uniform Personalization Level — Phase Diagram", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_hardware_phase_interaction(agg: pd.DataFrame, out_path: Path) -> None:
    """
    Line plot: P0 ratio vs λ, split by hardware profile, averaged over alpha/dataset.
    Shows cost-hardware interaction on sparsity.
    """
    hw_cols = ["p0_ratio_LOW", "p0_ratio_MEDIUM", "p0_ratio_HIGH"]
    if not all(c in agg.columns for c in hw_cols):
        return

    melted_rows = []
    for _, row in agg.iterrows():
        for hw, col in zip(["LOW", "MEDIUM", "HIGH"], hw_cols):
            if col in row and pd.notna(row[col]):
                melted_rows.append({
                    "dataset": row["dataset"],
                    "lambda": row["lambda"],
                    "hardware": hw,
                    "p0_ratio": row[col],
                })
    if not melted_rows:
        return

    mdf = pd.DataFrame(melted_rows)
    summary = (
        mdf.groupby(["dataset", "lambda", "hardware"], as_index=False)["p0_ratio"]
        .mean()
    )

    datasets = sorted(summary["dataset"].unique())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = summary[summary["dataset"] == ds]
        for hw in ["LOW", "MEDIUM", "HIGH"]:
            hw_sub = sub[sub["hardware"] == hw].sort_values("lambda")
            ax.plot(hw_sub["lambda"], hw_sub["p0_ratio"], marker="o", label=hw)
        ax.set_xlabel("λ (cost weight)")
        ax.set_ylabel("P0 oracle ratio")
        ax.set_title(ds)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title="Hardware")
        ax.grid(alpha=0.3)

    fig.suptitle("Hardware × Cost Sensitivity: When Is Global-Only Optimal?", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_phase_diagram_figures(
    results_df: pd.DataFrame,
    out_dir: Path,
    alphas: List[float],
    lambdas: List[float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = _aggregate_phase_df(results_df)
    datasets = sorted(agg["dataset"].unique())

    plot_p0_ratio_heatmap(agg, datasets, alphas, lambdas, out_dir / "p0_ratio_heatmap_by_dataset.png")
    plot_deep_ratio_heatmap(agg, datasets, alphas, lambdas, out_dir / "deep_ratio_heatmap_by_dataset.png")
    plot_oracle_uniform_gap_heatmap(
        agg, datasets, alphas, lambdas, out_dir / "oracle_uniform_gap_heatmap_by_dataset.png"
    )
    plot_personalize_all_penalty_heatmap(
        agg, datasets, alphas, lambdas, out_dir / "personalize_all_penalty_heatmap_by_dataset.png"
    )
    plot_best_uniform_level_heatmap(
        agg, datasets, alphas, lambdas, out_dir / "best_uniform_level_heatmap_by_dataset.png"
    )
    plot_hardware_phase_interaction(agg, out_dir / "hardware_phase_interaction.png")
