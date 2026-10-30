"""Phase-1 figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from paper2.src.config import FIGURES_DIR


def _setup():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams["figure.dpi"] = 150


def plot_ablation_regret(metrics: pd.DataFrame, out_path: Path, split: str = "random", model: str = "gradient_boosting"):
    _setup()
    sub = metrics[(metrics.split_name == split) & (metrics.model_name == model)].copy()
    if sub.empty:
        sub = metrics[(metrics.split_name == split) & (metrics.model_name == "random_forest")].copy()
    sub = sub.sort_values("mean_regret")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(sub["feature_set"], sub["mean_regret"], color="#C44E52")
    ax.set_xlabel("Mean utility regret")
    ax.set_title(f"Phase-1 ablation regret ({split}, {model})")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_accuracy(metrics: pd.DataFrame, out_path: Path, split: str = "random", model: str = "gradient_boosting"):
    _setup()
    sub = metrics[(metrics.split_name == split) & (metrics.model_name == model)].copy()
    if sub.empty:
        sub = metrics[(metrics.split_name == split) & (metrics.model_name == "random_forest")].copy()
    sub = sub.sort_values("accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(sub["feature_set"], sub["accuracy"], color="#4C72B0")
    ax.set_xlabel("Action accuracy")
    ax.set_title(f"Phase-1 ablation accuracy ({split})")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_geometry_vs_probe(metrics: pd.DataFrame, out_path: Path, model: str = "gradient_boosting"):
    _setup()
    rand = metrics[metrics.split_name == "random"]
    sets = ["context_only", "context_geometry", "context_probes", "context_geometry_probes"]
    sub = rand[(rand.feature_set.isin(sets)) & (rand.model_name == model)]
    if sub.empty:
        sub = rand[rand.feature_set.isin(sets) & (rand.model_name == "random_forest")]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sub))
    ax.bar(x, sub["mean_regret"], color=["#999999", "#55A868", "#DD8452", "#4C72B0"])
    ax.set_xticks(x)
    ax.set_xticklabels(sub["feature_set"], rotation=20, ha="right")
    ax.set_ylabel("Mean utility regret")
    ax.set_title("Geometry vs probe regret (random split)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_loo_by_feature_set(metrics: pd.DataFrame, out_path: Path, prefix: str, title: str):
    _setup()
    sub = metrics[metrics.split_name.str.startswith(prefix)].copy()
    if sub.empty:
        return
    sub["fold"] = sub["split_name"].str.replace(prefix, "", regex=False)
    pivot = sub[sub.model_name.isin(["gradient_boosting", "random_forest"])].pivot_table(
        index="fold", columns="feature_set", values="mean_regret", aggfunc="min"
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax, rot=45)
    ax.set_ylabel("Mean utility regret")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_geometry_feature_importance(importances: Dict[str, float], out_path: Path):
    _setup()
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
    ax.barh(names[::-1], vals[::-1], color="#55A868")
    ax.set_xlabel("Importance")
    ax.set_title("Geometry feature importance (full model, geometry cols only)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_phase1_figures(metrics: pd.DataFrame, geo_importance: Optional[Dict[str, float]] = None) -> Dict[str, Path]:
    out = FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "ablation_regret": out / "phase1_ablation_regret.png",
        "ablation_accuracy": out / "phase1_ablation_accuracy.png",
        "geometry_vs_probe": out / "geometry_vs_probe_regret.png",
        "loo_dataset": out / "leave_one_dataset_geometry.png",
        "loo_lambda": out / "leave_one_lambda_geometry.png",
        "geo_importance": out / "geometry_feature_importance.png",
    }
    plot_ablation_regret(metrics, paths["ablation_regret"])
    plot_ablation_accuracy(metrics, paths["ablation_accuracy"])
    plot_geometry_vs_probe(metrics, paths["geometry_vs_probe"])
    plot_loo_by_feature_set(metrics, paths["loo_dataset"], "loo_dataset_", "Leave-one-dataset regret by feature set")
    plot_loo_by_feature_set(metrics, paths["loo_lambda"], "loo_lambda_", "Leave-one-lambda regret by feature set")
    if geo_importance:
        plot_geometry_feature_importance(geo_importance, paths["geo_importance"])
    return paths
