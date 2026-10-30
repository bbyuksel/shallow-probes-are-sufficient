"""FedPBA Phase-2 figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from paper2.src.config import FIGURES_DIR, LEVELS


def _setup():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams["figure.dpi"] = 150


def plot_regret_vs_baselines(
    summary: pd.DataFrame,
    out_path: Path,
    split: str = "random",
    feature_set: str = "context_probes",
    model_name: Optional[str] = None,
):
    _setup()
    sub = summary[(summary.split_name == split) & (summary.feature_set == feature_set)].copy()
    if model_name:
        sub = sub[sub.model_name == model_name]
    else:
        fed = sub[sub.policy == "fedpba"].sort_values("mean_regret")
        if len(fed):
            model_name = fed.iloc[0]["model_name"]
            sub = sub[sub.model_name == model_name]

    order = ["oracle", "fedpba", "best_uniform", "global_only", "ditto", "random", "personalize_all"]
    plot = sub[sub.policy.isin(order)].copy()
    plot["policy"] = pd.Categorical(plot["policy"], categories=order, ordered=True)
    plot = plot.sort_values("policy")

    colors = {
        "oracle": "#2ca02c",
        "fedpba": "#1f77b4",
        "best_uniform": "#ff7f0e",
        "global_only": "#9467bd",
        "ditto": "#8c564b",
        "random": "#7f7f7f",
        "personalize_all": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = plot["policy"].astype(str).tolist()
    ax.bar(labels, plot["mean_regret"], color=[colors.get(p, "#333333") for p in plot["policy"]])
    ax.set_ylabel("Mean utility regret vs oracle")
    ax.set_title(f"FedPBA regret vs baselines ({split})")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_action_distribution(
    summary: pd.DataFrame,
    out_path: Path,
    split: str = "random",
    feature_set: str = "context_probes",
):
    _setup()
    sub = summary[
        (summary.split_name == split) & (summary.feature_set == feature_set) & (summary.policy == "fedpba")
    ]
    if sub.empty:
        return
    row = sub.sort_values("mean_regret").iloc[0]
    fed_fracs = [row.get(f"action_frac_{lv}", 0) for lv in LEVELS]
    oracle_fracs = [row.get(f"oracle_frac_{lv}", 0) for lv in LEVELS]

    x = np.arange(len(LEVELS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, oracle_fracs, w, label="Oracle", color="#2ca02c", alpha=0.85)
    ax.bar(x + w / 2, fed_fracs, w, label="FedPBA", color="#1f77b4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(LEVELS)
    ax.set_ylabel("Action fraction")
    ax.set_title(f"Action distribution: FedPBA vs oracle ({split})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_loo_regret(
    summary: pd.DataFrame,
    out_path: Path,
    prefix: str,
    title: str,
    feature_set: str = "context_probes",
    policies: Optional[List[str]] = None,
):
    _setup()
    policies = policies or ["fedpba", "best_uniform", "global_only"]
    sub = summary[
        summary.split_name.str.startswith(prefix)
        & (summary.feature_set == feature_set)
        & (summary.policy.isin(policies))
    ].copy()
    if sub.empty:
        return
    sub["fold"] = sub["split_name"].str.replace(prefix, "", regex=False)
    # Best model per fold for fedpba
    fed = sub[sub.policy == "fedpba"].sort_values("mean_regret").groupby("fold", as_index=False).first()
    others = sub[sub.policy != "fedpba"]
    plot_df = pd.concat([fed, others], ignore_index=True)

    pivot = plot_df.pivot_table(index="fold", columns="policy", values="mean_regret", aggfunc="min")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, rot=45)
    ax.set_ylabel("Mean utility regret")
    ax.set_title(title)
    ax.legend(title="Policy")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_regret(
    ablation_summary: pd.DataFrame,
    out_path: Path,
    split: str = "random",
    model_name: str = "gradient_boosting",
):
    _setup()
    sub = ablation_summary[
        (ablation_summary.split_name == split)
        & (ablation_summary.policy == "fedpba")
        & (ablation_summary.model_name == model_name)
    ].copy()
    if sub.empty:
        sub = ablation_summary[
            (ablation_summary.split_name == split) & (ablation_summary.policy == "fedpba")
        ].sort_values("mean_regret").groupby("feature_set", as_index=False).first()
    sub = sub.sort_values("mean_regret")
    labels = {
        "context_only": "A: context",
        "context_cheap_data": "B: +cheap data",
        "context_geometry": "C: +geometry",
        "context_probes": "D: +probes",
        "context_geometry_probes": "E: +geo+probes",
    }
    sub["label"] = sub["feature_set"].map(labels).fillna(sub["feature_set"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(sub["label"], sub["mean_regret"], color="#C44E52")
    ax.set_xlabel("Mean utility regret")
    ax.set_title(f"FedPBA ablation regret ({split})")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    detail: pd.DataFrame,
    out_path: Path,
    split: str = "random",
    feature_set: str = "context_probes",
):
    _setup()
    sub = detail[(detail.split_name == split) & (detail.feature_set == feature_set)].copy()
    if sub.empty:
        return
    if "model_name" in sub.columns:
        model = sub.groupby("model_name").size().idxmax()
        sub = sub[sub.model_name == model]

    le = LabelEncoder()
    le.fit(LEVELS)
    y_true = le.transform(sub["oracle_action"])
    y_pred = le.transform(sub["fedpba_action"])
    cm = np.zeros((len(LEVELS), len(LEVELS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LEVELS, yticklabels=LEVELS, ax=ax)
    ax.set_xlabel("FedPBA predicted")
    ax.set_ylabel("Oracle action")
    ax.set_title(f"FedPBA confusion matrix ({split})")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_phase2_figures(
    summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    detail: pd.DataFrame,
    feature_set: str = "context_probes",
    model_name: Optional[str] = None,
) -> Dict[str, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_name = model_name or summary[
        (summary.split_name == "random") & (summary.policy == "fedpba") & (summary.feature_set == feature_set)
    ].sort_values("mean_regret").iloc[0]["model_name"]

    paths = {
        "regret_vs_baselines": FIGURES_DIR / "fedpba_regret_vs_baselines.png",
        "action_distribution": FIGURES_DIR / "fedpba_action_distribution.png",
        "cross_dataset_regret": FIGURES_DIR / "fedpba_cross_dataset_regret.png",
        "cross_lambda_regret": FIGURES_DIR / "fedpba_cross_lambda_regret.png",
        "ablation_regret": FIGURES_DIR / "fedpba_ablation_regret.png",
        "confusion_matrix": FIGURES_DIR / "fedpba_confusion_matrix.png",
    }

    plot_regret_vs_baselines(summary, paths["regret_vs_baselines"], feature_set=feature_set, model_name=model_name)
    plot_action_distribution(summary, paths["action_distribution"], feature_set=feature_set)
    plot_loo_regret(
        summary,
        paths["cross_dataset_regret"],
        "loo_dataset_",
        "Cross-dataset regret (leave-one-dataset-out)",
        feature_set=feature_set,
    )
    plot_loo_regret(
        summary,
        paths["cross_lambda_regret"],
        "loo_lambda_",
        "Cross-lambda regret (leave-one-lambda-out)",
        feature_set=feature_set,
    )
    plot_ablation_regret(ablation_summary, paths["ablation_regret"], model_name=model_name)
    plot_confusion_matrix(detail, paths["confusion_matrix"], feature_set=feature_set)
    return paths
