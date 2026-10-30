"""Phase-3 online figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from paper2.src.config import FIGURES_DIR, LEVELS


def _setup():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams["figure.dpi"] = 150


def plot_regret_vs_baselines(summary: pd.DataFrame, out_path: Path):
    _setup()
    agg = summary.groupby("policy")["mean_regret"].mean().reset_index()
    order = ["oracle", "fedpba", "best_uniform", "global_only", "ditto", "personalize_all"]
    agg["policy"] = pd.Categorical(agg["policy"], categories=order, ordered=True)
    agg = agg.sort_values("policy")
    colors = {"oracle": "#2ca02c", "fedpba": "#1f77b4", "best_uniform": "#ff7f0e", "global_only": "#9467bd"}
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        agg["policy"].astype(str),
        agg["mean_regret"],
        color=[colors.get(p, "#888888") for p in agg["policy"]],
    )
    ax.set_ylabel("Mean utility regret vs oracle")
    ax.set_title("Phase-3 online: regret vs baselines (aggregated)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_utility_vs_baselines(summary: pd.DataFrame, out_path: Path):
    _setup()
    agg = summary.groupby("policy")["mean_utility"].mean().reset_index()
    order = ["oracle", "fedpba", "best_uniform", "global_only", "ditto", "personalize_all"]
    agg["policy"] = pd.Categorical(agg["policy"], categories=order, ordered=True)
    agg = agg.sort_values("policy")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(agg["policy"].astype(str), agg["mean_utility"], color="#4C72B0")
    ax.set_ylabel("Mean utility")
    ax.set_title("Phase-3 online: utility vs baselines")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_action_distribution(detail: pd.DataFrame, out_path: Path):
    _setup()
    fed_frac = detail["fedpba_action"].value_counts(normalize=True).reindex(LEVELS, fill_value=0)
    ora_frac = detail["oracle_action"].value_counts(normalize=True).reindex(LEVELS, fill_value=0)
    x = np.arange(len(LEVELS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, ora_frac.values, w, label="Oracle", color="#2ca02c", alpha=0.85)
    ax.bar(x + w / 2, fed_frac.values, w, label="FedPBA online", color="#1f77b4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(LEVELS)
    ax.set_ylabel("Action fraction")
    ax.set_title("Phase-3 online action distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_probe_cost_vs_oracle(summary: pd.DataFrame, out_path: Path):
    _setup()
    fed = summary[summary.policy == "fedpba"].iloc[0] if len(summary[summary.policy == "fedpba"]) else None
    if fed is None:
        return
    labels = ["Probe trainings", "Oracle trainings", "FedPBA extra (P3–P5)"]
    vals = [
        fed.get("probe_trainings", 0),
        fed.get("oracle_trainings", 0),
        fed.get("fedpba_extra_trainings", 0),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#d62728"])
    ax.set_ylabel("Count (per session, representative)")
    ax.set_title("Probe vs oracle training cost")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_breakdown(summary: pd.DataFrame, out_path: Path):
    _setup()
    sub = summary[summary.policy.isin(["fedpba", "best_uniform", "oracle"])].copy()
    pivot = sub.pivot_table(index="dataset", columns="policy", values="mean_regret", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylabel("Mean regret vs oracle")
    ax.set_title("Phase-3 online regret by dataset")
    ax.legend(title="Policy")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_phase3_figures(summary: pd.DataFrame, detail: pd.DataFrame) -> Dict[str, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "regret": FIGURES_DIR / "phase3_online_regret_vs_baselines.png",
        "utility": FIGURES_DIR / "phase3_online_utility_vs_baselines.png",
        "actions": FIGURES_DIR / "phase3_online_action_distribution.png",
        "probe_cost": FIGURES_DIR / "phase3_online_probe_cost_vs_oracle.png",
        "dataset": FIGURES_DIR / "phase3_online_dataset_breakdown.png",
    }
    plot_regret_vs_baselines(summary, paths["regret"])
    plot_utility_vs_baselines(summary, paths["utility"])
    plot_action_distribution(detail, paths["actions"])
    plot_probe_cost_vs_oracle(summary, paths["probe_cost"])
    plot_dataset_breakdown(summary, paths["dataset"])
    return paths
