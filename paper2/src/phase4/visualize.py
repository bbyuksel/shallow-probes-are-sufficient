"""Phase-4 figures with confidence intervals."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from paper2.src.config import FIGURES_DIR


def _setup():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams["figure.dpi"] = 150


def plot_regret_ci(policy_stats: pd.DataFrame, out_path: Path):
    _setup()
    sub = policy_stats.sort_values("mean_regret")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sub))
    ax.bar(x, sub["mean_regret"], color="#4C72B0", alpha=0.85)
    ax.errorbar(
        x,
        sub["mean_regret"],
        yerr=[sub["mean_regret"] - sub["regret_ci_lo"], sub["regret_ci_hi"] - sub["mean_regret"]],
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sub["policy"], rotation=20, ha="right")
    ax.set_ylabel("Mean regret vs oracle")
    ax.set_title("Phase-4: regret by policy (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_utility_ci(policy_stats: pd.DataFrame, out_path: Path):
    _setup()
    sub = policy_stats.sort_values("mean_utility", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sub))
    ax.bar(x, sub["mean_utility"], color="#55A868", alpha=0.85)
    ax.errorbar(
        x,
        sub["mean_utility"],
        yerr=[sub["mean_utility"] - sub["utility_ci_lo"], sub["utility_ci_hi"] - sub["mean_utility"]],
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sub["policy"], rotation=20, ha="right")
    ax.set_ylabel("Mean utility")
    ax.set_title("Phase-4: utility by policy (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_regret_by_dataset(ds_stats: pd.DataFrame, out_path: Path):
    _setup()
    fig, ax = plt.subplots(figsize=(9, 5))
    datasets = sorted(ds_stats["dataset"].unique())
    x = np.arange(len(datasets))
    w = 0.35
    fed = ds_stats[ds_stats.policy == "fedpba"].set_index("dataset").reindex(datasets)
    bu = ds_stats[ds_stats.policy == "best_uniform"].set_index("dataset").reindex(datasets)
    ax.bar(x - w / 2, fed["mean_regret"], w, label="FedPBA", color="#1f77b4")
    ax.bar(x + w / 2, bu["mean_regret"], w, label="Best uniform", color="#ff7f0e")
    ax.errorbar(
        x - w / 2,
        fed["mean_regret"],
        yerr=[fed["mean_regret"] - fed["regret_ci_lo"], fed["regret_ci_hi"] - fed["mean_regret"]],
        fmt="none",
        color="black",
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Mean regret")
    ax.set_title("Regret by dataset (95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_regret_by_lambda(lam_stats: pd.DataFrame, out_path: Path):
    _setup()
    fig, ax = plt.subplots(figsize=(8, 5))
    lams = sorted(lam_stats["lambda"].unique())
    x = np.arange(len(lams))
    w = 0.35
    fed = lam_stats[lam_stats.policy == "fedpba"].set_index("lambda").reindex(lams)
    bu = lam_stats[lam_stats.policy == "best_uniform"].set_index("lambda").reindex(lams)
    ax.bar(x - w / 2, fed["mean_regret"], w, label="FedPBA", color="#1f77b4")
    ax.bar(x + w / 2, bu["mean_regret"], w, label="Best uniform", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in lams])
    ax.set_xlabel("λ")
    ax.set_ylabel("Mean regret")
    ax.set_title("Regret by λ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_paired_fedpba_vs_uniform(detail: pd.DataFrame, out_path: Path):
    _setup()
    diff = detail["fedpba_utility"] - detail["best_uniform_utility"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diff.dropna(), bins=40, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", label="zero")
    ax.axvline(diff.mean(), color="green", linestyle="-", label=f"mean={diff.mean():.4f}")
    ax.set_xlabel("FedPBA utility − best uniform utility")
    ax.set_ylabel("Count")
    ax.set_title("Paired utility difference distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_probe_speedup(summary: pd.DataFrame, out_path: Path):
    _setup()
    fed = summary[summary.policy == "fedpba"].drop_duplicates(subset=["config_id"])
    if fed.empty:
        return
    ratio = fed["probe_sec"] / fed["oracle_levels_sec"].replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratio.dropna(), bins=20, color="#DD8452", alpha=0.85, edgecolor="white")
    ax.axvline(ratio.median(), color="black", linestyle="--", label=f"median={ratio.median():.2f}")
    ax.set_xlabel("Probe time / oracle time")
    ax.set_ylabel("Sessions")
    ax.set_title(f"Probe cost speedup (~{1/ratio.median():.1f}× faster)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_phase4_figures(
    policy_stats: pd.DataFrame,
    ds_stats: pd.DataFrame,
    lam_stats: pd.DataFrame,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
) -> Dict[str, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "regret_ci": FIGURES_DIR / "phase4_regret_ci_by_policy.png",
        "utility_ci": FIGURES_DIR / "phase4_utility_ci_by_policy.png",
        "regret_ds": FIGURES_DIR / "phase4_regret_by_dataset_ci.png",
        "regret_lam": FIGURES_DIR / "phase4_regret_by_lambda_ci.png",
        "paired": FIGURES_DIR / "phase4_fedpba_vs_uniform_paired.png",
        "speedup": FIGURES_DIR / "phase4_probe_cost_speedup.png",
    }
    plot_regret_ci(policy_stats, paths["regret_ci"])
    plot_utility_ci(policy_stats, paths["utility_ci"])
    plot_regret_by_dataset(ds_stats, paths["regret_ds"])
    plot_regret_by_lambda(lam_stats, paths["regret_lam"])
    plot_paired_fedpba_vs_uniform(detail, paths["paired"])
    plot_probe_speedup(summary, paths["speedup"])
    return paths
