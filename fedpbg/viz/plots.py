"""
Generate feasibility study figures in the results folder.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LEVEL_ORDER = ["P0", "P1", "P2", "P3", "P4", "P5"]


def _level_num(level: str) -> int:
    return int(level[1:])


def plot_heterogeneity_vs_budget(oracle_df: pd.DataFrame, out_path: Path, lambda_: float = 0.3) -> None:
    """Scatter: heterogeneity score vs oracle personalization level."""
    sub = oracle_df[oracle_df["lambda"] == lambda_].copy()
    sub["oracle_level_num"] = sub["oracle_level"].map(_level_num)

    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=sub,
        x="heterogeneity",
        y="oracle_level_num",
        scatter_kws={"alpha": 0.7, "s": 60},
        line_kws={"color": "crimson"},
    )
    plt.yticks(range(6), LEVEL_ORDER)
    plt.xlabel("Heterogeneity score H_i")
    plt.ylabel("Oracle personalization level")
    plt.title(f"Heterogeneity vs Oracle Budget (λ={lambda_})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_hardware_vs_budget(oracle_df: pd.DataFrame, out_path: Path, lambda_: float = 0.3) -> None:
    """Box/strip plot: hardware profile vs oracle level."""
    sub = oracle_df[oracle_df["lambda"] == lambda_].copy()
    sub["oracle_level_num"] = sub["oracle_level"].map(_level_num)

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=sub, x="hardware", y="oracle_level_num", order=["LOW", "MEDIUM", "HIGH"])
    sns.stripplot(
        data=sub, x="hardware", y="oracle_level_num", order=["LOW", "MEDIUM", "HIGH"],
        color="black", alpha=0.5, jitter=True,
    )
    plt.yticks(range(6), LEVEL_ORDER)
    plt.xlabel("Hardware profile")
    plt.ylabel("Oracle personalization level")
    plt.title(f"Hardware Profile vs Oracle Budget (λ={lambda_})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_accuracy_cost_pareto(client_df: pd.DataFrame, out_path: Path) -> None:
    """Pareto-style scatter of macro-F1 vs normalized cost across all runs."""
    plt.figure(figsize=(8, 6))
    for level in LEVEL_ORDER:
        sub = client_df[client_df["level"] == level]
        plt.scatter(
            sub["normalized_cost"],
            sub["macro_f1"],
            label=level,
            alpha=0.6,
            s=40,
        )
    plt.xlabel("Normalized cost")
    plt.ylabel("Macro-F1")
    plt.title("Accuracy–Cost Trade-off by Personalization Level")
    plt.legend(title="Level", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_client_budget_distribution(oracle_df: pd.DataFrame, out_path: Path, lambda_: float = 0.3) -> None:
    """Bar chart of oracle level counts across clients."""
    sub = oracle_df[oracle_df["lambda"] == lambda_]
    counts = sub["oracle_level"].value_counts().reindex(LEVEL_ORDER, fill_value=0)

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.xlabel("Oracle personalization level")
    plt.ylabel("Number of clients")
    plt.title(f"Client Oracle Budget Distribution (λ={lambda_})")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_all_plots(
    client_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    results_dir: Path,
    default_lambda: float = 0.3,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_heterogeneity_vs_budget(
        oracle_df, results_dir / "heterogeneity_vs_budget.png", default_lambda
    )
    plot_hardware_vs_budget(
        oracle_df, results_dir / "hardware_vs_budget.png", default_lambda
    )
    plot_accuracy_cost_pareto(client_df, results_dir / "accuracy_cost_pareto.png")
    plot_client_budget_distribution(
        oracle_df, results_dir / "client_budget_distribution.png", default_lambda
    )
