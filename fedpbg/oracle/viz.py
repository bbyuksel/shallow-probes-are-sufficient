"""
Figures for oracle sparsity validation study.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LEVEL_ORDER = ["P0", "P1", "P2", "P3", "P4", "P5"]


def _filter_primary_lambda(df: pd.DataFrame, lambda_val: float = 0.3) -> pd.DataFrame:
    if "lambda" in df.columns:
        return df[df["lambda"] == lambda_val].copy()
    return df.copy()


def plot_oracle_budget_distribution_by_dataset(
    summary_df: pd.DataFrame,
    client_oracle_df: pd.DataFrame,
    out_path: Path,
    lambda_val: float = 0.3,
) -> None:
    """Stacked bar: fraction of clients at each oracle level per dataset."""
    sub = _filter_primary_lambda(client_oracle_df, lambda_val)
    counts = (
        sub.groupby(["dataset", "oracle_level"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LEVEL_ORDER, fill_value=0)
    )
    frac = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    frac.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", edgecolor="black")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Fraction of clients")
    ax.set_title(f"Oracle Budget Distribution by Dataset (λ={lambda_val})")
    ax.legend(title="Level", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_p0_ratio_by_dataset(summary_df: pd.DataFrame, out_path: Path, lambda_val: float = 0.3) -> None:
    sub = _filter_primary_lambda(summary_df, lambda_val)
    agg = sub.groupby("dataset")["p0_ratio"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(8, 5))
    plt.bar(agg["dataset"], agg["mean"], yerr=agg["std"], capsize=4, color="steelblue", edgecolor="black")
    plt.axhline(0.5, color="crimson", linestyle="--", label="Majority threshold (50%)")
    plt.ylabel("P0 oracle ratio")
    plt.xlabel("Dataset")
    plt.title(f"P0 (Global-Only) Oracle Ratio by Dataset (λ={lambda_val})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_oracle_vs_uniform_utility_gap(summary_df: pd.DataFrame, out_path: Path, lambda_val: float = 0.3) -> None:
    sub = _filter_primary_lambda(summary_df, lambda_val)
    agg = sub.groupby("dataset")["oracle_vs_uniform_rel_gap"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(8, 5))
    plt.bar(
        agg["dataset"],
        100 * agg["mean"],
        yerr=100 * agg["std"],
        capsize=4,
        color="darkorange",
        edgecolor="black",
    )
    plt.ylabel("Relative utility gap (%)")
    plt.xlabel("Dataset")
    plt.title(f"Oracle vs Best Uniform — Relative Utility Gap (λ={lambda_val})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_personalize_all_penalty(summary_df: pd.DataFrame, out_path: Path, lambda_val: float = 0.3) -> None:
    sub = _filter_primary_lambda(summary_df, lambda_val)
    agg = sub.groupby("dataset").agg(
        oracle=("oracle_utility", "mean"),
        personalize_all=("personalize_all_utility", "mean"),
    ).reset_index()

    x = range(len(agg))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], agg["oracle"], width, label="Oracle", color="seagreen")
    plt.bar([i + width / 2 for i in x], agg["personalize_all"], width, label="Personalize-all (P5)", color="indianred")
    plt.xticks(list(x), agg["dataset"], rotation=15)
    plt.ylabel("Mean utility")
    plt.title(f"Oracle vs Personalize-All Utility (λ={lambda_val})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_hardware_oracle_distribution(
    client_oracle_df: pd.DataFrame,
    out_path: Path,
    lambda_val: float = 0.3,
) -> None:
    sub = _filter_primary_lambda(client_oracle_df, lambda_val)
    sub = sub.copy()
    sub["oracle_level_num"] = sub["oracle_level"].str.replace("P", "").astype(int)

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=sub, x="hardware", y="oracle_level_num", order=["LOW", "MEDIUM", "HIGH"])
    sns.stripplot(
        data=sub,
        x="hardware",
        y="oracle_level_num",
        order=["LOW", "MEDIUM", "HIGH"],
        color="black",
        alpha=0.3,
        jitter=True,
    )
    plt.yticks(range(6), LEVEL_ORDER)
    plt.xlabel("Hardware profile")
    plt.ylabel("Oracle personalization level")
    plt.title(f"Hardware vs Oracle Budget (λ={lambda_val})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_validation_figures(
    summary_df: pd.DataFrame,
    client_oracle_df: pd.DataFrame,
    out_dir: Path,
    lambda_val: float = 0.3,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_oracle_budget_distribution_by_dataset(
        summary_df, client_oracle_df, out_dir / "oracle_budget_distribution_by_dataset.png", lambda_val
    )
    plot_p0_ratio_by_dataset(summary_df, out_dir / "p0_ratio_by_dataset.png", lambda_val)
    plot_oracle_vs_uniform_utility_gap(summary_df, out_dir / "oracle_vs_uniform_utility_gap.png", lambda_val)
    plot_personalize_all_penalty(summary_df, out_dir / "personalize_all_penalty.png", lambda_val)
    plot_hardware_oracle_distribution(
        client_oracle_df, out_dir / "hardware_oracle_distribution.png", lambda_val
    )
