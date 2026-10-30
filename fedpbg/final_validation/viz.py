"""
Figures for oracle regret final validation study.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASELINES = ["RANDOM", "BEST_UNIFORM", "PERSONALIZE_ALL", "GLOBAL_ONLY"]
COLORS = {
    "RANDOM": "#e74c3c",
    "BEST_UNIFORM": "#f39c12",
    "PERSONALIZE_ALL": "#9b59b6",
    "GLOBAL_ONLY": "#3498db",
}


def _baseline_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["policy"].isin(BASELINES)].copy()


def plot_regret_by_dataset(df: pd.DataFrame, out_path: Path) -> None:
    sub = _baseline_df(df)
    agg = sub.groupby(["dataset", "policy"])["regret"].mean().reset_index()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=agg, x="dataset", y="regret", hue="policy", palette=COLORS)
    plt.ylabel("Mean regret (Oracle − Policy)")
    plt.xlabel("Dataset")
    plt.title("Oracle Regret by Dataset")
    plt.legend(title="Policy", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_regret_by_lambda(df: pd.DataFrame, out_path: Path) -> None:
    sub = _baseline_df(df)
    agg = sub.groupby(["lambda", "policy"])["regret"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    for policy in BASELINES:
        psub = agg[agg["policy"] == policy].sort_values("lambda")
        plt.plot(psub["lambda"], psub["regret"], marker="o", label=policy, color=COLORS[policy])
    plt.xlabel("λ (cost weight)")
    plt.ylabel("Mean regret")
    plt.title("Oracle Regret vs Cost Sensitivity λ")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_regret_by_alpha(df: pd.DataFrame, out_path: Path) -> None:
    sub = _baseline_df(df)
    agg = sub.groupby(["dirichlet_alpha", "policy"])["regret"].mean().reset_index()

    plt.figure(figsize=(9, 5))
    for policy in BASELINES:
        psub = agg[agg["policy"] == policy].sort_values("dirichlet_alpha")
        plt.plot(psub["dirichlet_alpha"], psub["regret"], marker="o", label=policy, color=COLORS[policy])
    plt.xlabel("Dirichlet α")
    plt.ylabel("Mean regret")
    plt.title("Oracle Regret vs Label Heterogeneity α")
    plt.xscale("log")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _oracle_vs_policy_plot(df: pd.DataFrame, policy: str, out_path: Path, title: str) -> None:
    sub = df[df["policy"].isin(["ORACLE", policy])].copy()
    agg = (
        sub.groupby(["dataset", "lambda", "dirichlet_alpha", "seed", "policy"])["mean_utility"]
        .first()
        .unstack("policy")
        .reset_index()
    )
    if policy not in agg.columns or "ORACLE" not in agg.columns:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(agg[policy], agg["ORACLE"], alpha=0.7, s=50, edgecolors="k", linewidths=0.3)
    lims = [
        min(agg[policy].min(), agg["ORACLE"].min()) - 0.02,
        max(agg[policy].max(), agg["ORACLE"].max()) + 0.02,
    ]
    plt.plot(lims, lims, "k--", alpha=0.5, label="y = x")
    plt.xlabel(f"{policy} mean utility")
    plt.ylabel("ORACLE mean utility")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_regret_figures(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_regret_by_dataset(df, out_dir / "regret_by_dataset.png")
    plot_regret_by_lambda(df, out_dir / "regret_by_lambda.png")
    plot_regret_by_alpha(df, out_dir / "regret_by_alpha.png")
    _oracle_vs_policy_plot(df, "RANDOM", out_dir / "oracle_vs_random.png", "Oracle vs Random")
    _oracle_vs_policy_plot(df, "BEST_UNIFORM", out_dir / "oracle_vs_uniform.png", "Oracle vs Best Uniform")
    _oracle_vs_policy_plot(df, "PERSONALIZE_ALL", out_dir / "oracle_vs_p5all.png", "Oracle vs Personalize-All (P5)")
    _oracle_vs_policy_plot(df, "GLOBAL_ONLY", out_dir / "oracle_vs_p0all.png", "Oracle vs Global-Only (P0)")
