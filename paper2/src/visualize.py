"""Figure generation for Paper-2 Phase-0."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from paper2.src.config import FIGURES_DIR, LEVELS


def _setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150


def plot_accuracy_comparison(metrics_df: pd.DataFrame, out_path: Path):
    _setup_style()
    sub = metrics_df[metrics_df["split_name"] == "random"].copy()
    sub = sub[sub["model_name"] != "oracle"]
    sub = sub.sort_values("accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0" if "regression" in m or m in ("mlp", "xgboost") else "#55A868" for m in sub["model_name"]]
    ax.barh(sub["model_name"], sub["accuracy"], color=colors)
    ax.set_xlabel("Action accuracy (random split)")
    ax.set_xlim(0, 1)
    ax.set_title("Predictor accuracy comparison")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_regret_comparison(metrics_df: pd.DataFrame, out_path: Path):
    _setup_style()
    sub = metrics_df[metrics_df["split_name"] == "random"].copy()
    sub = sub.sort_values("mean_regret", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sub["model_name"], sub["mean_regret"], color="#C44E52")
    ax.set_xlabel("Mean utility regret (random split)")
    ax.set_title("Utility regret comparison (lower is better)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_test, y_pred, out_path: Path, title: str = "Best model"):
    _setup_style()
    le = LabelEncoder()
    le.fit(LEVELS)
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True oracle action")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_loo_dataset(metrics_df: pd.DataFrame, out_path: Path):
    _setup_style()
    sub = metrics_df[metrics_df["split_name"].str.startswith("loo_dataset_")].copy()
    if sub.empty:
        return
    sub["dataset"] = sub["split_name"].str.replace("loo_dataset_", "", regex=False)
    models = ["majority", "best_uniform", "random_forest", "gradient_boosting"]
    if "xgboost" in sub["model_name"].values:
        models.append("xgboost")
    pivot = sub[sub["model_name"].isin(models)].pivot_table(
        index="dataset", columns="model_name", values="mean_regret", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylabel("Mean utility regret")
    ax.set_title("Leave-one-dataset-out regret")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_action_distribution(y_test, y_pred, out_path: Path):
    _setup_style()
    le = LabelEncoder()
    le.fit(LEVELS)
    true_counts = pd.Series(y_test).value_counts().reindex(LEVELS, fill_value=0)
    pred_counts = pd.Series(y_pred).value_counts().reindex(LEVELS, fill_value=0)
    x = np.arange(len(LEVELS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, true_counts.values, w, label="Oracle", color="#4C72B0")
    ax.bar(x + w / 2, pred_counts.values, w, label="Predicted", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(LEVELS)
    ax.set_ylabel("Count")
    ax.set_title("Predicted vs oracle action distribution (random split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model, feature_names: List[str], out_path: Path, title: str = "Feature importance"):
    _setup_style()
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
    if importances is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Feature importance not available for this model", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.25)))
    ax.barh(np.array(feature_names)[idx], importances[idx], color="#55A868")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(
    metrics_df: pd.DataFrame,
    best_result: Optional[Dict],
    out_dir: Optional[Path] = None,
    fi_result: Optional[Dict] = None,
) -> Dict[str, Path]:
    out = out_dir or FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "accuracy": out / "predictor_accuracy_comparison.png",
        "regret": out / "predictor_regret_comparison.png",
        "loo_dataset": out / "leave_one_dataset_results.png",
        "feature_importance": out / "feature_importance.png",
    }
    plot_accuracy_comparison(metrics_df, paths["accuracy"])
    plot_regret_comparison(metrics_df, paths["regret"])
    plot_loo_dataset(metrics_df, paths["loo_dataset"])

    if best_result and "y_pred" in best_result:
        paths["confusion"] = out / "confusion_matrix_best_model.png"
        paths["action_dist"] = out / "predicted_vs_oracle_action_distribution.png"
        plot_confusion_matrix(
            best_result["y_test"],
            best_result["y_pred"],
            paths["confusion"],
            title=f"Confusion matrix — {best_result['model_name']}",
        )
        plot_action_distribution(best_result["y_test"], best_result["y_pred"], paths["action_dist"])
        fi = fi_result or best_result
        if "model" in fi and "used_cols" in fi:
            clf = fi["model"]
            inner = clf.named_steps["clf"] if hasattr(clf, "named_steps") else clf
            plot_feature_importance(
                inner,
                fi["used_cols"],
                paths["feature_importance"],
                title=f"Feature importance — {fi['model_name']}",
            )
    return paths
