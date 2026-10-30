"""Probe sufficiency audit using Phase-4/5 outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from fedpbg.metrics.utility import compute_utility
from paper2.src.config import FIGURES_DIR, REPORTS_DIR, RESULTS_DIR, TABLES_DIR
from paper2.src.online.policies import utility_at_level
from paper2.src.online.session import PROBE_LEVELS, load_session

ONLINE_DIR = RESULTS_DIR / "online"
SESSION_DIR = ONLINE_DIR / "sessions"
PHASE5_DETAIL = ONLINE_DIR / "phase5_contextual_per_client.csv"
PHASE4_DETAIL = ONLINE_DIR / "phase4_online_per_client.csv"
PHASE5_STATS = TABLES_DIR / "phase5_contextual_policy_stats.csv"
PHASE5_SUMMARY = TABLES_DIR / "phase5_contextual_policy_summary.csv"

CONTEXT_ONLY_POLICIES = [
    "multinomial_logistic",
    "cost_sensitive_logistic",
    "contextual_tree",
    "contextual_random_forest",
]

POLICY_CLASSES = {
    "A_fixed": ["majority", "uniform_depth"],
    "B_context_only": CONTEXT_ONLY_POLICIES,
    "C_probe_nonlearned": ["greedy_probe"],
    "D_probe_learned": ["fedpba"],
}

DEEP_LEVELS = {"P3", "P4", "P5"}
SHALLOW_LEVELS = {"P0", "P1", "P2"}


def load_merged_detail() -> pd.DataFrame:
    p5 = pd.read_csv(PHASE5_DETAIL)
    p4 = pd.read_csv(PHASE4_DETAIL)[
        [
            "config_id",
            "lambda",
            "client_id",
            "global_only_utility",
            "p5_all_utility",
        ]
    ]
    return p5.merge(p4, on=["config_id", "lambda", "client_id"], how="left")


def load_policy_stats() -> pd.DataFrame:
    stats = pd.read_csv(PHASE5_STATS).set_index("policy")
    p4 = pd.read_csv(Path(TABLES_DIR / "phase4_policy_stats.csv")).set_index("policy")
    for pol in ["global_only", "personalize_all"]:
        if pol in p4.index:
            row = p4.loc[pol]
            stats.loc[pol] = {
                "mean_utility": row["mean_utility"],
                "mean_regret": row["mean_regret"],
                "oracle_utility_recovery_pct": row["mean_utility"] / 0.44071415029297845 * 100,
                "action_accuracy": np.nan,
                "n": row["n"],
            }
    return stats.reset_index()


def best_context_only_policy(stats: pd.DataFrame) -> str:
    sub = stats[stats["policy"].isin(CONTEXT_ONLY_POLICIES)]
    return str(sub.sort_values("mean_regret").iloc[0]["policy"])


def compute_sufficiency_metrics(stats: pd.DataFrame) -> Dict:
    s = stats.set_index("policy")
    bu_r = float(s.loc["uniform_depth", "mean_regret"])
    ctx_pol = best_context_only_policy(stats)
    ctx_r = float(s.loc[ctx_pol, "mean_regret"])
    gp_r = float(s.loc["greedy_probe", "mean_regret"])
    fed_r = float(s.loc["fedpba", "mean_regret"])

    def suff(r: float) -> float:
        return 1.0 - r / bu_r if bu_r > 0 else np.nan

    return {
        "context_only_best_policy": ctx_pol,
        "oracle_recovery": {
            "greedy_probe": float(s.loc["greedy_probe", "oracle_utility_recovery_pct"]),
            "fedpba": float(s.loc["fedpba", "oracle_utility_recovery_pct"]),
            "uniform_depth": float(s.loc["uniform_depth", "oracle_utility_recovery_pct"]),
            "context_only_best": float(s.loc[ctx_pol, "oracle_utility_recovery_pct"]),
            "global_only": float(s.loc["global_only", "oracle_utility_recovery_pct"]),
            "personalize_all": float(s.loc["personalize_all", "oracle_utility_recovery_pct"]),
        },
        "regret_reduction_vs_uniform": {
            "greedy_probe": bu_r - gp_r,
            "fedpba": bu_r - fed_r,
            "context_only_best": bu_r - ctx_r,
        },
        "regret_reduction_vs_context_only": {
            "greedy_probe": ctx_r - gp_r,
            "fedpba": ctx_r - fed_r,
            "uniform_depth": ctx_r - bu_r,
        },
        "probe_information_gain": {
            "greedy_probe": ctx_r - gp_r,
            "fedpba": ctx_r - fed_r,
        },
        "probe_sufficiency_ratio": {
            "greedy_probe": suff(gp_r),
            "fedpba": suff(fed_r),
            "context_only_best": suff(ctx_r),
        },
        "mean_regret": {
            "greedy_probe": gp_r,
            "fedpba": fed_r,
            "uniform_depth": bu_r,
            "context_only_best": ctx_r,
            "global_only": float(s.loc["global_only", "mean_regret"]),
            "personalize_all": float(s.loc["personalize_all", "mean_regret"]),
        },
    }


def enrich_with_probe_utilities(detail: pd.DataFrame) -> pd.DataFrame:
    """Add P0/P1/P2 utilities and max shallow utility per client from session caches."""
    rows = []
    cache: Dict[str, pd.DataFrame] = {}

    for _, r in detail.iterrows():
        key = r["config_id"]
        if key not in cache:
            path = SESSION_DIR / f"{key}_client_level.csv"
            cache[key] = load_session(path).client_level_df

        df = cache[key]
        grp = df[df["client_id"] == r["client_id"]]
        lam = float(r["lambda"])
        probe_utils = {}
        for lv in PROBE_LEVELS:
            probe_utils[f"u_{lv.lower()}"] = utility_at_level(grp, lv, lam)
        row = r.to_dict()
        row.update(probe_utils)
        row["u_max_shallow"] = max(probe_utils.values())
        row["oracle_shallow"] = r["oracle_action"] in SHALLOW_LEVELS
        row["oracle_deep"] = r["oracle_action"] in DEEP_LEVELS
        rows.append(row)

    return pd.DataFrame(rows)


def mechanism_analysis(enriched: pd.DataFrame) -> Dict:
    oracle_u = enriched["oracle_utility"].astype(float)
    u0 = enriched["u_p0"].astype(float)
    u1 = enriched["u_p1"].astype(float)
    u2 = enriched["u_p2"].astype(float)
    umax = enriched["u_max_shallow"].astype(float)
    gp_u = enriched["greedy_probe_utility"].astype(float)

    def corr(a, b):
        m = ~(a.isna() | b.isna())
        if m.sum() < 2:
            return np.nan
        return float(np.corrcoef(a[m], b[m])[0, 1])

    oracle_shallow_pct = float(enriched["oracle_shallow"].mean() * 100)
    deep_mask = enriched["oracle_deep"]
    gp_low_regret_deep = float(
        ((oracle_u[deep_mask] - gp_u[deep_mask]) <= 0.05).mean() * 100
    ) if deep_mask.sum() else np.nan

    deep_oracle = enriched[deep_mask]
    gp_fail_deep = deep_oracle[
        (deep_oracle["oracle_utility"] - deep_oracle["greedy_probe_utility"]) > 0.05
    ]

    return {
        "corr_oracle_u_p0": corr(oracle_u, u0),
        "corr_oracle_u_p1": corr(oracle_u, u1),
        "corr_oracle_u_p2": corr(oracle_u, u2),
        "corr_oracle_u_max_shallow": corr(oracle_u, umax),
        "corr_oracle_greedy_probe_utility": corr(oracle_u, gp_u),
        "oracle_action_shallow_pct": oracle_shallow_pct,
        "oracle_action_deep_pct": 100.0 - oracle_shallow_pct,
        "oracle_deep_count": int(deep_mask.sum()),
        "greedy_low_regret_when_oracle_deep_pct": gp_low_regret_deep,
        "greedy_high_regret_deep_cases": int(len(gp_fail_deep)),
        "top_missed_deep_oracle_actions": (
            gp_fail_deep["oracle_action"].value_counts().head(5).to_dict()
            if len(gp_fail_deep)
            else {}
        ),
    }


def failure_mode_analysis(enriched: pd.DataFrame) -> Dict:
    df = enriched.copy()
    df["gp_regret"] = df["oracle_utility"] - df["greedy_probe_utility"]
    df["fed_regret"] = df["oracle_utility"] - df["fedpba_utility"]
    df["gp_pred_p0_oracle_deep"] = (df["greedy_probe_action"] == "P0") & df["oracle_deep"]
    df["fed_pred_p0_oracle_deep"] = (df["fedpba_action"] == "P0") & df["oracle_deep"]

    def summarize(mask: pd.Series, regret_col: str) -> Dict:
        sub = df[mask]
        return {
            "count": int(len(sub)),
            "mean_regret": float(sub[regret_col].mean()) if len(sub) else np.nan,
            "high_regret_count": int((sub[regret_col] > 0.05).sum()) if len(sub) else 0,
        }

    by_dataset = {}
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        by_dataset[ds] = {
            "greedy_probe_regret": float(sub["gp_regret"].mean()),
            "fedpba_regret": float(sub["fed_regret"].mean()),
            "gp_p0_oracle_deep": int(sub["gp_pred_p0_oracle_deep"].sum()),
            "fed_p0_oracle_deep": int(sub["fed_pred_p0_oracle_deep"].sum()),
        }

    by_lambda = {}
    for lam in sorted(df["lambda"].unique()):
        sub = df[df["lambda"] == lam]
        by_lambda[float(lam)] = {
            "greedy_probe_regret": float(sub["gp_regret"].mean()),
            "fedpba_regret": float(sub["fed_regret"].mean()),
            "gp_p0_oracle_deep": int(sub["gp_pred_p0_oracle_deep"].sum()),
            "fed_p0_oracle_deep": int(sub["fed_pred_p0_oracle_deep"].sum()),
        }

    confusion_gp = (
        df[df["oracle_deep"]]
        .groupby(["greedy_probe_action", "oracle_action"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    return {
        "overall_gp_p0_oracle_deep": summarize(df["gp_pred_p0_oracle_deep"], "gp_regret"),
        "overall_fed_p0_oracle_deep": summarize(df["fed_pred_p0_oracle_deep"], "fed_regret"),
        "by_dataset": by_dataset,
        "by_lambda": by_lambda,
        "deep_oracle_confusion_gp": confusion_gp,
        "ptbxl_gp_p0_oracle_deep": int(df[df["dataset"].str.contains("ptbxl", case=False)]["gp_pred_p0_oracle_deep"].sum()),
        "ptbxl_fed_p0_oracle_deep": int(df[df["dataset"].str.contains("ptbxl", case=False)]["fed_pred_p0_oracle_deep"].sum()),
    }


def policy_class_runtime(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate probe cost and timing by policy."""
    agg = (
        summary.groupby("policy")
        .agg(
            mean_probe_trainings=("probe_trainings", "mean"),
            mean_probe_cost_ratio=("probe_cost_ratio", "mean"),
            mean_probe_sec=("probe_sec", "mean"),
            mean_oracle_sec=("oracle_levels_sec", "mean"),
        )
        .reset_index()
    )
    return agg


def generate_figures(
    metrics: Dict,
    stats: pd.DataFrame,
    enriched: pd.DataFrame,
    failure: Dict,
) -> Dict[str, Path]:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. Sufficiency ratio
    fig, ax = plt.subplots(figsize=(7, 4))
    pols = ["greedy_probe", "fedpba", "context_only_best"]
    labels = ["Greedy probe", "FedPBA", f"Context-only\n({metrics['context_only_best_policy']})"]
    vals = [metrics["probe_sufficiency_ratio"][p] for p in pols]
    colors = ["#dd8452", "#c44e52", "#4c72b0"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Probe sufficiency ratio\n1 - Regret / Regret(uniform)")
    ax.set_title("Phase-6: Probe sufficiency ratio")
    fig.tight_layout()
    p = FIGURES_DIR / "phase6_probe_sufficiency_ratio.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["sufficiency"] = p

    # 2. Policy class regret
    class_regret = []
    class_labels = []
    for cls, pols in POLICY_CLASSES.items():
        sub = stats[stats["policy"].isin(pols)]
        class_regret.append(float(sub["mean_regret"].mean()))
        class_labels.append(cls.replace("_", " ").title())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(class_labels, class_regret, color=["#8c8c8c", "#4c72b0", "#dd8452", "#c44e52"])
    ax.set_ylabel("Mean regret (class average)")
    ax.set_title("Mean regret by policy class")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    p = FIGURES_DIR / "phase6_policy_class_regret.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["class_regret"] = p

    # 3. Oracle recovery by class
    class_recovery = []
    for cls, pols in POLICY_CLASSES.items():
        sub = stats[stats["policy"].isin(pols)]
        class_recovery.append(float(sub["oracle_utility_recovery_pct"].mean()))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(class_labels, class_recovery, color=["#8c8c8c", "#4c72b0", "#dd8452", "#c44e52"])
    ax.axhline(100, color="gray", linestyle="--")
    ax.set_ylabel("Oracle utility recovery (%)")
    ax.set_title("Oracle recovery by policy class")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    p = FIGURES_DIR / "phase6_oracle_recovery_by_policy_class.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["class_recovery"] = p

    # 4. Scatter max shallow vs oracle utility
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sample = enriched.sample(min(800, len(enriched)), random_state=42)
    ax.scatter(sample["u_max_shallow"], sample["oracle_utility"], alpha=0.25, s=12, c="#4c72b0")
    lim = max(sample["u_max_shallow"].max(), sample["oracle_utility"].max())
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5)
    r = metrics.get("_corr_display", enriched["u_max_shallow"].corr(enriched["oracle_utility"]))
    ax.set_xlabel("max(P0,P1,P2) utility")
    ax.set_ylabel("Oracle utility")
    ax.set_title(f"Shallow probe vs. oracle utility (r={r:.3f})")
    fig.tight_layout()
    p = FIGURES_DIR / "phase6_probe_vs_oracle_utility_scatter.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["scatter"] = p

    # 5. Deep-action failure heatmap (dataset x lambda)
    df = enriched.copy()
    df["gp_p0_deep"] = (df["greedy_probe_action"] == "P0") & df["oracle_deep"]
    pivot = df.pivot_table(
        index="dataset",
        columns="lambda",
        values="gp_p0_deep",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=pivot.values.max())
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("lambda")
    ax.set_ylabel("dataset")
    ax.set_title("P(greedy→P0 | oracle deep)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p = FIGURES_DIR / "phase6_deep_action_failure_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["heatmap"] = p

    return paths
