"""Revision-v3 figures: Fig3 scatter, fixed Fig9, policy regret CI."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEEP = {"P3", "P4", "P5"}

POLICY_LABELS = {
    "greedy": "Greedy probe",
    "fedpba": "FedPBA",
    "best_uniform": "Best uniform",
    "context_only": "Context-only",
    "majority": "Majority",
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig3_shallow_vs_oracle_scatter(
    enriched: pd.DataFrame,
    out_path: Path,
    overall_r: float,
    deep_r: float,
) -> Path:
    """
    Scatter shallow_best vs oracle utility.
    Shallow-oracle: light; deep-oracle: bold x; y=x reference.
    """
    df = enriched.copy()
    if "oracle_deep" not in df.columns:
        df["oracle_deep"] = df["oracle_action"].astype(str).isin(DEEP)

    shallow = df[~df["oracle_deep"]]
    deep = df[df["oracle_deep"]]

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    if len(shallow):
        ax.scatter(
            shallow["shallow_best_utility"],
            shallow["oracle_utility"],
            s=10,
            c="#9e9e9e",
            alpha=0.25,
            label=f"Shallow oracle (n={len(shallow)})",
            rasterized=True,
        )
    if len(deep):
        ax.scatter(
            deep["shallow_best_utility"],
            deep["oracle_utility"],
            s=36,
            c="#c44e52",
            marker="x",
            linewidths=1.2,
            alpha=0.9,
            label=f"Deep oracle (n={len(deep)})",
        )

    vals = np.concatenate(
        [
            df["shallow_best_utility"].to_numpy(float),
            df["oracle_utility"].to_numpy(float),
        ]
    )
    vals = vals[np.isfinite(vals)]
    lo = float(np.min(vals)) if len(vals) else 0.0
    hi = float(np.max(vals)) if len(vals) else 1.0
    pad = 0.02 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0, alpha=0.7, label="y = x")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(r"Shallow-best utility $\max(U_{P0},U_{P1},U_{P2})$")
    ax.set_ylabel("Oracle utility")
    ax.set_title(
        f"Shallow-best vs oracle\n"
        f"overall Pearson r={overall_r:.4f} (n={len(df)}); "
        f"deep-only r={deep_r:.4f} (n={len(deep)})"
    )
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    return _save(fig, out_path)


def fig9_greedy_p0_given_deep(
    oof_or_enriched: pd.DataFrame,
    out_path: Path,
    audit_csv_path: Path,
) -> Path:
    """
    Fixed Fig9: rate = count(greedy==P0 AND oracle deep) / count(oracle deep)
    per dataset × lambda. Denom=0 → N/A (not 0).
    """
    df = oof_or_enriched.copy()
    if "greedy_action" in df.columns:
        greedy = df["greedy_action"].astype(str)
    elif "greedy_probe_action" in df.columns:
        greedy = df["greedy_probe_action"].astype(str)
    else:
        raise KeyError("Need greedy_action or greedy_probe_action")

    oracle_deep = df["oracle_action"].astype(str).isin(DEEP)
    gp_p0_and_deep = (greedy == "P0") & oracle_deep

    datasets = ["cifar10", "cifar100", "ptbxl"]
    lambdas = sorted(df["lambda"].astype(float).unique())
    name_ds = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "ptbxl": "PTB-XL"}

    audit_rows: List[Dict] = []
    rate_mat = np.full((len(datasets), len(lambdas)), np.nan)
    annot = [["" for _ in lambdas] for _ in datasets]

    for i, ds in enumerate(datasets):
        for j, lam in enumerate(lambdas):
            mask = (df["dataset"].astype(str) == ds) & (df["lambda"].astype(float) == float(lam))
            denom = int(oracle_deep[mask].sum())
            numer = int(gp_p0_and_deep[mask].sum())
            if denom == 0:
                rate = float("nan")
                cell = "N/A\n(0 deep)"
                display_rate = None
            else:
                rate = numer / denom
                rate_mat[i, j] = rate
                cell = f"{rate:.3f}\n({numer}/{denom})"
                display_rate = rate
            annot[i][j] = cell
            audit_rows.append(
                {
                    "dataset": ds,
                    "lambda": float(lam),
                    "numerator_greedy_p0_and_oracle_deep": numer,
                    "denominator_oracle_deep": denom,
                    "rate": display_rate if display_rate is not None else np.nan,
                    "rate_display": "N/A" if denom == 0 else f"{rate:.6f}",
                    "definition": "P(greedy=P0 | oracle deep, dataset, lambda)",
                }
            )

    audit = pd.DataFrame(audit_rows)
    audit_csv_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    vmax = float(np.nanmax(rate_mat)) if np.isfinite(rate_mat).any() else 1.0
    vmax = max(vmax, 0.05)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(np.ma.masked_invalid(rate_mat), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([rf"$\lambda$={lam:g}" for lam in lambdas])
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([name_ds[d] for d in datasets])
    ax.set_xlabel(r"Cost sensitivity $\lambda$")
    ax.set_ylabel("Dataset")
    ax.set_title(r"$P(\mathrm{greedy}=P_0 \mid \mathrm{oracle\ deep})$")
    for i in range(len(datasets)):
        for j in range(len(lambdas)):
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Conditional rate")
    return _save(fig, out_path)


def fig_policy_regret_ci(
    boot: pd.DataFrame,
    out_path: Path,
    policies: Optional[Sequence[str]] = None,
) -> Path:
    """Mean regret with session-cluster bootstrap 95% CI."""
    order = list(policies) if policies is not None else [
        "greedy",
        "fedpba",
        "best_uniform",
        "context_only",
    ]
    sub = boot[boot["policy"].isin(order)].copy()
    sub["policy"] = pd.Categorical(sub["policy"], categories=order, ordered=True)
    sub = sub.dropna(subset=["mean_regret"]).sort_values("policy")
    if sub.empty:
        raise ValueError("No policies available for regret CI figure")

    names = [POLICY_LABELS.get(p, p) for p in sub["policy"]]
    y = sub["mean_regret"].to_numpy(float)
    lo = sub["regret_ci_lo"].to_numpy(float)
    hi = sub["regret_ci_hi"].to_numpy(float)
    err = np.vstack([y - lo, hi - y])

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(names))
    ax.bar(
        x,
        y,
        yerr=err,
        capsize=3,
        color="#4c78a8",
        edgecolor="black",
        linewidth=0.5,
        ecolor="black",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Mean regret vs. Oracle")
    ax.set_title("Session-cluster bootstrap 95% CI (methodology v2)")
    ax.axhline(0, color="gray", lw=0.8)
    return _save(fig, out_path)


def _copy_figures(paths: Dict[str, Path], dest_dirs: Sequence[Path]) -> None:
    for dest in dest_dirs:
        dest.mkdir(parents=True, exist_ok=True)
        for p in paths.values():
            if p is not None and Path(p).is_file():
                shutil.copy2(p, dest / Path(p).name)


def generate_figures_v3(
    out_dir: Path | str,
    enriched: pd.DataFrame,
    oof: pd.DataFrame,
    cluster_bootstrap: pd.DataFrame,
    overall_r: float,
    deep_r: float,
    sinconf_root: Path | str,
) -> Dict[str, Path]:
    """
    Write figures under ``out_dir/figures`` and copy to submission/figures and v3/figures.
    """
    out_dir = Path(out_dir)
    sinconf_root = Path(sinconf_root)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    paths["fig3"] = fig3_shallow_vs_oracle_scatter(
        enriched,
        fig_dir / "phase6_probe_vs_oracle_utility_scatter.png",
        overall_r=overall_r,
        deep_r=deep_r,
    )
    alias3 = fig_dir / "fig3_shallow_vs_oracle_scatter_v3.png"
    shutil.copy2(paths["fig3"], alias3)
    paths["fig3_v3"] = alias3

    paths["fig9"] = fig9_greedy_p0_given_deep(
        oof,
        fig_dir / "phase6_deep_action_failure_heatmap.png",
        audit_csv_path=out_dir / "figure9_denominator_audit.csv",
    )
    alias9 = fig_dir / "fig9_greedy_p0_given_deep_v3.png"
    shutil.copy2(paths["fig9"], alias9)
    paths["fig9_v3"] = alias9

    paths["regret_ci"] = fig_policy_regret_ci(
        cluster_bootstrap,
        fig_dir / "phase4_regret_ci_by_policy.png",
    )
    alias_r = fig_dir / "fig_policy_regret_ci_v3.png"
    shutil.copy2(paths["regret_ci"], alias_r)
    paths["regret_ci_v3"] = alias_r

    _copy_figures(
        paths,
        [
            sinconf_root / "submission" / "figures",
            sinconf_root / "v3" / "figures",
        ],
    )
    return paths
