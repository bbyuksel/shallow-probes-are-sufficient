"""Compare old (phase4/phase6) headline numbers to methodology-v2 metrics.

Old values are loaded from existing CSV/JSON files only — never hard-coded.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from paper2.src.methodology_v2 import PROTOCOL


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_old_metrics(
    tables_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Load old-protocol headline numbers from phase4 / phase6 artifacts."""
    root = _project_root()
    tables_dir = Path(tables_dir) if tables_dir else root / "paper2" / "tables"

    old: Dict[str, float] = {}

    # --- phase4_policy_stats.csv ---
    p4 = tables_dir / "phase4_policy_stats.csv"
    if p4.exists():
        df = pd.read_csv(p4)
        by = df.set_index("policy")
        oracle_u = _safe_float(by.loc["oracle", "mean_utility"]) if "oracle" in by.index else float("nan")
        for policy, key in [
            ("fedpba", "FEDPBA"),
            ("best_uniform", "Best uniform"),
        ]:
            if policy not in by.index:
                continue
            old[f"{key} mean regret"] = _safe_float(by.loc[policy, "mean_regret"])
            mu = _safe_float(by.loc[policy, "mean_utility"])
            if np.isfinite(oracle_u) and oracle_u != 0:
                old[f"{key} recovery"] = mu / oracle_u * 100.0

    # --- phase6_probe_sufficiency_metrics.json ---
    p6 = tables_dir / "phase6_probe_sufficiency_metrics.json"
    if p6.exists():
        with open(p6, encoding="utf-8") as f:
            m = json.load(f)
        regret = m.get("mean_regret", {})
        recovery = m.get("oracle_recovery", {})
        if "greedy_probe" in regret:
            old["Greedy mean regret"] = _safe_float(regret["greedy_probe"])
        if "fedpba" in regret and "FEDPBA mean regret" not in old:
            old["FEDPBA mean regret"] = _safe_float(regret["fedpba"])
        if "uniform_depth" in regret and "Best uniform mean regret" not in old:
            old["Best uniform mean regret"] = _safe_float(regret["uniform_depth"])
        if "context_only_best" in regret:
            old["Context-only mean regret"] = _safe_float(regret["context_only_best"])
        if "greedy_probe" in recovery:
            old["Greedy recovery"] = _safe_float(recovery["greedy_probe"])
        if "fedpba" in recovery and "FEDPBA recovery" not in old:
            old["FEDPBA recovery"] = _safe_float(recovery["fedpba"])

    # --- phase4 paired comparisons (n=2700 row-level) ---
    paired = tables_dir / "phase4_paired_comparisons.csv"
    if paired.exists():
        pdf = pd.read_csv(paired)
        for _, r in pdf.iterrows():
            name = str(r["comparison"])
            old[f"old_paired_{name}_p"] = _safe_float(r.get("wilcoxon_p"))
            old[f"old_paired_{name}_dz"] = _safe_float(r.get("cohens_dz"))

    # --- lambda / dataset from phase4 ---
    lam = tables_dir / "phase4_lambda_stats.csv"
    if lam.exists():
        ldf = pd.read_csv(lam)
        fed = ldf[ldf["policy"] == "fedpba"] if "policy" in ldf.columns else ldf
        for _, r in fed.iterrows():
            old[f"λ={r['lambda']} regret"] = _safe_float(r["mean_regret"])

    ds = tables_dir / "phase4_dataset_stats.csv"
    if ds.exists():
        ddf = pd.read_csv(ds)
        fed = ddf[ddf["policy"] == "fedpba"] if "policy" in ddf.columns else ddf
        for _, r in fed.iterrows():
            if str(r.get("dataset")) == "ptbxl":
                old["PTB-XL mean regret"] = _safe_float(r["mean_regret"])

    # --- failure counts from phase6 enriched if present ---
    enriched = tables_dir / "phase6_enriched_per_client.csv"
    if enriched.exists():
        edf = pd.read_csv(enriched)
        deep = {"P3", "P4", "P5"}
        if "oracle_deep" in edf.columns:
            deep_mask = edf["oracle_deep"].astype(bool)
        else:
            deep_mask = edf["oracle_action"].isin(deep)
        old["Deep-oracle count"] = float(deep_mask.sum())
        if "greedy_probe_action" in edf.columns:
            old["Greedy P0→deep failure count"] = float(
                ((edf["greedy_probe_action"] == "P0") & deep_mask).sum()
            )
        if "fedpba_action" in edf.columns:
            old["FEDPBA P0→deep failure count"] = float(
                ((edf["fedpba_action"] == "P0") & deep_mask).sum()
            )
        # Probe–oracle correlation on shallow max utility vs oracle when columns exist
        if {"u_max_shallow", "oracle_utility"}.issubset(edf.columns):
            old["Probe–oracle correlation"] = float(
                edf["u_max_shallow"].corr(edf["oracle_utility"])
            )

    return old


def extract_new_metrics(
    group_kfold_metrics: Dict[str, Dict[str, float]],
    cluster_tables: Dict[str, pd.DataFrame],
    oof: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Pull new-protocol numbers from grouped CV + cluster stats outputs."""
    new: Dict[str, float] = {}

    def pol(name: str) -> Dict[str, float]:
        return group_kfold_metrics.get(name, {})

    mapping = [
        ("greedy", "Greedy"),
        ("fedpba", "FEDPBA"),
        ("best_uniform", "Best uniform"),
        ("context_only", "Context-only"),
    ]
    for key, label in mapping:
        m = pol(key)
        if "mean_regret" in m:
            new[f"{label} mean regret"] = _safe_float(m["mean_regret"])
        if "oracle_utility_recovery_pct" in m:
            new[f"{label} recovery"] = _safe_float(m["oracle_utility_recovery_pct"])

    # Session-level primary tests
    holm = cluster_tables.get("holm_corrected_primary_tests")
    if holm is not None and len(holm):
        for _, r in holm.iterrows():
            cmp = str(r["comparison"])
            new[f"primary p {cmp}"] = _safe_float(r.get("wilcoxon_p"))
            new[f"primary Holm p {cmp}"] = _safe_float(r.get("holm_adjusted_p"))
            new[f"primary dz {cmp}"] = _safe_float(r.get("cohens_dz"))

    # Lambda / dataset from stratified tables
    lam = cluster_tables.get("lambda_stratified_results")
    if lam is not None and len(lam):
        fed = lam[lam["policy"] == "fedpba"]
        for _, r in fed.iterrows():
            new[f"λ={r['lambda']} regret"] = _safe_float(r["mean_regret"])

    ds = cluster_tables.get("dataset_stratified_results")
    if ds is not None and len(ds):
        fed = ds[(ds["policy"] == "fedpba") & (ds["dataset"].astype(str) == "ptbxl")]
        if len(fed):
            new["PTB-XL mean regret"] = _safe_float(fed.iloc[0]["mean_regret"])

    if oof is not None and len(oof):
        deep = {"P3", "P4", "P5"}
        deep_mask = oof["oracle_action"].isin(deep)
        new["Deep-oracle count"] = float(deep_mask.sum())
        g_col = "greedy_action" if "greedy_action" in oof.columns else "greedy_probe_action"
        if g_col in oof.columns:
            new["Greedy P0→deep failure count"] = float(
                ((oof[g_col] == "P0") & deep_mask).sum()
            )
        if "fedpba_action" in oof.columns:
            new["FEDPBA P0→deep failure count"] = float(
                ((oof["fedpba_action"] == "P0") & deep_mask).sum()
            )
        # Probe–oracle: max shallow utility vs oracle utility
        shallow_cols = [c for c in ("utility_p0", "utility_p1", "utility_p2") if c in oof.columns]
        if shallow_cols and "oracle_utility" in oof.columns:
            u_max = oof[shallow_cols].max(axis=1)
            new["Probe–oracle correlation"] = float(u_max.corr(oof["oracle_utility"]))

    return new


def write_revision_comparison(
    old: Dict[str, float],
    new: Dict[str, float],
    out_path: Path | str,
    reason_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Write methodology_revision_comparison.csv."""
    reason_map = reason_map or {}
    default_reason = (
        "Grouped session CV + session-cluster inference (sinconf2026_methodology_v2); "
        "not comparable as iid n=2700"
    )
    keys = sorted(set(old) | set(new))
    rows: List[Dict] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for metric in keys:
        o = old.get(metric, float("nan"))
        n = new.get(metric, float("nan"))
        abs_chg = n - o if np.isfinite(o) and np.isfinite(n) else float("nan")
        if np.isfinite(o) and o != 0 and np.isfinite(n):
            rel = abs_chg / abs(o)
        else:
            rel = float("nan")
        rows.append(
            {
                "metric": metric,
                "old_protocol": o,
                "new_protocol": n,
                "absolute_change": abs_chg,
                "relative_change": rel,
                "reason_for_change": reason_map.get(metric, default_reason),
                "protocol_version": PROTOCOL,
                "timestamp_utc": ts,
            }
        )
    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def run_comparison(
    group_kfold_metrics: Dict[str, Dict[str, float]],
    cluster_tables: Dict[str, pd.DataFrame],
    out_dir: Path | str,
    oof: Optional[pd.DataFrame] = None,
    tables_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load old artifacts, extract new metrics, write comparison CSV."""
    out_dir = Path(out_dir)
    old = load_old_metrics(tables_dir=tables_dir)
    new = extract_new_metrics(group_kfold_metrics, cluster_tables, oof=oof)
    return write_revision_comparison(
        old, new, out_dir / "methodology_revision_comparison.csv"
    )
