"""Build geometry-enriched oracle decision dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from paper2.src.config import RESULTS_DIR


def _merge_sidecar_dir(df: pd.DataFrame, sidecar_dir: Path, suffix: str) -> pd.DataFrame:
    if not sidecar_dir.exists():
        return df
    files = sorted(sidecar_dir.glob("*.csv"))
    if not files:
        return df
    side = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    key = ["config_id", "client_id"]
    new_cols = [c for c in side.columns if c not in key]
    rename = {c: f"{c}_{suffix}" if c in df.columns else c for c in new_cols}
    side = side.rename(columns=rename)
    merged = df.merge(side, on=key, how="left")
    for c in new_cols:
        src = rename.get(c, c)
        if src in merged.columns and c in df.columns:
            merged[c] = merged[c].fillna(merged[src])
            merged = merged.drop(columns=[src])
        elif src in merged.columns:
            merged = merged.rename(columns={src: c})
    return merged


def build_geometry_dataset(
    oracle_path: Optional[Path] = None,
    geometry_dir: Optional[Path] = None,
    client_data_dir: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    oracle_path = oracle_path or RESULTS_DIR / "oracle_decision_dataset.csv"
    geometry_dir = geometry_dir or RESULTS_DIR / "geometry_probe"
    client_data_dir = client_data_dir or RESULTS_DIR / "client_data"
    out_path = out_path or RESULTS_DIR / "oracle_decision_dataset_geometry.csv"

    df = pd.read_csv(oracle_path)

    # Paper-1 cache heterogeneity -> geometry proxies
    if "heterogeneity" in df.columns:
        df["representation_distance"] = np.sqrt(df["heterogeneity"].clip(lower=0))
        if "prototype_distance" not in df.columns or df["prototype_distance"].isna().all():
            df["prototype_distance"] = df["representation_distance"]
        if "client_drift_score" not in df.columns or df["client_drift_score"].isna().all():
            df["client_drift_score"] = df["heterogeneity"]

    df = _merge_sidecar_dir(df, client_data_dir, "cd")
    df = _merge_sidecar_dir(df, geometry_dir, "geo")

    # Prefer probe heterogeneity when present
    for col in [
        "heterogeneity",
        "representation_distance",
        "prototype_distance",
        "client_drift_score",
        "gradient_norm",
        "gradient_variance",
        "cosine_to_global_update",
        "loss_on_global",
        "mean_predictive_entropy",
        "num_samples",
        "label_entropy",
        "majority_class_ratio",
        "class_count",
        "class_balance_score",
    ]:
        geo_col = f"{col}_geo"
        cd_col = f"{col}_cd"
        if geo_col in df.columns:
            df[col] = df[col].fillna(df[geo_col]) if col in df.columns else df[geo_col]
            df = df.drop(columns=[geo_col])
        if cd_col in df.columns:
            df[col] = df[col].fillna(df[cd_col]) if col in df.columns else df[cd_col]
            df = df.drop(columns=[cd_col])

    df.to_csv(out_path, index=False)
    return df
