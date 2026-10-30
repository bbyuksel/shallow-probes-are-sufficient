"""Build oracle decision datasets from Paper-1 client-level caches."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from fedpbg.hardware.profiles import HARDWARE_PROFILES, HardwareType
from fedpbg.metrics.utility import compute_utility, oracle_level_for_client

from paper2.src.config import (
    CACHE_DIRS,
    CHEAP_FEATURE_COLS,
    EXPENSIVE_FEATURE_COLS,
    LAMBDA_VALUES,
    LEVELS,
    RESULTS_DIR,
)

CACHE_PATTERN = re.compile(
    r"^(?P<dataset>\w+)_s(?P<seed>\d+)_a(?P<alpha>[\d.]+)_n(?P<clients>\d+)_client_level\.csv$"
)

# Client-data and geometry columns not yet in Paper-1 caches
PLACEHOLDER_COLS = [
    "num_samples",
    "label_entropy",
    "majority_class_ratio",
    "class_count",
    "class_balance_score",
    "gradient_norm",
    "gradient_variance",
    "cosine_to_global_update",
    "representation_distance",
    "prototype_distance",
]


def discover_cache_files(cache_dirs: Optional[List[Path]] = None) -> List[Path]:
    dirs = cache_dirs or CACHE_DIRS
    files: List[Path] = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*_client_level.csv")))
    seen: set[str] = set()
    unique: List[Path] = []
    for f in files:
        if f.name not in seen:
            seen.add(f.name)
            unique.append(f)
    return unique


def _hardware_score(hw: str) -> float:
    try:
        profile = HARDWARE_PROFILES[HardwareType(hw)]
    except (KeyError, ValueError):
        return np.nan
    weights = {"compute": 0.35, "memory": 0.20, "bandwidth": 0.20, "latency": 0.15, "energy": 0.10}
    return float(
        weights["compute"] * profile.compute
        + weights["memory"] * profile.memory
        + weights["bandwidth"] * profile.bandwidth
        + weights["latency"] * profile.latency
        + weights["energy"] * profile.energy
    )


def parse_cache_meta(path: Path, client_df: pd.DataFrame) -> Dict:
    m = CACHE_PATTERN.match(path.name)
    if m:
        return {
            "dataset": client_df["dataset"].iloc[0] if "dataset" in client_df.columns else m.group("dataset"),
            "seed": int(m.group("seed")),
            "alpha": float(m.group("alpha")),
            "num_clients": int(m.group("clients")),
        }
    return {
        "dataset": client_df["dataset"].iloc[0],
        "seed": int(client_df["seed"].iloc[0]),
        "alpha": float(client_df["dirichlet_alpha"].iloc[0]),
        "num_clients": int(client_df["num_clients"].iloc[0]),
    }


def _pivot_client_levels(group: pd.DataFrame) -> Dict:
    """Wide-format metrics for one client across P0–P5."""
    out: Dict = {}
    for lv in LEVELS:
        row = group[group["level"] == lv]
        if row.empty:
            out[f"macro_f1_{lv.lower()}"] = np.nan
            out[f"cost_{lv.lower()}"] = np.nan
            out[f"utility_{lv.lower()}"] = np.nan
            continue
        r = row.iloc[0]
        out[f"macro_f1_{lv.lower()}"] = float(r["macro_f1"])
        out[f"cost_{lv.lower()}"] = float(r["normalized_cost"])
    return out


def build_rows_from_cache(path: Path, lambda_values: List[float]) -> List[Dict]:
    client_df = pd.read_csv(path)
    meta = parse_cache_meta(path, client_df)
    config_id = path.stem.replace("_client_level", "")
    rows: List[Dict] = []

    for cid, group in client_df.groupby("client_id"):
        base = group.iloc[0]
        wide = _pivot_client_levels(group)
        heterogeneity = float(base["heterogeneity"])
        hw = str(base["hardware"])
        hw_mult = float(base["hardware_multiplier"])

        for lam in lambda_values:
            oracle_level, oracle_utility = oracle_level_for_client(group, lam)
            utilities = {
                lv: compute_utility(wide[f"macro_f1_{lv.lower()}"], wide[f"cost_{lv.lower()}"], lam)
                for lv in LEVELS
                if not np.isnan(wide.get(f"macro_f1_{lv.lower()}", np.nan))
            }
            row = {
                "config_id": config_id,
                "dataset": meta["dataset"],
                "alpha": meta["alpha"],
                "lambda": lam,
                "seed": meta["seed"],
                "num_clients": meta["num_clients"],
                "client_id": int(cid),
                "hardware_profile": hw,
                "hardware_score": _hardware_score(hw),
                "hardware_multiplier": hw_mult,
                "heterogeneity": heterogeneity,
                "client_drift_score": heterogeneity,
                "oracle_action": oracle_level,
                "oracle_utility": oracle_utility,
            }
            for lv in LEVELS:
                key = lv.lower()
                row[f"utility_{key}"] = utilities.get(lv, np.nan)
                row[f"cost_{key}"] = wide.get(f"cost_{key}", np.nan)
                row[f"{key}_macro_f1"] = wide.get(f"macro_f1_{key}", np.nan)
            for col in PLACEHOLDER_COLS:
                row[col] = np.nan
            rows.append(row)
    return rows


def build_oracle_dataset(
    lambda_values: Optional[List[float]] = None,
    cache_dirs: Optional[List[Path]] = None,
) -> pd.DataFrame:
    lambdas = lambda_values or LAMBDA_VALUES
    cache_files = discover_cache_files(cache_dirs)
    if not cache_files:
        raise FileNotFoundError("No Paper-1 client_level caches found.")

    all_rows: List[Dict] = []
    for path in cache_files:
        all_rows.extend(build_rows_from_cache(path, lambdas))
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["dataset", "alpha", "seed", "num_clients", "client_id", "lambda"]).reset_index(drop=True)
    return df


def _rename_for_spec(df: pd.DataFrame) -> pd.DataFrame:
    """Align column names with Phase-0 spec (p0_macro_f1, cost_p0, etc.)."""
    rename = {}
    for lv in LEVELS:
        key = lv.lower()
        if f"macro_f1_{key}" in df.columns:
            rename[f"macro_f1_{key}"] = f"{key}_macro_f1"
    out = df.rename(columns=rename)
    return out


def make_full_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """All features including P0–P5 utilities (analysis only)."""
    cols = [
        "config_id",
        "dataset",
        "alpha",
        "lambda",
        "seed",
        "num_clients",
        "client_id",
        "oracle_action",
        "oracle_utility",
    ]
    for lv in LEVELS:
        key = lv.lower()
        cols.extend([f"utility_{key}", f"{key}_macro_f1", f"cost_{key}"])
    cols.extend(
        [
            "hardware_profile",
            "hardware_score",
            "hardware_multiplier",
            "heterogeneity",
            "client_drift_score",
        ]
        + PLACEHOLDER_COLS
    )
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()


def make_cheap_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Deployable features only; target = oracle_action."""
    feature_cols = [c for c in CHEAP_FEATURE_COLS if c in df.columns]
    exclude = set(EXPENSIVE_FEATURE_COLS)
    feature_cols = [c for c in feature_cols if c not in exclude]
    meta = ["config_id", "client_id", "seed", "oracle_action", "oracle_utility"]
    for lv in LEVELS:
        meta.append(f"utility_{lv.lower()}")
    cols = meta + feature_cols
    cols = list(dict.fromkeys(c for c in cols if c in df.columns))
    return df[cols].copy()


def save_datasets(df: pd.DataFrame, out_dir: Optional[Path] = None) -> Dict[str, Path]:
    out = out_dir or RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    df = _rename_for_spec(df)
    full = make_full_dataset(df)
    cheap = make_cheap_dataset(df)
    paths = {
        "main": out / "oracle_decision_dataset.csv",
        "full": out / "oracle_decision_dataset_full.csv",
        "cheap": out / "oracle_decision_dataset_cheap.csv",
    }
    df.to_csv(paths["main"], index=False)
    full.to_csv(paths["full"], index=False)
    cheap.to_csv(paths["cheap"], index=False)
    return paths
