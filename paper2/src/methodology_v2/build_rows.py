"""Build policy-learning rows from online ``*_client_level.csv`` session caches."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from paper2.src.methodology_v2.ids import add_identity_columns, make_session_id, parse_config_id

LEVELS = ["P0", "P1", "P2", "P3", "P4", "P5"]
PROBE_LEVELS = ["P0", "P1", "P2"]
DEFAULT_LAMBDAS = [0.1, 0.3, 1.0]

# FEDPBA features (matches paper2.src.feature_sets FEATURE_SETS["context_probes"])
CONTEXT_PROBES_FEATURES = [
    "dataset",
    "alpha",
    "lambda",
    "num_clients",
    "hardware_profile",
    "hardware_score",
    "hardware_multiplier",
    "p0_macro_f1",
    "p1_macro_f1",
    "p2_macro_f1",
    "cost_p0",
    "cost_p1",
    "cost_p2",
]

CONTEXT_ONLY_FEATURES = [
    "dataset",
    "alpha",
    "lambda",
    "num_clients",
    "hardware_profile",
    "hardware_score",
    "hardware_multiplier",
]


def _compute_utility(macro_f1: float, normalized_cost: float, lambda_: float) -> float:
    """U = Macro-F1 - lambda * normalized_cost (prefer fedpbg implementation)."""
    try:
        from fedpbg.metrics.utility import compute_utility

        return float(compute_utility(macro_f1, normalized_cost, lambda_))
    except ImportError:
        return float(macro_f1 - lambda_ * normalized_cost)


def _hardware_score(hw: str) -> float:
    """
    Hardware capability score matching ``paper2.src.online.features._hardware_score``.

    Uses fedpbg HARDWARE_PROFILES when available; otherwise the same numeric
    table (avoids importing torch via online.session).
    """
    try:
        from fedpbg.hardware.profiles import HARDWARE_PROFILES, HardwareType

        profile = HARDWARE_PROFILES[HardwareType(str(hw))]
        weights = {"compute": 0.35, "memory": 0.20, "bandwidth": 0.20, "latency": 0.15, "energy": 0.10}
        return float(
            weights["compute"] * profile.compute
            + weights["memory"] * profile.memory
            + weights["bandwidth"] * profile.bandwidth
            + weights["latency"] * profile.latency
            + weights["energy"] * profile.energy
        )
    except Exception:
        pass

    # Same numbers as HARDWARE_PROFILES × weights above (no heavy imports).
    table = {
        "LOW": 0.35 * 0.25 + 0.20 * 0.20 + 0.20 * 0.15 + 0.15 * 0.30 + 0.10 * 0.35,
        "MEDIUM": 0.35 * 0.55 + 0.20 * 0.50 + 0.20 * 0.50 + 0.15 * 0.55 + 0.10 * 0.60,
        "HIGH": 0.35 * 1.00 + 0.20 * 1.00 + 0.20 * 1.00 + 0.15 * 1.00 + 0.10 * 1.00,
    }
    key = str(hw).upper()
    if key in table:
        return float(table[key])
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _level_map(group: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for _, r in group.iterrows():
        out[str(r["level"])] = r
    return out


def _utility_at(level_rows: Dict[str, pd.Series], level: str, lambda_: float) -> float:
    r = level_rows.get(level)
    if r is None:
        return float("nan")
    return _compute_utility(float(r["macro_f1"]), float(r["normalized_cost"]), lambda_)


def _oracle_action_utility(
    level_rows: Dict[str, pd.Series], lambda_: float
) -> Tuple[str, float]:
    best_lv, best_u = "P0", -np.inf
    for lv in LEVELS:
        u = _utility_at(level_rows, lv, lambda_)
        if not np.isnan(u) and u > best_u:
            best_u = float(u)
            best_lv = lv
    if best_u == -np.inf:
        return "P0", float("nan")
    return best_lv, best_u


def _greedy_probe_action(level_rows: Dict[str, pd.Series], lambda_: float) -> str:
    best_lv, best_u = "P0", -np.inf
    for lv in PROBE_LEVELS:
        u = _utility_at(level_rows, lv, lambda_)
        if not np.isnan(u) and u > best_u:
            best_u = float(u)
            best_lv = lv
    return best_lv


def _session_meta_from_path(path: Path, df: pd.DataFrame) -> Dict:
    """Derive config metadata from filename and/or CSV columns."""
    stem = path.name.replace("_client_level.csv", "")
    try:
        dataset, alpha, seed, num_clients = parse_config_id(stem)
        config_id = stem
    except ValueError:
        dataset = str(df["dataset"].iloc[0]) if "dataset" in df.columns else "unknown"
        alpha = float(df["dirichlet_alpha"].iloc[0]) if "dirichlet_alpha" in df.columns else float(df["alpha"].iloc[0])
        seed = int(df["seed"].iloc[0])
        num_clients = int(df["num_clients"].iloc[0]) if "num_clients" in df.columns else int(df["client_id"].nunique())
        config_id = f"{dataset}_s{seed}_a{alpha}_n{num_clients}"

    if "dataset" in df.columns and pd.notna(df["dataset"].iloc[0]):
        dataset = str(df["dataset"].iloc[0])
    if "dirichlet_alpha" in df.columns:
        alpha = float(df["dirichlet_alpha"].iloc[0])
    elif "alpha" in df.columns:
        alpha = float(df["alpha"].iloc[0])
    if "seed" in df.columns:
        seed = int(df["seed"].iloc[0])
    if "num_clients" in df.columns:
        num_clients = int(df["num_clients"].iloc[0])

    return {
        "config_id": config_id,
        "dataset": dataset,
        "alpha": alpha,
        "seed": seed,
        "num_clients": num_clients,
        "session_id": make_session_id(dataset, alpha, seed),
    }


def build_policy_learning_rows(
    sessions_dir: Path | str,
    lambdas: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Build one policy-learning row per (session, client, lambda).

    For each session CSV × λ ∈ {0.1, 0.3, 1.0} × client:
      - utilities at P0–P5 via Macro-F1 − λ · normalized_cost
      - oracle_action / oracle_utility
      - greedy_probe_action = argmax among P0–P2
      - context + probe features

    Identity columns (session_id, client_instance_id, group_id) are added.
    """
    sessions_dir = Path(sessions_dir)
    lambdas = list(lambdas) if lambdas is not None else list(DEFAULT_LAMBDAS)
    paths = sorted(sessions_dir.glob("*_client_level.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_client_level.csv under {sessions_dir}")

    rows: List[Dict] = []
    for path in paths:
        df = pd.read_csv(path)
        meta = _session_meta_from_path(path, df)

        for cid, group in df.groupby("client_id"):
            level_rows = _level_map(group)
            base = group.iloc[0]
            hw = str(base.get("hardware", base.get("hardware_profile", "MEDIUM")))
            hw_mult = float(base.get("hardware_multiplier", np.nan))
            het = float(base.get("heterogeneity", np.nan))

            probe_f1 = {}
            probe_cost = {}
            for lv in PROBE_LEVELS:
                key = lv.lower()
                r = level_rows.get(lv)
                if r is None:
                    probe_f1[key] = np.nan
                    probe_cost[key] = np.nan
                else:
                    probe_f1[key] = float(r["macro_f1"])
                    probe_cost[key] = float(r["normalized_cost"])

            for lam in lambdas:
                utils = {lv: _utility_at(level_rows, lv, lam) for lv in LEVELS}
                oracle_action, oracle_utility = _oracle_action_utility(level_rows, lam)
                greedy = _greedy_probe_action(level_rows, lam)

                row: Dict = {
                    "config_id": meta["config_id"],
                    "dataset": meta["dataset"],
                    "alpha": meta["alpha"],
                    "seed": meta["seed"],
                    "num_clients": meta["num_clients"],
                    "client_id": int(cid),
                    "lambda": float(lam),
                    "hardware_profile": hw,
                    "hardware_score": _hardware_score(hw),
                    "hardware_multiplier": hw_mult,
                    "heterogeneity": het,
                    "p0_macro_f1": probe_f1["p0"],
                    "p1_macro_f1": probe_f1["p1"],
                    "p2_macro_f1": probe_f1["p2"],
                    "cost_p0": probe_cost["p0"],
                    "cost_p1": probe_cost["p1"],
                    "cost_p2": probe_cost["p2"],
                    "oracle_action": oracle_action,
                    "oracle_utility": oracle_utility,
                    "greedy_probe_action": greedy,
                    "greedy_probe_utility": utils.get(greedy, np.nan),
                }
                for lv in LEVELS:
                    row[f"utility_{lv.lower()}"] = utils[lv]
                rows.append(row)

    out = pd.DataFrame(rows)
    out = add_identity_columns(out)
    # Stable column order for reproducibility
    out = out.sort_values(
        ["dataset", "alpha", "seed", "lambda", "client_id"]
    ).reset_index(drop=True)
    return out


def list_session_files(sessions_dir: Path | str) -> List[Path]:
    return sorted(Path(sessions_dir).glob("*_client_level.csv"))
