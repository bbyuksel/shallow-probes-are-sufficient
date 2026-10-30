"""Post-hoc cost-model sensitivity via rebuilt utilities + GroupKFold FedPBA retrain."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from paper2.src.methodology_v2.build_rows import (
    CONTEXT_PROBES_FEATURES,
    DEFAULT_LAMBDAS,
    LEVELS,
    PROBE_LEVELS,
    _compute_utility,
    _hardware_score,
)
from paper2.src.methodology_v2.grouped_eval import run_group_kfold
from paper2.src.methodology_v2.ids import add_identity_columns, make_session_id, parse_config_id

WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    "default": {"w_param": 0.25, "w_flop": 0.20, "w_time": 0.25, "w_comm": 0.15, "w_mem": 0.15},
    "equal": {"w_param": 0.20, "w_flop": 0.20, "w_time": 0.20, "w_comm": 0.20, "w_mem": 0.20},
    "compute_heavy": {"w_param": 0.20, "w_flop": 0.30, "w_time": 0.30, "w_comm": 0.10, "w_mem": 0.10},
    "comm_heavy": {"w_param": 0.10, "w_flop": 0.10, "w_time": 0.15, "w_comm": 0.50, "w_mem": 0.15},
    "mem_heavy": {"w_param": 0.10, "w_flop": 0.10, "w_time": 0.15, "w_comm": 0.15, "w_mem": 0.50},
    "time_heavy": {"w_param": 0.10, "w_flop": 0.10, "w_time": 0.50, "w_comm": 0.15, "w_mem": 0.15},
}

# Prefer completeness: N1 all profiles; N2 all profiles (sklearn CV only).
N1_PROFILES = list(WEIGHT_PROFILES.keys())
N2_PROFILES = list(WEIGHT_PROFILES.keys())


def _assert_weights_sum_to_one(w: Dict[str, float], tol: float = 1e-9) -> None:
    s = sum(w.values())
    if abs(s - 1.0) > tol:
        raise ValueError(f"Weight profile must sum to 1, got {s}: {w}")


for _name, _w in WEIGHT_PROFILES.items():
    _assert_weights_sum_to_one(_w)


def _minmax(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return np.zeros_like(a, dtype=float)
    return (a - lo) / (hi - lo)


def _components(row: pd.Series) -> Dict[str, float]:
    return {
        "param_term": float(row["trainable_params"]) / 1e5,
        "flop_term": float(row["flops"]) / 1e8,
        "time_term": float(row["train_time_sec"]),
        "comm_term": float(row["comm_bytes"]) / 1e6,
        "mem_term": float(row["memory_proxy_mb"]) / 100.0,
        "hardware_multiplier": float(row["hardware_multiplier"]),
    }


def normalize_costs_for_client(
    level_df: pd.DataFrame,
    weights: Dict[str, float],
    normalization: str,
) -> Dict[str, float]:
    """
    Return level -> normalized_cost for one client (6 levels).

    N1: weighted raw sum * hardware_multiplier, then min-max across levels.
    N2: min-max each component across levels, then weighted sum (weights sum to 1).
    """
    levels_present = [str(lv) for lv in level_df["level"].astype(str).tolist()]
    comps = {lv: _components(level_df[level_df["level"].astype(str) == lv].iloc[0]) for lv in levels_present}

    if normalization == "N1":
        raw = []
        order = []
        for lv in LEVELS:
            if lv not in comps:
                continue
            c = comps[lv]
            r = (
                weights["w_param"] * c["param_term"]
                + weights["w_flop"] * c["flop_term"]
                + weights["w_time"] * c["time_term"]
                + weights["w_comm"] * c["comm_term"]
                + weights["w_mem"] * c["mem_term"]
            ) * c["hardware_multiplier"]
            raw.append(r)
            order.append(lv)
        normed = _minmax(np.asarray(raw, dtype=float))
        return {lv: float(v) for lv, v in zip(order, normed)}

    if normalization == "N2":
        # Component-first: normalize each term across levels, then weighted sum.
        term_names = ["param_term", "flop_term", "time_term", "comm_term", "mem_term"]
        order = [lv for lv in LEVELS if lv in comps]
        if not order:
            return {}
        norm_terms: Dict[str, np.ndarray] = {}
        for t in term_names:
            vals = np.asarray([comps[lv][t] for lv in order], dtype=float)
            norm_terms[t] = _minmax(vals)
        out = {}
        for i, lv in enumerate(order):
            out[lv] = float(
                weights["w_param"] * norm_terms["param_term"][i]
                + weights["w_flop"] * norm_terms["flop_term"][i]
                + weights["w_time"] * norm_terms["time_term"][i]
                + weights["w_comm"] * norm_terms["comm_term"][i]
                + weights["w_mem"] * norm_terms["mem_term"][i]
            )
        return out

    raise ValueError(f"Unknown normalization: {normalization}")


def _session_meta(path: Path, df: pd.DataFrame) -> Dict[str, Any]:
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


def build_rows_for_cost_scenario(
    sessions_dir: Path | str,
    weights: Dict[str, float],
    normalization: str,
    lambdas: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Rebuild policy-learning rows under a cost weight/normalization scenario."""
    sessions_dir = Path(sessions_dir)
    lambdas = list(lambdas) if lambdas is not None else list(DEFAULT_LAMBDAS)
    paths = sorted(sessions_dir.glob("*_client_level.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_client_level.csv under {sessions_dir}")

    required = [
        "trainable_params",
        "flops",
        "train_time_sec",
        "comm_bytes",
        "memory_proxy_mb",
        "hardware_multiplier",
        "macro_f1",
        "level",
        "client_id",
    ]
    rows: List[Dict[str, Any]] = []
    for path in paths:
        df = pd.read_csv(path)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")
        meta = _session_meta(path, df)

        for cid, group in df.groupby("client_id"):
            level_rows = {str(r["level"]): r for _, r in group.iterrows()}
            costs = normalize_costs_for_client(group, weights, normalization)
            base = group.iloc[0]
            hw = str(base.get("hardware", base.get("hardware_profile", "MEDIUM")))
            hw_mult = float(base.get("hardware_multiplier", np.nan))
            het = float(base.get("heterogeneity", np.nan))

            probe_f1 = {}
            probe_cost = {}
            for lv in PROBE_LEVELS:
                key = lv.lower()
                r = level_rows.get(lv)
                if r is None or lv not in costs:
                    probe_f1[key] = np.nan
                    probe_cost[key] = np.nan
                else:
                    probe_f1[key] = float(r["macro_f1"])
                    probe_cost[key] = float(costs[lv])

            for lam in lambdas:
                utils: Dict[str, float] = {}
                for lv in LEVELS:
                    r = level_rows.get(lv)
                    if r is None or lv not in costs:
                        utils[lv] = float("nan")
                    else:
                        utils[lv] = _compute_utility(float(r["macro_f1"]), float(costs[lv]), float(lam))

                best_lv, best_u = "P0", -np.inf
                for lv in LEVELS:
                    u = utils[lv]
                    if not np.isnan(u) and u > best_u:
                        best_u = float(u)
                        best_lv = lv
                if best_u == -np.inf:
                    best_u = float("nan")

                greedy_lv, greedy_u = "P0", -np.inf
                for lv in PROBE_LEVELS:
                    u = utils[lv]
                    if not np.isnan(u) and u > greedy_u:
                        greedy_u = float(u)
                        greedy_lv = lv
                if greedy_u == -np.inf:
                    greedy_u = float("nan")

                row: Dict[str, Any] = {
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
                    "oracle_action": best_lv,
                    "oracle_utility": best_u,
                    "greedy_probe_action": greedy_lv,
                    "greedy_probe_utility": greedy_u,
                }
                for lv in LEVELS:
                    row[f"utility_{lv.lower()}"] = utils[lv]
                rows.append(row)

    out = pd.DataFrame(rows)
    out = add_identity_columns(out)
    return out.sort_values(["dataset", "alpha", "seed", "lambda", "client_id"]).reset_index(drop=True)


def _aggregate_scenario_metrics(oof: pd.DataFrame) -> Dict[str, Any]:
    deep = oof["oracle_action"].astype(str).isin(["P3", "P4", "P5"])
    shallow_pct = float((~deep).mean() * 100.0)
    deep_count = int(deep.sum())

    def _pol(action_col: str, util_col: str) -> Dict[str, float]:
        u = oof[util_col].astype(float)
        ou = oof["oracle_utility"].astype(float)
        regret = ou - u
        denom = float(ou.sum())
        recovery = float(u.sum() / denom * 100.0) if denom != 0 else float("nan")
        return {
            "mean_regret": float(regret.mean()),
            "oracle_utility_recovery_pct": recovery,
        }

    greedy = _pol("greedy_action", "greedy_utility")
    fed = _pol("fedpba_action", "fedpba_utility")
    bu = _pol("best_uniform_action", "best_uniform_utility")
    ctx = _pol("context_only_action", "context_only_utility")

    greedy_p0_deep = int(
        ((oof["greedy_action"].astype(str) == "P0") & deep).sum()
    )
    mean_deep_greedy_regret = float("nan")
    if deep.any():
        mean_deep_greedy_regret = float(
            (oof.loc[deep, "oracle_utility"].astype(float) - oof.loc[deep, "greedy_utility"].astype(float)).mean()
        )

    return {
        "n_rows": int(len(oof)),
        "shallow_oracle_pct": shallow_pct,
        "deep_oracle_count": deep_count,
        "greedy_mean_regret": greedy["mean_regret"],
        "greedy_recovery_pct": greedy["oracle_utility_recovery_pct"],
        "fedpba_mean_regret": fed["mean_regret"],
        "fedpba_recovery_pct": fed["oracle_utility_recovery_pct"],
        "best_uniform_mean_regret": bu["mean_regret"],
        "best_uniform_recovery_pct": bu["oracle_utility_recovery_pct"],
        "context_only_mean_regret": ctx["mean_regret"],
        "context_only_recovery_pct": ctx["oracle_utility_recovery_pct"],
        "greedy_p0_to_deep_count": greedy_p0_deep,
        "mean_deep_oracle_regret_greedy": mean_deep_greedy_regret,
    }


def run_one_scenario(
    sessions_dir: Path | str,
    profile_name: str,
    normalization: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    weights = WEIGHT_PROFILES[profile_name]
    rows = build_rows_for_cost_scenario(sessions_dir, weights, normalization)
    # Ensure required feature columns exist
    missing_feat = [c for c in CONTEXT_PROBES_FEATURES if c not in rows.columns]
    if missing_feat:
        raise ValueError(f"Missing features for FedPBA: {missing_feat}")
    oof, metrics, _integrity = run_group_kfold(
        rows, n_splits=n_splits, random_state=random_state, out_dir=None
    )
    agg = _aggregate_scenario_metrics(oof)
    return {
        "profile": profile_name,
        "normalization": normalization,
        **weights,
        **agg,
        "status": "ok",
        "error": "",
    }


def run_cost_sensitivity(
    sessions_dir: Path | str,
    out_dir: Path | str,
    n_splits: int = 5,
    random_state: int = 42,
    n1_profiles: Optional[Sequence[str]] = None,
    n2_profiles: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Run cost sensitivity scenarios; write summary CSV + main-text JSON.

    Continues on per-scenario errors (records error, does not abort).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n1_profiles = list(n1_profiles) if n1_profiles is not None else list(N1_PROFILES)
    n2_profiles = list(n2_profiles) if n2_profiles is not None else list(N2_PROFILES)

    scenarios: List[Tuple[str, str]] = []
    for p in n1_profiles:
        scenarios.append((p, "N1"))
    for p in n2_profiles:
        scenarios.append((p, "N2"))

    results: List[Dict[str, Any]] = []
    for i, (profile, norm) in enumerate(scenarios, start=1):
        print(f"[cost_sensitivity] ({i}/{len(scenarios)}) {profile} / {norm}", flush=True)
        try:
            row = run_one_scenario(
                sessions_dir,
                profile,
                norm,
                n_splits=n_splits,
                random_state=random_state,
            )
            results.append(row)
            print(
                f"  ok greedy_regret={row['greedy_mean_regret']:.6f} "
                f"shallow%={row['shallow_oracle_pct']:.2f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — continue on scenario failure
            err = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {err}", flush=True)
            traceback.print_exc()
            results.append(
                {
                    "profile": profile,
                    "normalization": norm,
                    "status": "error",
                    "error": err,
                }
            )

    summary = pd.DataFrame(results)
    summary.to_csv(out_dir / "cost_sensitivity_summary.csv", index=False)

    if "status" not in summary.columns:
        summary["status"] = "ok"
    ok = summary[summary["status"] == "ok"].copy()

    main_text: Dict[str, Any] = {
        "n_scenarios_attempted": int(len(summary)),
        "n_scenarios_ok": int(len(ok)),
        "n_scenarios_error": int((summary["status"] == "error").sum()),
    }
    if len(ok):
        main_text.update(
            {
                "greedy_regret_min": float(ok["greedy_mean_regret"].min()),
                "greedy_regret_max": float(ok["greedy_mean_regret"].max()),
                "greedy_recovery_min": float(ok["greedy_recovery_pct"].min()),
                "greedy_recovery_max": float(ok["greedy_recovery_pct"].max()),
                "shallow_oracle_pct_min": float(ok["shallow_oracle_pct"].min()),
                "shallow_oracle_pct_max": float(ok["shallow_oracle_pct"].max()),
                "profiles_ok": ok[["profile", "normalization"]].to_dict(orient="records"),
            }
        )
    else:
        main_text["error"] = "no successful cost-sensitivity scenarios"

    with open(out_dir / "cost_sensitivity_main_text_summary.json", "w", encoding="utf-8") as f:
        json.dump(main_text, f, indent=2)

    return {"summary": summary, "main_text": main_text}
