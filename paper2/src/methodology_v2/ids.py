"""Session / client identity helpers for grouped evaluation (no leakage)."""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

import pandas as pd

# config_id pattern: {dataset}_s{seed}_a{alpha}_n{num_clients}
_CONFIG_RE = re.compile(
    r"^(?P<dataset>.+)_s(?P<seed>\d+)_a(?P<alpha>[0-9.]+)_n(?P<num_clients>\d+)$"
)


def make_session_id(dataset: Any, alpha: Any, seed: Any) -> str:
    """Canonical FedAvg session identity: dataset × alpha × seed."""
    return f"{dataset}__a{alpha}__seed{seed}"


def make_client_instance_id(session_id: str, client_id: Any) -> str:
    """Client within a session (all λ copies share this id)."""
    return f"{session_id}__client{client_id}"


def parse_config_id(config_id: str) -> Tuple[str, float, int, Optional[int]]:
    """
    Parse ``cifar10_s42_a0.1_n20`` → (dataset, alpha, seed, num_clients).

    Raises ValueError if the string does not match the expected pattern.
    """
    m = _CONFIG_RE.match(str(config_id).strip())
    if not m:
        raise ValueError(f"Unrecognized config_id: {config_id!r}")
    return (
        m.group("dataset"),
        float(m.group("alpha")),
        int(m.group("seed")),
        int(m.group("num_clients")),
    )


def _derive_dataset_alpha_seed(row: pd.Series) -> Tuple[str, float, int]:
    """Prefer explicit columns; fall back to parsing config_id."""
    dataset = row.get("dataset")
    alpha = row.get("alpha", row.get("dirichlet_alpha"))
    seed = row.get("seed")

    if pd.isna(dataset) or dataset is None or pd.isna(alpha) or pd.isna(seed):
        if "config_id" not in row.index or pd.isna(row.get("config_id")):
            raise ValueError(
                "Cannot derive identity: need dataset/alpha/seed columns or config_id"
            )
        ds, a, s, _ = parse_config_id(str(row["config_id"]))
        dataset = dataset if not (pd.isna(dataset) or dataset is None) else ds
        alpha = alpha if not pd.isna(alpha) else a
        seed = seed if not pd.isna(seed) else s

    return str(dataset), float(alpha), int(seed)


def add_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``session_id``, ``client_instance_id``, and ``group_id`` (= session_id).

    Derives dataset / alpha / seed from columns when present, otherwise parses
    ``config_id`` (e.g. ``cifar10_s42_a0.1_n20``). Does not invent missing
    client_id values.
    """
    out = df.copy()
    if "client_id" not in out.columns:
        raise ValueError("add_identity_columns requires a client_id column")

    session_ids = []
    client_instance_ids = []
    datasets, alphas, seeds = [], [], []

    for _, row in out.iterrows():
        dataset, alpha, seed = _derive_dataset_alpha_seed(row)
        sid = make_session_id(dataset, alpha, seed)
        cid = make_client_instance_id(sid, int(row["client_id"]))
        session_ids.append(sid)
        client_instance_ids.append(cid)
        datasets.append(dataset)
        alphas.append(alpha)
        seeds.append(seed)

    if "dataset" not in out.columns or out["dataset"].isna().any():
        out["dataset"] = datasets
    if "alpha" not in out.columns or out["alpha"].isna().any():
        out["alpha"] = alphas
    if "seed" not in out.columns or out["seed"].isna().any():
        out["seed"] = seeds

    out["session_id"] = session_ids
    out["client_instance_id"] = client_instance_ids
    out["group_id"] = session_ids
    return out
