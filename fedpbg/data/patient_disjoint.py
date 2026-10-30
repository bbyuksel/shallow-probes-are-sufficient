"""
Patient-disjoint federated partitioning for PTB-XL.

Protocol (sinconf2026_methodology_v2 / ptbxl_patient_disjoint_v2):
1. Load records with patient_id; drop rows missing patient_id.
2. Patient-safe subsample to approximately max_records.
3. Global patient-level train/val/test (70/15/15), stratified by patient label.
4. Assign each patient to exactly one federated client (Dirichlet on patient labels).
5. Client train/val/test indices come only from the matching global split.

Patient label rule: Abnormal (1) if any record of the patient is Abnormal; else Normal (0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedpbg.data.federated_split import ClientData, dirichlet_partition, rebalance_empty_clients


def patient_label_from_records(record_labels: Sequence[int]) -> int:
    """Deterministic patient label: any Abnormal => Abnormal."""
    arr = np.asarray(record_labels, dtype=np.int64)
    if len(arr) == 0:
        return 1
    return int(1 if np.any(arr == 1) else 0)


def subsample_patients_for_max_records(
    meta: pd.DataFrame,
    max_records: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Subsample whole patients until the retained record count is <= max_records.
    Never splits a patient's records across keep/drop.
    """
    if "patient_id" not in meta.columns:
        raise ValueError("meta must contain patient_id")
    if len(meta) <= max_records:
        return meta.copy()

    rng = np.random.default_rng(random_seed)
    patients = meta.groupby("patient_id").size().rename("n_records").reset_index()
    # Shuffle patients, then greedily add until capacity
    order = rng.permutation(len(patients))
    kept_patients: List = []
    n_kept = 0
    for i in order:
        pid = patients.iloc[i]["patient_id"]
        n = int(patients.iloc[i]["n_records"])
        if n_kept + n > max_records and n_kept > 0:
            continue
        if n_kept + n > max_records and n_kept == 0:
            # Single patient larger than budget: keep it (cannot split)
            kept_patients.append(pid)
            n_kept += n
            break
        kept_patients.append(pid)
        n_kept += n
        if n_kept >= max_records:
            break
    return meta[meta["patient_id"].isin(kept_patients)].copy()


def build_patient_table(meta: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """One row per patient with aggregated label and record count."""
    if len(meta) != len(labels):
        raise ValueError("meta and labels length mismatch")
    tmp = meta.copy()
    tmp = tmp.reset_index(drop=True)
    tmp["record_label"] = np.asarray(labels, dtype=np.int64)
    if tmp["patient_id"].isna().any():
        n_miss = int(tmp["patient_id"].isna().sum())
        raise ValueError(f"{n_miss} records missing patient_id; refuse to invent IDs")

    rows = []
    for pid, g in tmp.groupby("patient_id"):
        rows.append(
            {
                "patient_id": pid,
                "patient_label": patient_label_from_records(g["record_label"].values),
                "n_records": len(g),
                "n_normal_records": int((g["record_label"] == 0).sum()),
                "n_abnormal_records": int((g["record_label"] == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def global_patient_split(
    patient_table: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Stratified patient-level train/val/test split. Returns patient_id arrays."""
    pids = patient_table["patient_id"].values
    y = patient_table["patient_label"].values
    # Stratify when both classes present with enough patients
    strat = y if len(np.unique(y)) > 1 else None
    try:
        train_pid, temp_pid, y_train, y_temp = train_test_split(
            pids, y, train_size=train_ratio, random_state=random_seed, stratify=strat, shuffle=True
        )
    except ValueError:
        train_pid, temp_pid, y_train, y_temp = train_test_split(
            pids, y, train_size=train_ratio, random_state=random_seed, stratify=None, shuffle=True
        )
    val_rel = val_ratio / max(1e-12, (1.0 - train_ratio))
    strat_temp = y_temp if len(np.unique(y_temp)) > 1 else None
    try:
        val_pid, test_pid = train_test_split(
            temp_pid, train_size=val_rel, random_state=random_seed, stratify=strat_temp, shuffle=True
        )
    except ValueError:
        val_pid, test_pid = train_test_split(
            temp_pid, train_size=val_rel, random_state=random_seed, stratify=None, shuffle=True
        )
    return {
        "train": np.asarray(train_pid),
        "val": np.asarray(val_pid),
        "test": np.asarray(test_pid),
    }


def assign_patients_to_clients(
    patient_ids: np.ndarray,
    patient_labels: np.ndarray,
    num_clients: int,
    alpha: float,
    random_seed: int,
) -> Dict[int, np.ndarray]:
    """Dirichlet-assign patients to clients; returns client_id -> patient_id array."""
    # Map patients to positions, Dirichlet over positions, then map back to patient_ids
    n = len(patient_ids)
    if n == 0:
        return {i: np.array([], dtype=object) for i in range(num_clients)}
    pos_partition = dirichlet_partition(patient_labels, num_clients, alpha, random_seed)
    all_pos = np.arange(n)
    pos_partition = rebalance_empty_clients(pos_partition, num_clients, all_pos, random_seed)
    out: Dict[int, np.ndarray] = {}
    for cid in range(num_clients):
        idxs = pos_partition.get(cid, np.array([], dtype=np.int64))
        out[cid] = patient_ids[np.asarray(idxs, dtype=np.int64)]
    return out


@dataclass
class PatientDisjointSplitResult:
    clients: List[ClientData]
    patient_table: pd.DataFrame
    global_split_patients: Dict[str, np.ndarray]
    client_patients: Dict[int, np.ndarray]
    record_patient_ids: np.ndarray
    split_report_rows: List[Dict]
    protocol: str = "ptbxl_patient_disjoint_v2"


def build_patient_disjoint_client_splits(
    labels: np.ndarray,
    patient_ids: np.ndarray,
    num_clients: int,
    alpha: float,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> PatientDisjointSplitResult:
    """
    Full patient-disjoint federated split.

    Assertions enforced:
    - global train/val/test patient sets pairwise disjoint
    - each patient assigned to exactly one client
    - within each client, train/val/test patients are disjoint
    """
    patient_ids = np.asarray(patient_ids)
    labels = np.asarray(labels, dtype=np.int64)
    if len(patient_ids) != len(labels):
        raise ValueError("patient_ids and labels length mismatch")
    if pd.isna(patient_ids).any():
        raise ValueError("patient_ids contain NaN")

    meta = pd.DataFrame({"patient_id": patient_ids, "label": labels})
    # index aligns with record positions 0..n-1
    meta["record_idx"] = np.arange(len(meta))
    patient_table = build_patient_table(
        meta[["patient_id"]].assign(label=labels),
        labels,
    )
    # rebuild with correct API
    patient_table = (
        meta.groupby("patient_id")
        .agg(
            patient_label=("label", lambda s: patient_label_from_records(s.values)),
            n_records=("label", "size"),
            n_normal_records=("label", lambda s: int((s == 0).sum())),
            n_abnormal_records=("label", lambda s: int((s == 1).sum())),
        )
        .reset_index()
    )

    global_split = global_patient_split(patient_table, train_ratio, val_ratio, random_seed)
    train_set = set(global_split["train"].tolist())
    val_set = set(global_split["val"].tolist())
    test_set = set(global_split["test"].tolist())
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    # 1) Dirichlet-assign TRAIN patients first so FedAvg clients are non-empty.
    train_table = patient_table[patient_table["patient_id"].isin(train_set)].reset_index(drop=True)
    train_client_patients = assign_patients_to_clients(
        train_table["patient_id"].values,
        train_table["patient_label"].values.astype(np.int64),
        num_clients,
        alpha,
        random_seed,
    )
    # Force every client to have at least one train patient when possible.
    train_client_patients = _ensure_nonempty_client_patients(
        train_client_patients, train_table["patient_id"].values, num_clients, random_seed
    )

    # 2) Assign val/test patients via separate Dirichlet draws (same alpha).
    val_table = patient_table[patient_table["patient_id"].isin(val_set)].reset_index(drop=True)
    test_table = patient_table[patient_table["patient_id"].isin(test_set)].reset_index(drop=True)
    val_client_patients = assign_patients_to_clients(
        val_table["patient_id"].values,
        val_table["patient_label"].values.astype(np.int64),
        num_clients,
        alpha,
        random_seed + 1,
    )
    test_client_patients = assign_patients_to_clients(
        test_table["patient_id"].values,
        test_table["patient_label"].values.astype(np.int64),
        num_clients,
        alpha,
        random_seed + 2,
    )

    client_patients: Dict[int, np.ndarray] = {}
    for cid in range(num_clients):
        merged = np.concatenate(
            [
                np.asarray(train_client_patients.get(cid, []), dtype=object),
                np.asarray(val_client_patients.get(cid, []), dtype=object),
                np.asarray(test_client_patients.get(cid, []), dtype=object),
            ]
        )
        client_patients[cid] = merged

    pids = patient_table["patient_id"].values

    # Verify no patient in two clients
    seen = {}
    for cid, cps in client_patients.items():
        for pid in cps:
            if pid in seen:
                raise RuntimeError(f"Patient {pid} assigned to clients {seen[pid]} and {cid}")
            seen[pid] = cid
    if len(seen) != len(pids):
        missing = set(map(str, pids.tolist())) - set(map(str, seen.keys()))
        raise RuntimeError(f"Patients not assigned to any client: {len(missing)}")

    # Redistribute so every client has >=1 val and >=1 test patient when pool allows.
    client_patients = _redistribute_split_patients(
        client_patients, val_set, "val", num_clients, random_seed + 3
    )
    client_patients = _redistribute_split_patients(
        client_patients, test_set, "test", num_clients, random_seed + 4
    )

    clients: List[ClientData] = []
    for cid in range(num_clients):
        cps = set(client_patients[cid].tolist())
        train_pids = cps & train_set
        val_pids = cps & val_set
        test_pids = cps & test_set
        assert train_pids.isdisjoint(val_pids)
        assert train_pids.isdisjoint(test_pids)
        assert val_pids.isdisjoint(test_pids)

        train_idx = meta.loc[meta["patient_id"].isin(train_pids), "record_idx"].to_numpy(dtype=np.int64)
        val_idx = meta.loc[meta["patient_id"].isin(val_pids), "record_idx"].to_numpy(dtype=np.int64)
        test_idx = meta.loc[meta["patient_id"].isin(test_pids), "record_idx"].to_numpy(dtype=np.int64)
        if len(train_idx) == 0:
            raise RuntimeError(f"Client {cid} has empty train set after patient-disjoint assignment")
        if len(val_idx) == 0 or len(test_idx) == 0:
            raise RuntimeError(
                f"Client {cid} has empty val/test after redistribution "
                f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})"
            )
        clients.append(ClientData(client_id=cid, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))

    # Cross-client patient overlap check
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            inter = set(client_patients[i].tolist()) & set(client_patients[j].tolist())
            if inter:
                raise RuntimeError(f"Patient overlap between clients {i} and {j}: {len(inter)}")

    split_report_rows: List[Dict] = []
    for split_name, pset in [("train", train_set), ("val", val_set), ("test", test_set)]:
        sub = patient_table[patient_table["patient_id"].isin(pset)]
        rec = meta[meta["patient_id"].isin(pset)]
        split_report_rows.append(
            {
                "seed": random_seed,
                "split": split_name,
                "num_patients": int(len(sub)),
                "num_records": int(len(rec)),
                "normal_patients": int((sub["patient_label"] == 0).sum()),
                "abnormal_patients": int((sub["patient_label"] == 1).sum()),
                "normal_records": int((rec["label"] == 0).sum()),
                "abnormal_records": int((rec["label"] == 1).sum()),
            }
        )

    return PatientDisjointSplitResult(
        clients=clients,
        patient_table=patient_table,
        global_split_patients=global_split,
        client_patients=client_patients,
        record_patient_ids=patient_ids,
        split_report_rows=split_report_rows,
    )


def _ensure_nonempty_client_patients(
    client_patients: Dict[int, np.ndarray],
    all_patient_ids: np.ndarray,
    num_clients: int,
    random_seed: int,
) -> Dict[int, np.ndarray]:
    """Steal patients from large clients so every client has >=1 patient."""
    rng = np.random.default_rng(random_seed)
    bags = {cid: list(client_patients.get(cid, [])) for cid in range(num_clients)}
    for cid in range(num_clients):
        if len(bags[cid]) > 0:
            continue
        donors = [d for d in range(num_clients) if len(bags[d]) > 1]
        if not donors:
            used = {p for ps in bags.values() for p in ps}
            unused = [p for p in all_patient_ids if p not in used]
            if unused:
                bags[cid].append(unused[0])
            continue
        donor = int(rng.choice(donors))
        bags[cid].append(bags[donor].pop())
    return {cid: np.asarray(bags[cid], dtype=object) for cid in range(num_clients)}


def _redistribute_split_patients(
    client_patients: Dict[int, np.ndarray],
    split_patient_set: set,
    split_name: str,
    num_clients: int,
    random_seed: int,
) -> Dict[int, np.ndarray]:
    """
    Ensure every client has at least one patient from ``split_patient_set``.
    Steals from clients that have >=2 patients in that split.
    """
    rng = np.random.default_rng(random_seed)
    bags = {cid: list(client_patients.get(cid, [])) for cid in range(num_clients)}

    def split_count(cid: int) -> int:
        return sum(1 for p in bags[cid] if p in split_patient_set)

    for cid in range(num_clients):
        if split_count(cid) > 0:
            continue
        donors = [d for d in range(num_clients) if split_count(d) > 1]
        if not donors:
            raise RuntimeError(
                f"Cannot give every client a {split_name} patient; "
                f"pool too small ({len(split_patient_set)} patients for {num_clients} clients)."
            )
        donor = int(rng.choice(donors))
        # pop a split patient from donor
        for i, p in enumerate(bags[donor]):
            if p in split_patient_set:
                bags[cid].append(bags[donor].pop(i))
                break
    return {cid: np.asarray(bags[cid], dtype=object) for cid in range(num_clients)}
