"""
PTB-XL dataset loader for binary ECG classification (Normal vs Abnormal).

Designed for single-lead (Lead II) initially; channel dimension is kept explicit
so 12-lead support can be added by changing `lead_indices` without rewriting
the rest of the pipeline.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

# Standard 12-lead order in PTB-XL WFDB records
LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _parse_scp_codes(raw: str) -> Dict[str, float]:
    """Parse SCP-ECG code dictionary stored as string in ptbxl_database.csv."""
    if isinstance(raw, dict):
        return raw
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}


def binary_label_from_scp(scp_codes: Dict[str, float], min_certainty: float = 80.0) -> int:
    """
    Map SCP codes to binary label.

    0 = Normal (NORM present with sufficient certainty, no other major codes)
    1 = Abnormal
    """
    if not scp_codes:
        return 1
    norm_score = scp_codes.get("NORM", 0.0)
    # Significant non-NORM diagnostic codes
    other = [v for k, v in scp_codes.items() if k != "NORM" and v >= min_certainty]
    if norm_score >= min_certainty and len(other) == 0:
        return 0
    return 1


def load_ptbxl_metadata(data_dir: Path) -> pd.DataFrame:
    """Load and preprocess PTB-XL metadata CSV."""
    csv_path = data_dir / "ptbxl_database.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"PTB-XL metadata not found at {csv_path}. "
            "Download PTB-XL from PhysioNet and place files under data/ptbxl/."
        )
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(_parse_scp_codes)
    df["label"] = df["scp_codes"].apply(binary_label_from_scp)
    return df


def load_ecg_signal(
    record_path: Path,
    lead_indices: Sequence[int],
    target_length: int,
) -> np.ndarray:
    """
    Load ECG waveform(s) and resample to fixed length.

    Returns array of shape (num_leads, target_length).
    """
    import wfdb

    record_stem = str(record_path.with_suffix(""))
    sig, _ = wfdb.rdsamp(record_stem)
    # sig shape: (time, 12)
    leads = sig[:, list(lead_indices)].astype(np.float32)
    # Per-lead z-score normalization
    leads = (leads - leads.mean(axis=0, keepdims=True)) / (leads.std(axis=0, keepdims=True) + 1e-8)
    resampled = np.stack([resample(leads[:, i], target_length) for i in range(leads.shape[1])], axis=0)
    return resampled.astype(np.float32)


class ECGDataset(Dataset):
    """PyTorch dataset wrapping preloaded ECG arrays and labels."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = torch.from_numpy(signals)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.labels[idx]


def resolve_record_path(data_dir: Path, row: pd.Series) -> Optional[Path]:
    """
    Resolve WFDB record path for a metadata row.

    Prefer 500 Hz (filename_hr / records500); fall back to 100 Hz (filename_lr)
    when only records100 is available on disk.
    """
    for col in ("filename_hr", "filename_lr"):
        if col not in row.index or pd.isna(row[col]):
            continue
        path = data_dir / str(row[col])
        if path.with_suffix(".hea").exists():
            return path
    return None


def build_ptbxl_arrays(
    data_dir: Path,
    lead_indices: Sequence[int] = (1,),
    target_length: int = 1000,
    max_records: Optional[int] = None,
    strat_fold: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load PTB-XL records into memory.

    Parameters
    ----------
    strat_fold : if set, use official train/test fold split (fold != strat_fold -> train pool)
    """
    df = load_ptbxl_metadata(data_dir)
    if "patient_id" not in df.columns:
        raise ValueError("PTB-XL metadata missing patient_id column")
    n_missing_pid = int(df["patient_id"].isna().sum())
    if n_missing_pid:
        df = df[~df["patient_id"].isna()].copy()
    if strat_fold is not None:
        df = df[df["strat_fold"] != strat_fold]

    # Keep only records whose WFDB files exist on disk (partial local mirrors).
    resolvable = []
    for ecg_id, row in df.iterrows():
        if resolve_record_path(data_dir, row) is not None:
            resolvable.append(ecg_id)
    df = df.loc[resolvable].copy()
    if len(df) == 0:
        raise RuntimeError("No PTB-XL WFDB records found on disk.")

    if max_records is not None and len(df) > max_records:
        # Patient-safe subsample: keep whole patients (ptbxl_patient_disjoint_v2)
        from fedpbg.data.patient_disjoint import subsample_patients_for_max_records

        df = subsample_patients_for_max_records(df, max_records=max_records, random_seed=42)

    records_dir = data_dir / "records500"
    signals: List[np.ndarray] = []
    labels: List[int] = []
    kept_ids: List[int] = []

    for ecg_id, row in df.iterrows():
        path = resolve_record_path(data_dir, row)
        if path is None:
            continue
        try:
            sig = load_ecg_signal(path, lead_indices, target_length)
            signals.append(sig)
            labels.append(int(row["label"]))
            kept_ids.append(int(ecg_id))
        except Exception:
            continue

    if not signals:
        raise RuntimeError("No PTB-XL records could be loaded. Check data path and WFDB files.")

    X = np.stack(signals, axis=0)
    y = np.array(labels, dtype=np.int64)
    meta = df.loc[kept_ids].copy()
    return X, y, meta


def build_synthetic_arrays(
    n_samples: int = 2000,
    n_leads: int = 1,
    target_length: int = 1000,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate synthetic ECG-like signals for smoke testing without PTB-XL download.

    Normal vs Abnormal classes differ in amplitude/noise so models can learn something.
    """
    rng = np.random.default_rng(random_seed)
    X = np.zeros((n_samples, n_leads, target_length), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    t = np.linspace(0, 2 * np.pi, target_length)

    for i in range(n_samples):
        label = rng.integers(0, 2)
        y[i] = label
        for c in range(n_leads):
            freq = 1.2 if label == 0 else 1.8
            amp = 0.5 if label == 0 else 1.0
            noise = rng.normal(0, 0.05 if label == 0 else 0.15, target_length)
            X[i, c] = (amp * np.sin(freq * t) + noise).astype(np.float32)

    meta = pd.DataFrame({"ecg_id": np.arange(n_samples), "label": y})
    meta.index.name = "ecg_id"
    return X, y, meta
