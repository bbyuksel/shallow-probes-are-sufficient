"""
Unified dataset registry for FedSUCP oracle sparsity validation.

Each entry defines modality, class count, input shape, and a loader function.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from fedpbg.data.cifar import load_cifar_arrays
from fedpbg.data.femnist import femnist_available, load_femnist_arrays
from fedpbg.data.image_dataset import ImageTensorDataset
from fedpbg.data.ptbxl import ECGDataset, build_ptbxl_arrays, build_synthetic_arrays


@dataclass
class DatasetSpec:
    name: str
    modality: str  # 'ecg' | 'image'
    num_classes: int
    in_channels: int
    input_size: int  # signal length (1D) or spatial size (2D)
    loader: Callable


def _attach_patient_ids(ds: Dataset, patient_ids: Optional[np.ndarray]) -> Dataset:
    setattr(ds, "patient_ids", patient_ids)
    return ds


def _load_ptbxl(
    data_dir: Path,
    max_records: Optional[int] = None,
    quick: bool = False,
    **_,
) -> Tuple[Dataset, np.ndarray, str]:
    if quick:
        X, y, meta = build_synthetic_arrays(
            n_samples=max_records or 800,
            n_leads=1,
            target_length=1000,
        )
        pids = np.arange(len(y))
        return _attach_patient_ids(ECGDataset(X, y), pids), y, "ptbxl_synthetic"

    ptb_dir = data_dir / "ptbxl"
    try:
        X, y, meta = build_ptbxl_arrays(
            ptb_dir,
            lead_indices=(1,),
            target_length=1000,
            max_records=max_records,
        )
        if "patient_id" not in meta.columns:
            raise ValueError("PTB-XL meta missing patient_id after load")
        pids = meta["patient_id"].to_numpy()
        return _attach_patient_ids(ECGDataset(X, y), pids), y, "ptbxl"
    except FileNotFoundError:
        X, y, meta = build_synthetic_arrays(
            n_samples=max_records or 2000,
            n_leads=1,
            target_length=1000,
        )
        pids = np.arange(len(y))
        return _attach_patient_ids(ECGDataset(X, y), pids), y, "ptbxl_synthetic"


def _load_cifar100(data_dir: Path, **_) -> Tuple[Dataset, np.ndarray, str]:
    X, y = load_cifar_arrays("cifar100", data_dir / "cifar100")
    return _attach_patient_ids(ImageTensorDataset(X, y), None), y, "cifar100"


def _load_cifar10(data_dir: Path, **_) -> Tuple[Dataset, np.ndarray, str]:
    X, y = load_cifar_arrays("cifar10", data_dir / "cifar10")
    return _attach_patient_ids(ImageTensorDataset(X, y), None), y, "cifar10"


def _load_femnist(data_dir: Path, **_) -> Tuple[Dataset, np.ndarray, str]:
    fem_dir = data_dir / "femnist"
    X, y, fallback = load_femnist_arrays(fem_dir)
    if fallback or len(y) == 0:
        X, y = load_cifar_arrays("cifar10", data_dir / "cifar10")
        return _attach_patient_ids(ImageTensorDataset(X, y), None), y, "cifar10_fallback"
    return _attach_patient_ids(ImageTensorDataset(X, y), None), y, "femnist"


DATASET_REGISTRY: dict = {
    "ptbxl": DatasetSpec(
        name="ptbxl",
        modality="ecg",
        num_classes=2,
        in_channels=1,
        input_size=1000,
        loader=_load_ptbxl,
    ),
    "cifar100": DatasetSpec(
        name="cifar100",
        modality="image",
        num_classes=100,
        in_channels=3,
        input_size=32,
        loader=_load_cifar100,
    ),
    "femnist": DatasetSpec(
        name="femnist",
        modality="image",
        num_classes=62,
        in_channels=1,
        input_size=28,
        loader=_load_femnist,
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        modality="image",
        num_classes=10,
        in_channels=3,
        input_size=32,
        loader=_load_cifar10,
    ),
}


def load_registered_dataset(
    name: str,
    data_root: Path,
    quick: bool = False,
    max_records: Optional[int] = None,
) -> Tuple[DatasetSpec, Dataset, np.ndarray, str]:
    """
    Load dataset by registry name.

    Returns (spec, pytorch_dataset, labels, resolved_name).
    For PTB-XL, ``dataset.patient_ids`` holds per-record patient IDs.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset {name}. Choose from {list(DATASET_REGISTRY)}")

    spec = DATASET_REGISTRY[name]
    if name == "femnist" and not femnist_available(data_root / "femnist") and not quick:
        ds, y, resolved = _load_femnist(data_root)
        fallback_spec = DATASET_REGISTRY["cifar10"]
        return fallback_spec, ds, y, resolved

    ds, y, resolved = spec.loader(
        data_root,
        quick=quick,
        max_records=max_records,
    )
    if resolved == "cifar10_fallback":
        return DATASET_REGISTRY["cifar10"], ds, y, resolved
    if resolved == "ptbxl_synthetic":
        return spec, ds, y, resolved
    return spec, ds, y, resolved
