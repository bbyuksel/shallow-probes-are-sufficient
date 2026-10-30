"""Save and load trained FedPBA allocators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from paper2.src.config import PAPER2_ROOT
from paper2.src.fedpba.allocator import FedPBAAllocator, train_allocator

MODELS_DIR = PAPER2_ROOT / "models"


def save_allocator(allocator: FedPBAAllocator, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": allocator.model_name,
        "feature_set": allocator.feature_set,
        "feature_cols": allocator.feature_cols,
        "model": allocator.model,
        "label_encoder": allocator.label_encoder,
        "cat_encoder": allocator.cat_encoder,
        "cat_cols": allocator.cat_cols,
    }
    joblib.dump(payload, path)
    meta = {
        "model_name": allocator.model_name,
        "feature_set": allocator.feature_set,
        "feature_cols": allocator.feature_cols,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_allocator(path: Path) -> FedPBAAllocator:
    payload = joblib.load(path)
    return FedPBAAllocator(
        model_name=payload["model_name"],
        feature_set=payload["feature_set"],
        feature_cols=payload["feature_cols"],
        model=payload["model"],
        label_encoder=payload["label_encoder"],
        cat_encoder=payload["cat_encoder"],
        cat_cols=payload["cat_cols"],
    )
