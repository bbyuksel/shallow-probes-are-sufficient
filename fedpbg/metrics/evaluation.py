"""
Classification metrics for client-level evaluation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute Accuracy, Macro-F1, and AUROC on a dataloader."""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    for x, y in dataloader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_labels.extend(y.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs)

    labels = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }
    try:
        if len(np.unique(labels)) > 1:
            metrics["auroc"] = float(roc_auc_score(labels, probs[:, 1]))
        else:
            metrics["auroc"] = float("nan")
    except ValueError:
        metrics["auroc"] = float("nan")

    return metrics
