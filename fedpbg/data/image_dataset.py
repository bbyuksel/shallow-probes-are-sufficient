"""
Generic image tensor dataset wrapper.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ImageTensorDataset(Dataset):
    """PyTorch dataset from preloaded (N, C, H, W) arrays."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]
