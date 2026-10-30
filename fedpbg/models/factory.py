"""
Model factory — returns modality-appropriate encoders with unified block interface.
"""

from __future__ import annotations

import torch.nn as nn

from fedpbg.models.ecg_resnet1d import ECGResNet1D, estimate_forward_flops
from fedpbg.models.resnet2d import ImageResNet2D, estimate_forward_flops_2d


def build_model(
    modality: str,
    in_channels: int,
    num_classes: int,
    input_size: int,
) -> nn.Module:
    """Build encoder model for ECG (1D) or image (2D) datasets."""
    if modality == "ecg":
        return ECGResNet1D(in_channels=in_channels, num_classes=num_classes)
    if modality == "image":
        return ImageResNet2D(
            in_channels=in_channels,
            num_classes=num_classes,
            input_size=input_size,
        )
    raise ValueError(f"Unknown modality: {modality}")


def estimate_flops(model: nn.Module, modality: str, input_size: int, batch_size: int = 1) -> int:
    if modality == "ecg":
        return estimate_forward_flops(model, input_size, batch_size)
    if modality == "image":
        return estimate_forward_flops_2d(model, input_size, batch_size)
    raise ValueError(f"Unknown modality: {modality}")
