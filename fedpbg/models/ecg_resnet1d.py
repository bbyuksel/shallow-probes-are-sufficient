"""
Lightweight 1D ResNet for ECG classification.

Named submodules (stem, encoder blocks, adapter, head) allow personalization
levels P0–P5 to freeze/unfreeze specific components without duplicating code.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """Basic 1D residual block with optional channel projection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class ECGResNet1D(nn.Module):
    """
    ResNet-style 1D encoder + bottleneck adapter + linear head.

    Block naming convention:
      - stem
      - encoder_block1 .. encoder_block4
      - adapter
      - head
    """

    BLOCK_NAMES = [
        "stem",
        "encoder_block1",
        "encoder_block2",
        "encoder_block3",
        "encoder_block4",
        "adapter",
        "head",
    ]

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_width: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_width, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(inplace=True),
        )
        self.encoder_block1 = ResBlock1D(base_width, base_width, stride=1)
        self.encoder_block2 = ResBlock1D(base_width, base_width * 2, stride=2)
        self.encoder_block3 = ResBlock1D(base_width * 2, base_width * 4, stride=2)
        self.encoder_block4 = ResBlock1D(base_width * 4, base_width * 4, stride=2)

        enc_dim = base_width * 4
        # Lightweight bottleneck adapter (personalization-friendly)
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(enc_dim, enc_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(enc_dim // 2, enc_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(enc_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding before the classification head (after adapter)."""
        x = self.stem(x)
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        x = self.encoder_block3(x)
        x = self.encoder_block4(x)
        return self.adapter(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encode(x)
        return self.head(emb)

    def get_blocks(self) -> Dict[str, nn.Module]:
        """Return named blocks for personalization control."""
        return {
            "stem": self.stem,
            "encoder_block1": self.encoder_block1,
            "encoder_block2": self.encoder_block2,
            "encoder_block3": self.encoder_block3,
            "encoder_block4": self.encoder_block4,
            "adapter": self.adapter,
            "head": self.head,
        }

    def trainable_parameter_groups(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def estimate_forward_flops(model: ECGResNet1D, seq_length: int, batch_size: int = 1) -> int:
    """
    Rough FLOPs estimate for one forward pass (multiply-adds counted as 2 FLOPs).

    Uses module parameter shapes rather than torch profiler for portability.
    """
    flops = 0
    # Conv layers: 2 * Cin * Cout * K * L_out per layer (approximate L_out)
    length = seq_length
    in_ch = model.in_channels
    channels = [32, 32, 64, 128, 128]

    # stem conv
    length = length // 2
    flops += 2 * in_ch * channels[0] * 15 * length * batch_size

    for i, (c_in, c_out, stride) in enumerate(
        [(32, 32, 1), (32, 64, 2), (64, 128, 2), (128, 128, 2)]
    ):
        length = max(1, length // stride)
        flops += 2 * c_in * c_out * 7 * length * batch_size * 2  # two convs per block

    enc_dim = 128
    # adapter MLP
    flops += 2 * enc_dim * (enc_dim // 2) * batch_size
    flops += 2 * (enc_dim // 2) * enc_dim * batch_size
    # head
    flops += 2 * enc_dim * model.num_classes * batch_size
    return int(flops)
