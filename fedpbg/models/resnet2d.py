"""
Lightweight 2D ResNet for image classification (CIFAR / FEMNIST).

Uses the same named-block interface as ECGResNet1D so P0–P5 personalization
applies identically across modalities.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2D(nn.Module):
    """Basic 2D residual block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ImageResNet2D(nn.Module):
    """ResNet-style 2D encoder + adapter + head."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_width: int = 32,
        input_size: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_size = input_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.encoder_block1 = ResBlock2D(base_width, base_width, stride=1)
        self.encoder_block2 = ResBlock2D(base_width, base_width * 2, stride=2)
        self.encoder_block3 = ResBlock2D(base_width * 2, base_width * 4, stride=2)
        self.encoder_block4 = ResBlock2D(base_width * 4, base_width * 4, stride=2)

        enc_dim = base_width * 4
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(enc_dim, enc_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(enc_dim // 2, enc_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(enc_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        x = self.encoder_block3(x)
        x = self.encoder_block4(x)
        return self.adapter(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

    def get_blocks(self) -> Dict[str, nn.Module]:
        return {
            "stem": self.stem,
            "encoder_block1": self.encoder_block1,
            "encoder_block2": self.encoder_block2,
            "encoder_block3": self.encoder_block3,
            "encoder_block4": self.encoder_block4,
            "adapter": self.adapter,
            "head": self.head,
        }


def estimate_forward_flops_2d(
    model: ImageResNet2D,
    input_size: int,
    batch_size: int = 1,
) -> int:
    """Rough FLOPs estimate for one 2D forward pass."""
    flops = 0
    h = w = input_size
    in_ch = model.in_channels
    channels = [32, 32, 64, 128, 128]

    # stem
    flops += 2 * in_ch * channels[0] * 9 * h * w * batch_size

    for c_in, c_out, stride in [(32, 32, 1), (32, 64, 2), (64, 128, 2), (128, 128, 2)]:
        h = max(1, (h + stride - 1) // stride if stride > 1 else h)
        w = max(1, (w + stride - 1) // stride if stride > 1 else w)
        flops += 2 * c_in * c_out * 9 * h * w * batch_size * 2

    enc_dim = 128
    flops += 2 * enc_dim * (enc_dim // 2) * batch_size
    flops += 2 * (enc_dim // 2) * enc_dim * batch_size
    flops += 2 * enc_dim * model.num_classes * batch_size
    return int(flops)
