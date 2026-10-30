"""
Personalization level definitions (P0–P5).

Each level specifies which named model blocks are trainable during local
fine-tuning on top of a globally aggregated encoder.
"""

from __future__ import annotations

from typing import Dict, Set

import torch.nn as nn

# P0: global model only — nothing trainable locally
# P1: head only
# P2: adapter + head
# P3: last encoder block + adapter + head
# P4: last two encoder blocks + adapter + head
# P5: full local personalization (all blocks)

PERSONALIZATION_TRAINABLE_BLOCKS: Dict[str, Set[str]] = {
    "P0": set(),
    "P1": {"head"},
    "P2": {"adapter", "head"},
    "P3": {"encoder_block4", "adapter", "head"},
    "P4": {"encoder_block3", "encoder_block4", "adapter", "head"},
    "P5": {
        "stem",
        "encoder_block1",
        "encoder_block2",
        "encoder_block3",
        "encoder_block4",
        "adapter",
        "head",
    },
}


def apply_personalization_level(model: nn.Module, level: str) -> None:
    """
    Freeze/unfreeze model blocks according to personalization level.

    Parameters
    ----------
    model : must expose get_blocks() returning name -> submodule mapping
    level : one of P0..P5
    """
    if level not in PERSONALIZATION_TRAINABLE_BLOCKS:
        raise ValueError(f"Unknown personalization level: {level}")

    trainable = PERSONALIZATION_TRAINABLE_BLOCKS[level]
    blocks = model.get_blocks()

    for name, module in blocks.items():
        requires_grad = name in trainable
        for param in module.parameters():
            param.requires_grad = requires_grad


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def communication_bytes(model: nn.Module) -> int:
    """
    Communication size proxy: bytes for trainable (personalized) parameters.

    Assumes float32 weights uploaded/downloaded during personalization.
    """
    return count_trainable_params(model) * 4
