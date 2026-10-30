"""
Cost proxy computation and normalization for utility-based oracle selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from fedpbg.hardware.profiles import HardwareProfile, hardware_cost_multiplier


@dataclass
class CostRecord:
    """Raw and normalized cost components for one (client, level) run."""

    trainable_params: int
    flops: int
    train_time_sec: float
    comm_bytes: int
    memory_proxy_mb: float
    hardware_multiplier: float
    raw_cost: float
    normalized_cost: float


def memory_proxy_mb(trainable_params: int, batch_size: int, signal_length: int) -> float:
    """
    Simple memory proxy: model params + activation buffer estimate.

    Not a precise profiler reading — sufficient for feasibility ranking.
    """
    param_mb = trainable_params * 4 / (1024 ** 2)
    activation_mb = batch_size * signal_length * 4 / (1024 ** 2) * 4  # rough depth factor
    return param_mb + activation_mb


def compute_raw_cost(
    trainable_params: int,
    flops: int,
    train_time_sec: float,
    comm_bytes: int,
    memory_mb: float,
    hardware: HardwareProfile,
) -> float:
    """Weighted sum of cost proxies scaled by hardware capability."""
    hw_mult = hardware_cost_multiplier(hardware)
    # Normalize magnitudes to comparable scales
    param_term = trainable_params / 1e5
    flop_term = flops / 1e8
    time_term = train_time_sec
    comm_term = comm_bytes / 1e6
    mem_term = memory_mb / 100.0

    raw = (
        0.25 * param_term
        + 0.20 * flop_term
        + 0.25 * time_term
        + 0.15 * comm_term
        + 0.15 * mem_term
    )
    return raw * hw_mult


def normalize_costs_across_levels(raw_costs: List[float]) -> List[float]:
    """Min-max normalize costs within one client across personalization levels."""
    arr = np.array(raw_costs, dtype=np.float64)
    c_min, c_max = arr.min(), arr.max()
    if c_max - c_min < 1e-12:
        return [0.0] * len(raw_costs)
    return ((arr - c_min) / (c_max - c_min)).tolist()


def build_cost_record(
    trainable_params: int,
    flops: int,
    train_time_sec: float,
    comm_bytes: int,
    batch_size: int,
    signal_length: int,
    hardware: HardwareProfile,
    normalized_cost: float,
) -> CostRecord:
    mem = memory_proxy_mb(trainable_params, batch_size, signal_length)
    raw = compute_raw_cost(
        trainable_params, flops, train_time_sec, comm_bytes, mem, hardware
    )
    return CostRecord(
        trainable_params=trainable_params,
        flops=flops,
        train_time_sec=train_time_sec,
        comm_bytes=comm_bytes,
        memory_proxy_mb=mem,
        hardware_multiplier=hardware_cost_multiplier(hardware),
        raw_cost=raw,
        normalized_cost=normalized_cost,
    )
