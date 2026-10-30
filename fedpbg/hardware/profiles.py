"""
Hardware profile simulation for federated clients.

Each profile defines proxy values for compute, memory, bandwidth, latency, and
energy. These scale raw training costs so oracle budget selection reflects
device constraints, not just model accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np


class HardwareType(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class HardwareProfile:
    """Normalized hardware capability proxies in [0, 1] (higher = better)."""

    name: HardwareType
    compute: float  # relative FLOPs/s
    memory: float  # relative RAM (GB proxy)
    bandwidth: float  # relative network Mbps
    latency: float  # inverse latency proxy (higher = lower latency)
    energy: float  # inverse energy per op (higher = more efficient)


HARDWARE_PROFILES: Dict[HardwareType, HardwareProfile] = {
    HardwareType.LOW: HardwareProfile(
        name=HardwareType.LOW,
        compute=0.25,
        memory=0.20,
        bandwidth=0.15,
        latency=0.30,
        energy=0.35,
    ),
    HardwareType.MEDIUM: HardwareProfile(
        name=HardwareType.MEDIUM,
        compute=0.55,
        memory=0.50,
        bandwidth=0.50,
        latency=0.55,
        energy=0.60,
    ),
    HardwareType.HIGH: HardwareProfile(
        name=HardwareType.HIGH,
        compute=1.00,
        memory=1.00,
        bandwidth=1.00,
        latency=1.00,
        energy=1.00,
    ),
}


def assign_hardware_profiles(num_clients: int, random_seed: int = 42) -> List[HardwareType]:
    """
    Assign hardware types to clients with a realistic skew (more LOW than HIGH).

    Distribution: 40% LOW, 40% MEDIUM, 20% HIGH
    """
    rng = np.random.default_rng(random_seed)
    probs = [0.4, 0.4, 0.2]
    types = [HardwareType.LOW, HardwareType.MEDIUM, HardwareType.HIGH]
    return [types[i] for i in rng.choice(len(types), size=num_clients, p=probs)]


def hardware_cost_multiplier(profile: HardwareProfile) -> float:
    """
    Combine hardware proxies into a single cost multiplier.

    Lower-end devices incur higher effective cost (slower compute, limited memory/bandwidth).
    """
    # Weights reflect relative impact on federated personalization cost
    weights = {
        "compute": 0.35,
        "memory": 0.20,
        "bandwidth": 0.20,
        "latency": 0.15,
        "energy": 0.10,
    }
    capability = (
        weights["compute"] * profile.compute
        + weights["memory"] * profile.memory
        + weights["bandwidth"] * profile.bandwidth
        + weights["latency"] * profile.latency
        + weights["energy"] * profile.energy
    )
    # Invert capability: low-end hardware => higher cost
    return float(1.0 / max(capability, 0.05))
