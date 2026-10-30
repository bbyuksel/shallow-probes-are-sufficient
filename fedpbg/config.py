"""
Central configuration for the FedPBG feasibility study.

All hyperparameters and paths are defined here so experiments remain
reproducible and easy to extend (e.g. 12-lead, multi-class, new alpha values).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# Project root (parent of fedpbg package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "ptbxl"

# Federated simulation
NUM_CLIENTS = 20
DIRICHLET_ALPHA = 0.1
DIRICHLET_ALPHA_OPTIONS: List[float] = [0.1, 0.3, 0.5, 1.0]

# Personalization levels P0–P5
PERSONALIZATION_LEVELS = ["P0", "P1", "P2", "P3", "P4", "P5"]

# Utility trade-off lambdas
LAMBDA_VALUES: List[float] = [0.1, 0.3, 0.5, 1.0]
DEFAULT_LAMBDA = 0.3

# ECG / model
LEAD_INDEX = 1  # Lead II in standard 12-lead ordering
SIGNAL_LENGTH = 1000  # samples after resampling (500 Hz -> 1000 = 2 s)
NUM_CLASSES = 2  # binary: Normal vs Abnormal
INPUT_CHANNELS = 1  # single-lead; set to 12 when extending to full leads

# Training
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FEDAVG_FRACTION = 0.5  # fraction of clients sampled per round

# Client train/val/test ratios (within each client's data)
CLIENT_TRAIN_RATIO = 0.7
CLIENT_VAL_RATIO = 0.15
# remainder is test

RANDOM_SEED = 42


@dataclass
class ExperimentConfig:
    """Runtime experiment settings (overridable via CLI)."""

    num_clients: int = NUM_CLIENTS
    dirichlet_alpha: float = DIRICHLET_ALPHA
    num_global_rounds: int = GLOBAL_ROUNDS
    local_epochs: int = LOCAL_EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    signal_length: int = SIGNAL_LENGTH
    lead_index: int = LEAD_INDEX
    num_classes: int = NUM_CLASSES
    input_channels: int = INPUT_CHANNELS
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)
    random_seed: int = RANDOM_SEED
    # Quick mode: synthetic data, fewer samples/epochs for smoke tests
    quick: bool = False
    max_records: Optional[int] = None  # cap PTB-XL records (None = all)
    device: str = "cpu"
