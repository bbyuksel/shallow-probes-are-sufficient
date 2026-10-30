"""Paper-2 Phase-0 configuration."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER2_ROOT = PROJECT_ROOT / "paper2"
RESULTS_DIR = PAPER2_ROOT / "results"
FIGURES_DIR = PAPER2_ROOT / "figures"
TABLES_DIR = PAPER2_ROOT / "tables"
REPORTS_DIR = PAPER2_ROOT / "reports"

CACHE_DIRS = [
    PROJECT_ROOT / "results" / "sparsity_phase_diagram" / "cache",
    PROJECT_ROOT / "results" / "oracle_sparsity_validation" / "cache",
]

LAMBDA_VALUES = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
LEVELS = ["P0", "P1", "P2", "P3", "P4", "P5"]
LEVEL_TO_IDX = {lv: i for i, lv in enumerate(LEVELS)}

# Features allowed in cheap (deployable) dataset
CHEAP_FEATURE_COLS = [
    "dataset",
    "alpha",
    "lambda",
    "num_clients",
    "hardware_profile",
    "hardware_score",
    "hardware_multiplier",
    "heterogeneity",
    "num_samples",
    "label_entropy",
    "majority_class_ratio",
    "class_count",
    "class_balance_score",
    "cost_p0",
    "cost_p1",
    "cost_p2",
    "p0_macro_f1",
    "p1_macro_f1",
    "p2_macro_f1",
    "gradient_norm",
    "gradient_variance",
    "cosine_to_global_update",
    "representation_distance",
    "prototype_distance",
    "client_drift_score",
]

# Expensive / oracle-only columns (excluded from cheap features)
EXPENSIVE_FEATURE_COLS = [
    "utility_p3",
    "utility_p4",
    "utility_p5",
    "p3_macro_f1",
    "p4_macro_f1",
    "p5_macro_f1",
    "oracle_utility",
]
