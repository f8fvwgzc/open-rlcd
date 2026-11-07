"""The tabular pack: generic CSV / Parquet tables (churn, fraud, lead scoring, approvals).

One calibrated head per label column; every head answers on one table row sent as `state = {column: value}`.
See pack.py (the pack), data.py (pack_config and reading the file), provider.py (the feature schema) and
synth.py (synthetic data with known probabilities, for tests and demos).
"""
from .data import TabularConfig, TabularConfigError, read_table
from .pack import TabularPack
from .provider import TabularProvider
from .synth import generate, synthetic_pack_config, true_probabilities, write_synthetic_csv

__all__ = ["TabularPack", "TabularProvider", "TabularConfig", "TabularConfigError", "read_table",
           "write_synthetic_csv", "synthetic_pack_config", "generate", "true_probabilities"]
