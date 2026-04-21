"""
Common Utilities for Scripts

Shared infrastructure for training, evaluation, and setup scripts.
"""

from .paths import add_project_to_path, get_default_paths
from .distributed import setup_distributed, cleanup_distributed, is_main_process
from .training import CheckpointManager, compute_loss, train_step
from .data import DummyDataset, get_dataloader

__all__ = [
    "add_project_to_path",
    "get_default_paths",
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "CheckpointManager",
    "compute_loss",
    "train_step",
    "DummyDataset",
    "get_dataloader",
]
