"""Smoke test for QASP training script --quick mode."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_train_quick_mode_completes() -> None:
    """The --quick training mode should run to completion without error."""

    from QASP.scripts.train import train
    import argparse

    args = argparse.Namespace(
        dataset_name="togethercomputer/RedPajama-Data-1T-Sample",
        dataset_config=None,
        text_column="text",
        preset=None,
        vocab_size=128,
        hidden_size=64,
        num_heads=4,
        num_key_value_heads=None,
        num_layers=2,
        max_seq_len=64,
        use_attnres=False,
        use_engram=False,
        use_rope=False,
        num_epochs=1,
        batch_size=2,
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=2,
        max_steps=5,
        gradient_clipping=1.0,
        mixed_precision="no",
        output_dir="/tmp/qasp_test_quick",
        log_every=1,
        save_every=10,
        device="cpu",
        quick=True,
    )

    train(args)
