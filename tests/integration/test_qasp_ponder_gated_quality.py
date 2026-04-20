"""Tests for ponder-gated quality computation in step()."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QASP.models import create_qasp_transformer


def test_step_skips_quality_when_gate_blocks() -> None:
    """When gate_quality_computation=True and gate is off, step should skip FFT."""

    model = create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=32,
        use_attnres=True,
        use_engram=False,
        gate_quality_computation=True,
    )
    model.eval()

    input_ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        logits, cache = model.prefill(input_ids)

    # After prefill, cache should have per_token_quality stored
    assert cache.per_token_quality is not None

    # Mock compute_quality_score to count calls
    call_count = 0
    original_compute = None

    def counting_compute(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_compute(*args, **kwargs)

    import QASP.models.components as comp
    original_compute = comp.compute_quality_score

    with patch.object(comp, "compute_quality_score", side_effect=counting_compute):
        with torch.no_grad():
            # Force gate to NOT fire by setting last_logits to a confident distribution
            confident_logits = torch.zeros(1, 64)
            confident_logits[0, 0] = 100.0  # very confident
            cache.last_logits = confident_logits

            _ = model.step(input_ids[:, -1:], cache)

    # With gate off, compute_quality_score should NOT be called in step()
    assert call_count == 0, f"Expected 0 quality computations when gate blocks, got {call_count}"


def test_step_computes_quality_when_gate_fires() -> None:
    """When gate_quality_computation=True and gate fires, step should compute quality."""

    model = create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=32,
        use_attnres=True,
        use_engram=False,
        gate_quality_computation=True,
    )
    model.eval()

    input_ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        logits, cache = model.prefill(input_ids)

    call_count = 0
    original_compute = None

    def counting_compute(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_compute(*args, **kwargs)

    import QASP.models.components as comp
    original_compute = comp.compute_quality_score

    with patch.object(comp, "compute_quality_score", side_effect=counting_compute):
        with torch.no_grad():
            # Force gate to fire by setting last_logits to uniform (high entropy)
            uniform_logits = torch.zeros(1, 64)
            cache.last_logits = uniform_logits

            _ = model.step(input_ids[:, -1:], cache)

    # With gate on, compute_quality_score should be called at least once
    assert call_count > 0, f"Expected >0 quality computations when gate fires, got {call_count}"
