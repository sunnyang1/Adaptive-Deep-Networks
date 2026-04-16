"""Smoke tests for lightweight QASP generation and incremental APIs."""

from __future__ import annotations

import torch

from QASP.inference import IncrementalInference, QASPGenerator
from QASP.models import create_qasp_transformer


def _build_tiny_model():
    model = create_qasp_transformer(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=64,
    )
    model.eval()
    return model


def test_generate_increases_output_length_by_max_new_tokens():
    model = _build_tiny_model()
    generator = QASPGenerator(model)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    max_new_tokens = 5

    output_ids = generator.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)

    assert output_ids.shape[1] == input_ids.shape[1] + max_new_tokens


def test_incremental_step_returns_batch_single_token_shape_after_prefill():
    model = _build_tiny_model()
    inference = IncrementalInference(model)
    input_ids = torch.tensor([[7, 8, 9]], dtype=torch.long)

    state = inference.prefill(input_ids)
    next_token = inference.step(state)

    assert next_token.shape == (input_ids.shape[0], 1)
