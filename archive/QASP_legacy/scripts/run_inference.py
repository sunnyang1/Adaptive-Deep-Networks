"""Run a tiny QASP incremental prefill+step demo."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QASP.inference import IncrementalInference
from QASP.models import create_qasp_transformer


def main() -> int:
    torch.manual_seed(0)
    model = create_qasp_transformer(
        vocab_size=128,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=128,
    )

    incremental = IncrementalInference(model)
    state = incremental.prefill(torch.tensor([[4, 5, 6]], dtype=torch.long))
    first_token = incremental.step(state)
    second_token = incremental.step(state)

    print(f"run_inference first token shape: {tuple(first_token.shape)}")
    print(f"run_inference second token shape: {tuple(second_token.shape)}")
    print(f"run_inference sequence length: {state.seq_len}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
