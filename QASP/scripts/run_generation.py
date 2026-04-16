"""Run a tiny QASP generation demo."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QASP.inference import QASPGenerator
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
    generator = QASPGenerator(model)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    output_ids = generator.generate(input_ids=input_ids, max_new_tokens=4)
    print(f"run_generation output shape: {tuple(output_ids.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
