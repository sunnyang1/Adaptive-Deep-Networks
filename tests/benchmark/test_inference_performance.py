"""
Performance benchmarks for inference optimization.

Measures actual computational savings from Ponder Gate and adaptive features.
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.configs import get_config
from src.models.adaptive_transformer import AdaptiveTransformer


def benchmark_ponder_gate_savings():
    """Benchmark how much compute Ponder Gate saves."""
    print("=" * 70)
    print("BENCHMARK: Ponder Gate Compute Savings")
    print("=" * 70)

    config = get_config("small")
    model = AdaptiveTransformer(config)
    model.eval()

    input_ids = torch.randint(0, 32000, (1, 50))
    max_new_tokens = 10

    print("\n1. Baseline (no qTTT)...")
    start = time.time()
    model.generate(input_ids, max_new_tokens=max_new_tokens, use_qttt=False)
    time_baseline = time.time() - start
    print(f"   Time: {time_baseline:.3f}s")

    print("\n2. Unconditional qTTT...")
    start = time.time()
    model.generate(
        input_ids, max_new_tokens=max_new_tokens, use_qttt=True, qttt_config={"num_steps": 2}
    )
    time_uncond = time.time() - start
    print(f"   Time: {time_uncond:.3f}s")

    for mode in ["strict", "balanced", "lenient"]:
        print(f"\n3. Adaptive qTTT ({mode} mode)...")
        start = time.time()
        model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            use_qttt="adaptive",
            ponder_gate_mode=mode,
            qttt_config={"num_steps": 2},
        )
        time_adaptive = time.time() - start
        savings = (time_uncond - time_adaptive) / time_uncond * 100
        print(f"   Time: {time_adaptive:.3f}s (saved ~{savings:.0f}%)")


def benchmark_adaptive_config_scaling():
    """Benchmark adaptive config scaling with sequence length."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Adaptive Config Sequence Length Scaling")
    print("=" * 70)

    from src.qttt.adaptive_config import create_adaptive_config

    cfg = create_adaptive_config("balanced")
    seq_lengths = [32, 128, 256, 512, 1024, 2048]

    print("\nSequence Length -> qTTT Steps:")
    for seq_len in seq_lengths:
        steps = cfg.get_steps_for_seq_len(seq_len)
        print(f"  {seq_len:4d} tokens -> {steps:2d} steps")


if __name__ == "__main__":
    benchmark_adaptive_config_scaling()
    benchmark_ponder_gate_savings()
    print("\n" + "=" * 70)
    print("BENCHMARKS COMPLETE")
    print("=" * 70)
