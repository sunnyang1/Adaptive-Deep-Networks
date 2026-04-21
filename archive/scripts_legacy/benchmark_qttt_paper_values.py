"""
Test qTTT with Paper Values vs Optimized Values

Compares:
- Paper values: num_steps=10, learning_rate=0.01, early_stop=None
- Optimized values: num_steps=2, learning_rate=0.02, early_stop=0.001
"""

import torch
import time
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.configs import ModelConfig
from src.models.adaptive_transformer import AdaptiveTransformer


def test_paper_vs_optimized():
    """Compare paper values vs optimized values."""
    print("=" * 70)
    print("qTTT PAPER VALUES vs OPTIMIZED VALUES")
    print("=" * 70)

    device = "cpu"

    config = ModelConfig(
        num_layers=8, hidden_dim=512, num_heads=8, num_blocks=4, vocab_size=1000, mlp_ratio=4
    )

    model = AdaptiveTransformer(config).to(device)
    model.eval()

    # Test input
    torch.manual_seed(42)
    prompt_len = 32
    max_new_tokens = 10
    input_ids = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

    print(f"\nPrompt: {prompt_len} tokens")
    print(f"Generating: {max_new_tokens} new tokens")
    print(f"Model: {config.num_layers}L x {config.hidden_dim}D")

    results = {}

    # 1. Baseline (no qTTT)
    print("\n" + "-" * 70)
    print("[1] BASELINE (no qTTT)")
    print("-" * 70)
    t0 = time.time()
    with torch.no_grad():
        output_baseline = model.generate(
            input_ids, max_new_tokens=max_new_tokens, use_qttt=False, use_attnres=True
        )
    t_baseline = time.time() - t0

    print(f"  Time: {t_baseline:.3f}s")
    print(f"  Tokens/sec: {max_new_tokens/t_baseline:.1f}")
    results["baseline"] = {"time": t_baseline, "tokens_per_sec": max_new_tokens / t_baseline}

    # 2. Paper values (10 steps, lr=0.01)
    print("\n" + "-" * 70)
    print("[2] PAPER VALUES (num_steps=10, lr=0.01)")
    print("-" * 70)
    paper_config = {
        "num_steps": 10,
        "learning_rate": 0.01,
        "early_stop_threshold": None,  # Disabled
        "span_length": 128,
    }

    t0 = time.time()
    with torch.no_grad():
        output_paper = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            use_qttt=True,
            qttt_config=paper_config,
            use_attnres=True,
        )
    t_paper = time.time() - t0

    print(f"  Time: {t_paper:.3f}s")
    print(f"  Overhead: {t_paper/t_baseline:.1f}x baseline")
    print(f"  Tokens/sec: {max_new_tokens/t_paper:.1f}")

    # Compare output quality
    match_count = sum(
        a == b
        for a, b in zip(
            output_baseline[0, prompt_len:].tolist(), output_paper[0, prompt_len:].tolist()
        )
    )
    print(f"  Token match with baseline: {match_count}/{max_new_tokens}")

    results["paper"] = {
        "time": t_paper,
        "overhead": t_paper / t_baseline,
        "tokens_per_sec": max_new_tokens / t_paper,
        "token_match": match_count,
        "config": paper_config,
    }

    # 3. Optimized values (2 steps, lr=0.02)
    print("\n" + "-" * 70)
    print("[3] OPTIMIZED VALUES (num_steps=2, lr=0.02)")
    print("-" * 70)
    optimized_config = {
        "num_steps": 2,
        "learning_rate": 0.02,
        "early_stop_threshold": 0.001,
        "span_length": 128,
    }

    t0 = time.time()
    with torch.no_grad():
        output_optimized = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            use_qttt=True,
            qttt_config=optimized_config,
            use_attnres=True,
        )
    t_optimized = time.time() - t0

    print(f"  Time: {t_optimized:.3f}s")
    print(f"  Overhead: {t_optimized/t_baseline:.1f}x baseline")
    print(f"  Tokens/sec: {max_new_tokens/t_optimized:.1f}")

    # Compare output quality
    match_count_opt = sum(
        a == b
        for a, b in zip(
            output_baseline[0, prompt_len:].tolist(), output_optimized[0, prompt_len:].tolist()
        )
    )
    print(f"  Token match with baseline: {match_count_opt}/{max_new_tokens}")

    results["optimized"] = {
        "time": t_optimized,
        "overhead": t_optimized / t_baseline,
        "tokens_per_sec": max_new_tokens / t_optimized,
        "token_match": match_count_opt,
        "config": optimized_config,
    }

    # 4. Paper short sequence values (2 steps, lr=0.01) - as mentioned in §3.3.4
    print("\n" + "-" * 70)
    print("[4] PAPER SHORT SEQ (num_steps=2, lr=0.01)")
    print("-" * 70)
    short_config = {
        "num_steps": 2,
        "learning_rate": 0.01,
        "early_stop_threshold": None,
        "span_length": 128,
    }

    t0 = time.time()
    with torch.no_grad():
        output_short = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            use_qttt=True,
            qttt_config=short_config,
            use_attnres=True,
        )
    t_short = time.time() - t0

    print(f"  Time: {t_short:.3f}s")
    print(f"  Overhead: {t_short/t_baseline:.1f}x baseline")
    print(f"  Tokens/sec: {max_new_tokens/t_short:.1f}")

    match_count_short = sum(
        a == b
        for a, b in zip(
            output_baseline[0, prompt_len:].tolist(), output_short[0, prompt_len:].tolist()
        )
    )
    print(f"  Token match with baseline: {match_count_short}/{max_new_tokens}")

    results["paper_short"] = {
        "time": t_short,
        "overhead": t_short / t_baseline,
        "tokens_per_sec": max_new_tokens / t_short,
        "token_match": match_count_short,
        "config": short_config,
    }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Configuration':<25} {'Time (s)':<12} {'Overhead':<12} {'Match':<10}")
    print("-" * 60)
    print(f"{'Baseline (no qTTT)':<25} {t_baseline:<12.3f} {'1.0x':<12} {'-':<10}")
    print(
        f"{'Paper (10 steps, 0.01)':<25} {t_paper:<12.3f} {t_paper/t_baseline:<12.1f} {match_count}/{max_new_tokens}"
    )
    print(
        f"{'Optimized (2 steps, 0.02)':<25} {t_optimized:<12.3f} {t_optimized/t_baseline:<12.1f} {match_count_opt}/{max_new_tokens}"
    )
    print(
        f"{'Paper Short (2 steps, 0.01)':<25} {t_short:<12.3f} {t_short/t_baseline:<12.1f} {match_count_short}/{max_new_tokens}"
    )

    # Speedup calculation
    print(f"\nSpeedup: Optimized is {t_paper/t_optimized:.1f}x faster than Paper (10 steps)")

    # Save results
    output_path = Path("results/qttt_paper_vs_optimized.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def test_adaptive_lr_by_length():
    """Test adaptive learning rate based on sequence length (as per paper §3.3.4)."""
    print("\n" + "=" * 70)
    print("ADAPTIVE LEARNING RATE BY SEQUENCE LENGTH")
    print("=" * 70)

    device = "cpu"

    config = ModelConfig(
        num_layers=8,
        hidden_dim=512,
        num_heads=8,
        num_blocks=4,
        vocab_size=1000,
    )

    model = AdaptiveTransformer(config).to(device)
    model.eval()

    # Test different sequence lengths
    test_configs = [
        ("Short (<4K)", 64, 0.01),
        ("Medium (4K-32K)", 128, 0.005),
        ("Long (>32K)", 256, 0.002),
    ]

    results = []

    for name, seq_len, lr in test_configs:
        print(f"\n{name}: seq_len={seq_len}, lr={lr}")

        torch.manual_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)

        # Baseline
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=5, use_qttt=False)
        t_base = time.time() - t0

        # With qTTT (paper values)
        qttt_cfg = {"num_steps": 10, "learning_rate": lr, "early_stop_threshold": None}
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=5, use_qttt=True, qttt_config=qttt_cfg)
        t_qttt = time.time() - t0

        print(f"  Baseline: {t_base:.3f}s, qTTT: {t_qttt:.3f}s, Overhead: {t_qttt/t_base:.1f}x")

        results.append({"name": name, "seq_len": seq_len, "lr": lr, "overhead": t_qttt / t_base})

    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("qTTT PAPER VALUES BENCHMARK")
    print("Testing with paper defaults: num_steps=10, learning_rate=0.01")
    print("=" * 70)

    # Main comparison
    results = test_paper_vs_optimized()

    # Adaptive LR test
    adaptive_results = test_adaptive_lr_by_length()

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
