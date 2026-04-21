"""
Quick Test qTTT with Paper Values vs Optimized Values (Lightweight)
"""

import torch
import time
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.configs import ModelConfig
from src.models.adaptive_transformer import AdaptiveTransformer


def test_quick_comparison():
    """Quick comparison with small model."""
    print("=" * 60)
    print("qTTT PAPER vs OPTIMIZED (Quick Test)")
    print("=" * 60)

    device = "cpu"

    # Small model for fast testing
    config = ModelConfig(
        num_layers=4, hidden_dim=256, num_heads=4, num_blocks=2, vocab_size=500, mlp_ratio=2
    )

    model = AdaptiveTransformer(config).to(device)
    model.eval()

    torch.manual_seed(42)
    prompt_len = 16
    max_new_tokens = 5
    input_ids = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

    print(f"\nConfig: {config.num_layers}L x {config.hidden_dim}D, {max_new_tokens} tokens")

    results = {}

    # 1. Baseline
    print("\n[1] Baseline (no qTTT)...", end=" ")
    t0 = time.time()
    with torch.no_grad():
        output_baseline = model.generate(
            input_ids, max_new_tokens=max_new_tokens, use_qttt=False, use_attnres=True
        )
    t_baseline = time.time() - t0
    print(f"{t_baseline:.3f}s")
    results["baseline"] = {"time": t_baseline}

    # 2. Paper values (10 steps, lr=0.01)
    print("[2] Paper (10 steps, lr=0.01)...", end=" ")
    paper_config = {"num_steps": 10, "learning_rate": 0.01, "early_stop_threshold": None}
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
    print(f"{t_paper:.3f}s ({t_paper/t_baseline:.1f}x)")
    results["paper"] = {"time": t_paper, "overhead": t_paper / t_baseline}

    # 3. Optimized (2 steps, lr=0.02)
    print("[3] Optimized (2 steps, lr=0.02)...", end=" ")
    opt_config = {"num_steps": 2, "learning_rate": 0.02, "early_stop_threshold": 0.001}
    t0 = time.time()
    with torch.no_grad():
        output_opt = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            use_qttt=True,
            qttt_config=opt_config,
            use_attnres=True,
        )
    t_opt = time.time() - t0
    print(f"{t_opt:.3f}s ({t_opt/t_baseline:.1f}x)")
    results["optimized"] = {"time": t_opt, "overhead": t_opt / t_baseline}

    # 4. Paper short (2 steps, lr=0.01)
    print("[4] Paper Short (2 steps, lr=0.01)...", end=" ")
    short_config = {"num_steps": 2, "learning_rate": 0.01, "early_stop_threshold": None}
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
    print(f"{t_short:.3f}s ({t_short/t_baseline:.1f}x)")
    results["paper_short"] = {"time": t_short, "overhead": t_short / t_baseline}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<30} {'Time':<10} {'Overhead':<10}")
    print("-" * 50)
    print(f"{'Baseline (no qTTT)':<30} {t_baseline:<10.3f} {'1.0x':<10}")
    print(f"{'Paper (10 steps, lr=0.01)':<30} {t_paper:<10.3f} {t_paper/t_baseline:<10.1f}")
    print(f"{'Optimized (2 steps, lr=0.02)':<30} {t_opt:<10.3f} {t_opt/t_baseline:<10.1f}")
    print(f"{'Paper Short (2 steps, lr=0.01)':<30} {t_short:<10.3f} {t_short/t_baseline:<10.1f}")

    print(f"\nOptimized is {t_paper/t_opt:.1f}x faster than Paper (10 steps)")
    print(f"Paper Short is {t_paper/t_short:.1f}x faster than Paper (10 steps)")

    # Save
    output_path = Path("results/qttt_paper_values_quick.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = test_quick_comparison()
    print("\n✓ Test complete!")
