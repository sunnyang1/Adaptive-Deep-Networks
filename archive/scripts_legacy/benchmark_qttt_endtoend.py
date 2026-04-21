"""
End-to-End qTTT (Query-only Test-Time Training) Benchmark

Tests qTTT in the complete generation pipeline with AdaptiveTransformer.
"""

import torch
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.configs import ModelConfig
from src.models.adaptive_transformer import AdaptiveTransformer


def test_qttt_basic_generation():
    """Test basic generation with qTTT enabled."""
    print("=" * 70)
    print("qTTT BASIC GENERATION TEST")
    print("=" * 70)

    device = "cpu"

    # Small model for testing
    config = ModelConfig(
        num_layers=4, hidden_dim=256, num_heads=4, num_blocks=2, vocab_size=500, mlp_ratio=4
    )

    model = AdaptiveTransformer(config).to(device)
    model.eval()

    # Test input
    prompt_len = 16
    max_new_tokens = 8
    input_ids = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

    print(f"\nPrompt: {prompt_len} tokens")
    print(f"Generating: {max_new_tokens} new tokens")

    # Baseline without qTTT
    print("\n[1] Baseline generation (no qTTT)...")
    t0 = time.time()
    with torch.no_grad():
        output_baseline = model.generate(
            input_ids, max_new_tokens=max_new_tokens, use_qttt=False, use_attnres=True
        )
    t_baseline = time.time() - t0

    print(f"  Output length: {output_baseline.shape[1]}")
    print(f"  Time: {t_baseline:.2f}s")
    print(f"  Generated tokens: {output_baseline[0, prompt_len:].tolist()}")

    # With qTTT (minimal steps for speed)
    print("\n[2] Generation with qTTT (4 steps)...")
    qttt_config = {
        "num_steps": 4,
        "learning_rate": 0.01,
        "span_length": 64,
        "margin_temperature": 1.0,
    }

    t0 = time.time()
    with torch.no_grad():
        output_qttt = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            use_qttt=True,
            qttt_config=qttt_config,
            use_attnres=True,
        )
    t_qttt = time.time() - t0

    print(f"  Output length: {output_qttt.shape[1]}")
    print(f"  Time: {t_qttt:.2f}s ({t_qttt/t_baseline:.1f}x baseline)")
    print(f"  Generated tokens: {output_qttt[0, prompt_len:].tolist()}")

    # Check if outputs are different (qTTT should affect generation)
    match = (output_baseline[0, prompt_len:] == output_qttt[0, prompt_len:]).all().item()
    print(f"\n  Tokens match baseline: {match}")

    if not match:
        diff_positions = (output_baseline[0, prompt_len:] != output_qttt[0, prompt_len:]).nonzero(
            as_tuple=True
        )[0]
        print(f"  Different at positions: {diff_positions.tolist()}")

    return {
        "baseline_time": t_baseline,
        "qttt_time": t_qttt,
        "tokens_match": match,
        "baseline_output": output_baseline[0, prompt_len:].tolist(),
        "qttt_output": output_qttt[0, prompt_len:].tolist(),
    }


def test_qttt_quality_comparison():
    """Compare generation quality with and without qTTT."""
    print("\n" + "=" * 70)
    print("qTTT QUALITY COMPARISON")
    print("=" * 70)

    device = "cpu"

    config = ModelConfig(
        num_layers=4,
        hidden_dim=256,
        num_heads=4,
        num_blocks=2,
        vocab_size=500,
    )

    model = AdaptiveTransformer(config).to(device)
    model.eval()

    # Test with a few different prompts
    torch.manual_seed(42)
    num_tests = 3
    prompt_len = 20
    max_new_tokens = 10

    results = []

    for i in range(num_tests):
        input_ids = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

        # Baseline
        with torch.no_grad():
            output_base = model.generate(input_ids, max_new_tokens=max_new_tokens, use_qttt=False)

        # With qTTT (more steps for better quality)
        with torch.no_grad():
            output_qttt = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                use_qttt=True,
                qttt_config={"num_steps": 8, "learning_rate": 0.01},
            )

        base_tokens = output_base[0, prompt_len:].tolist()
        qttt_tokens = output_qttt[0, prompt_len:].tolist()

        match_count = sum(a == b for a, b in zip(base_tokens, qttt_tokens))

        print(f"\nTest {i+1}:")
        print(f"  Baseline: {base_tokens}")
        print(f"  qTTT:     {qttt_tokens}")
        print(f"  Match:    {match_count}/{max_new_tokens} tokens")

        results.append(
            {
                "match_count": match_count,
                "total_tokens": max_new_tokens,
                "match_ratio": match_count / max_new_tokens,
            }
        )

    avg_match = sum(r["match_ratio"] for r in results) / len(results)
    print(f"\n  Average token match: {avg_match:.1%}")

    return results


def test_qttt_with_rabitq():
    """Test qTTT + RaBitQ combined."""
    print("\n" + "=" * 70)
    print("qTTT + RABITQ COMBINED TEST")
    print("=" * 70)

    device = "cpu"

    config = ModelConfig(
        num_layers=4,
        hidden_dim=256,
        num_heads=4,
        num_blocks=2,
        vocab_size=500,
    )

    model = AdaptiveTransformer(config).to(device)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 16), device=device)

    # Initialize RaBitQ
    model.init_rabitq_caches(total_bits=1, residual_window=64)

    print("\nGenerating with qTTT + RaBitQ 1-bit...")
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=8,
            use_qttt=True,
            qttt_config={"num_steps": 4},
            use_rabitq=True,
        )
    t_total = time.time() - t0

    print(f"  Completed in {t_total:.2f}s")
    print(f"  Output: {output[0].tolist()}")

    return {"time": t_total, "success": True}


def test_qttt_different_steps():
    """Test qTTT with different adaptation step counts."""
    print("\n" + "=" * 70)
    print("qTTT STEP COUNT COMPARISON")
    print("=" * 70)

    device = "cpu"

    config = ModelConfig(
        num_layers=4,
        hidden_dim=256,
        num_heads=4,
        num_blocks=2,
        vocab_size=500,
    )

    model = AdaptiveTransformer(config).to(device)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 16), device=device)
    max_new_tokens = 5

    step_configs = [0, 2, 4, 8]  # 0 = no qTTT
    results = []

    for steps in step_configs:
        print(f"\nTesting with {steps} qTTT steps...")

        t0 = time.time()
        with torch.no_grad():
            if steps == 0:
                output = model.generate(input_ids, max_new_tokens=max_new_tokens, use_qttt=False)
            else:
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    use_qttt=True,
                    qttt_config={"num_steps": steps},
                )
        t_elapsed = time.time() - t0

        tokens = output[0, 16:].tolist()
        print(f"  Time: {t_elapsed:.2f}s, Tokens: {tokens}")

        results.append({"steps": steps, "time": t_elapsed, "tokens": tokens})

    # Compare outputs
    baseline_tokens = results[0]["tokens"]
    print(f"\nComparison to baseline (0 steps):")
    for r in results[1:]:
        match = sum(a == b for a, b in zip(baseline_tokens, r["tokens"]))
        print(f"  {r['steps']} steps: {match}/{max_new_tokens} match, {r['time']:.2f}s")

    return results


def main():
    print("\n" + "=" * 70)
    print("qTTT END-TO-END BENCHMARK")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: CPU")

    all_results = {}

    try:
        all_results["basic"] = test_qttt_basic_generation()
    except Exception as e:
        print(f"\nBasic generation test failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        all_results["quality"] = test_qttt_quality_comparison()
    except Exception as e:
        print(f"\nQuality comparison failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        all_results["combined"] = test_qttt_with_rabitq()
    except Exception as e:
        print(f"\nCombined test failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        all_results["steps"] = test_qttt_different_steps()
    except Exception as e:
        print(f"\nStep count test failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if "basic" in all_results:
        r = all_results["basic"]
        print(
            f"\n✓ Basic generation: {r['qttt_time']:.2f}s ({r['qttt_time']/r['baseline_time']:.1f}x baseline)"
        )
        if not r["tokens_match"]:
            print("  → qTTT produces different (adapted) outputs as expected")

    if "combined" in all_results:
        print(f"✓ qTTT + RaBitQ: {all_results['combined']['time']:.2f}s")

    print("\n" + "=" * 70)
    print("✓ qTTT END-TO-END TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
