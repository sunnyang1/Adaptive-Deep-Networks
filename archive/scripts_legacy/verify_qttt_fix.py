#!/usr/bin/env python3
"""
Verify qTTT forward propagation fix.

Tests that qTTT now uses complete model forward instead of just attention.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.configs import get_config
from src.models.adaptive_transformer import AdaptiveTransformer
from src.qttt.polar_adaptation import PolarQTTT, PolarQTTTConfig
from src.qttt.adaptation import KVCache


def test_forward_with_frozen_kv():
    """Test the new forward_with_frozen_kv method."""
    print("=" * 70)
    print("TEST 1: forward_with_frozen_kv method")
    print("=" * 70)

    config = get_config("small")
    model = AdaptiveTransformer(config)
    model.eval()

    # Create input
    batch_size = 1
    seq_len = 5
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Get KV caches
    kv_caches = model.get_kv_cache(input_ids)
    print(f"✓ Generated KV caches for {len(kv_caches)} layers")

    # Test forward_with_frozen_kv
    logits = model.forward_with_frozen_kv(input_ids=input_ids, kv_caches=kv_caches)

    print(f"✓ forward_with_frozen_kv output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    print("✓ Output shape correct")

    # Compare with regular forward
    logits_regular = model.forward(input_ids)
    diff = torch.abs(logits - logits_regular).max()
    print(f"✓ Max difference from regular forward: {diff:.6f}")
    assert diff < 1e-3, f"Difference too large: {diff}"

    print("\n✅ TEST 1 PASSED\n")


def test_qttt_with_model_forward():
    """Test qTTT with full model forward."""
    print("=" * 70)
    print("TEST 2: qTTT with model forward")
    print("=" * 70)

    config = get_config("small")
    model = AdaptiveTransformer(config)
    model.eval()

    # Create input and caches
    batch_size = 1
    seq_len = 3
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    kv_caches = model.get_kv_cache(input_ids)

    # Create query
    queries = torch.randn(batch_size, seq_len, config.hidden_dim)

    # Create qTTT config
    cfg = PolarQTTTConfig(num_steps=2, learning_rate=0.01)
    qttt = PolarQTTT(cfg, config.hidden_dim, config.num_heads)

    print(f"✓ Query shape: {queries.shape}")
    print(f"✓ KV caches: {len(kv_caches)} layers")

    # Test with model (new API)
    try:
        with torch.enable_grad():
            adapted_queries, loss_history = qttt.adapt_query_projection(
                queries,
                kv_cache=kv_caches[-1],  # Legacy
                seq_positions=torch.arange(seq_len),
                model=model,  # NEW: Enable full forward
                input_ids=input_ids,  # NEW
                kv_caches=kv_caches,  # NEW
            )

        print(f"✓ Adapted query shape: {adapted_queries.shape}")
        print(f"✓ Loss history: {loss_history}")
        print(
            f"✓ Loss decreased: {loss_history[-1] < loss_history[0] if len(loss_history) > 1 else 'N/A'}"
        )

        print("\n✅ TEST 2 PASSED - qTTT now uses full model forward!\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def test_generate_with_fixed_qttt():
    """Test generation with fixed qTTT."""
    print("=" * 70)
    print("TEST 3: Generation with fixed qTTT")
    print("=" * 70)

    config = get_config("small")
    model = AdaptiveTransformer(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    # Generate without qTTT
    print("Generating without qTTT...")
    output_no_qttt = model.generate(input_ids, max_new_tokens=3, use_qttt=False)
    print(f"✓ Output shape (no qTTT): {output_no_qttt.shape}")

    # Generate with qTTT
    print("\nGenerating with qTTT (using full forward)...")
    try:
        output_with_qttt = model.generate(
            input_ids, max_new_tokens=3, use_qttt=True, qttt_config={"num_steps": 2}
        )
        print(f"✓ Output shape (with qTTT): {output_with_qttt.shape}")

        print(f"\nNo qTTT tokens: {output_no_qttt[0, -3:].tolist()}")
        print(f"With qTTT tokens: {output_with_qttt[0, -3:].tolist()}")

        print("\n✅ TEST 3 PASSED - Generation with fixed qTTT works!\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("qTTT FORWARD PROPAGATION FIX VERIFICATION")
    print("=" * 70 + "\n")

    results = []

    try:
        test_forward_with_frozen_kv()
        results.append(("forward_with_frozen_kv", True))
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}\n")
        results.append(("forward_with_frozen_kv", False))

    try:
        success = test_qttt_with_model_forward()
        results.append(("qttt_with_model_forward", success))
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}\n")
        results.append(("qttt_with_model_forward", False))

    try:
        success = test_generate_with_fixed_qttt()
        results.append(("generate_with_fixed_qttt", success))
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}\n")
        results.append(("generate_with_fixed_qttt", False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✅ ALL TESTS PASSED - qTTT fix verified!")
    else:
        print("\n❌ SOME TESTS FAILED")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
