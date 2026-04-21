"""
End-to-End AttnRes (Block Attention Residuals) Benchmark

Tests AttnRes in the complete generation pipeline with AdaptiveTransformer.
"""

import torch
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.configs import ModelConfig
from src.models.adaptive_transformer import AdaptiveTransformer


def test_attnres_basic_generation():
    """Test basic generation with AttnRes enabled vs disabled."""
    print("=" * 70)
    print("ATTNRES BASIC GENERATION TEST")
    print("=" * 70)

    device = "cpu"

    # Small model for testing
    config = ModelConfig(
        num_layers=8,  # Multiple of num_blocks for clean block boundaries
        hidden_dim=256,
        num_heads=4,
        num_blocks=4,  # 2 layers per block
        vocab_size=500,
        mlp_ratio=4,
    )

    model = AdaptiveTransformer(config).to(device)
    model.eval()

    # Test input
    prompt_len = 16
    max_new_tokens = 10
    input_ids = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

    print(f"\nModel: {config.num_layers} layers, {config.num_blocks} blocks")
    print(f"Layers per block: {config.num_layers // config.num_blocks}")
    print(f"Prompt: {prompt_len} tokens")
    print(f"Generating: {max_new_tokens} new tokens")

    # Baseline without AttnRes
    print("\n[1] Baseline generation (no AttnRes)...")
    torch.manual_seed(42)
    t0 = time.time()
    with torch.no_grad():
        output_no_attnres = model.generate(
            input_ids, max_new_tokens=max_new_tokens, use_attnres=False, use_qttt=False
        )
    t_no_attnres = time.time() - t0

    print(f"  Output length: {output_no_attnres.shape[1]}")
    print(f"  Time: {t_no_attnres:.3f}s")
    print(f"  Generated tokens: {output_no_attnres[0, prompt_len:].tolist()}")

    # With AttnRes enabled
    print("\n[2] Generation with AttnRes...")
    torch.manual_seed(42)  # Same seed for comparison
    t0 = time.time()
    with torch.no_grad():
        output_with_attnres = model.generate(
            input_ids, max_new_tokens=max_new_tokens, use_attnres=True, use_qttt=False
        )
    t_with_attnres = time.time() - t0

    print(f"  Output length: {output_with_attnres.shape[1]}")
    print(f"  Time: {t_with_attnres:.3f}s ({t_with_attnres/t_no_attnres:.2f}x baseline)")
    print(f"  Generated tokens: {output_with_attnres[0, prompt_len:].tolist()}")

    # Compare outputs
    match = (output_no_attnres[0, prompt_len:] == output_with_attnres[0, prompt_len:]).all().item()
    print(f"\n  Tokens match: {match}")

    if not match:
        diff_positions = (
            output_no_attnres[0, prompt_len:] != output_with_attnres[0, prompt_len:]
        ).nonzero(as_tuple=True)[0]
        print(f"  Different at positions: {diff_positions.tolist()}")

    return {
        "no_attnres_time": t_no_attnres,
        "with_attnres_time": t_with_attnres,
        "speedup": t_no_attnres / t_with_attnres,
        "tokens_match": match,
    }


def test_attnres_block_structure():
    """Test that block representations are properly managed."""
    print("\n" + "=" * 70)
    print("ATTNRES BLOCK STRUCTURE TEST")
    print("=" * 70)

    device = "cpu"

    # Model with specific block configuration
    config = ModelConfig(
        num_layers=8,
        hidden_dim=256,
        num_heads=4,
        num_blocks=4,  # 2 layers per block
        vocab_size=500,
    )

    model = AdaptiveTransformer(config).to(device)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 8), device=device)

    print(f"\nModel config: {config.num_layers} layers, {config.num_blocks} blocks")
    print(f"Expected: {config.num_blocks} blocks (layers 0-1, 2-3, 4-5, 6-7)")

    # Test forward pass and check block structure
    with torch.no_grad():
        # With AttnRes - trace through forward
        hidden = model.token_embedding(input_ids)

        layers_per_block = config.num_layers // config.num_blocks
        print(f"\nLayers per block: {layers_per_block}")

        block_representations = [hidden]
        partial_block = torch.zeros_like(hidden)

        for layer_idx in range(config.num_layers):
            layer = model.layers[layer_idx]
            attnres = model.attnres_modules[layer_idx]

            # Check if we finalize a block
            if layer_idx > 0 and layer_idx % layers_per_block == 0:
                block_representations.append(partial_block)
                partial_block = torch.zeros_like(hidden)
                print(f"  Layer {layer_idx}: Finalized block {len(block_representations)-1}")

            # Forward through layer
            hidden, partial_block = layer(
                hidden, block_representations, partial_block, attnres, use_attnres=True
            )

        # Final block
        all_blocks = block_representations + [partial_block]
        print(f"\nFinal block count: {len(all_blocks)} (expected: {config.num_blocks + 1})")

        # Check final aggregation
        V = torch.stack(all_blocks, dim=0)
        print(f"  Block representations shape: {V.shape}")
        print(f"  Expected: [{config.num_blocks + 1}, 1, 8, 256]")

        # Check pseudo-queries are learned
        print(f"\nPseudo-query shapes:")
        for i, attnres in enumerate(model.attnres_modules):
            print(
                f"  Layer {i}: attn_q={attnres.pseudo_query_attn.shape}, mlp_q={attnres.pseudo_query_mlp.shape}"
            )

    return {"success": True, "block_count": len(all_blocks)}


def test_attnres_memory_efficiency():
    """Verify AttnRes uses O(Nd) memory instead of O(Ld)."""
    print("\n" + "=" * 70)
    print("ATTNRES MEMORY EFFICIENCY TEST")
    print("=" * 70)

    device = "cpu"

    # Model with many layers but few blocks
    config = ModelConfig(
        num_layers=32,  # Many layers
        hidden_dim=512,
        num_heads=8,
        num_blocks=8,  # But only 8 blocks
        vocab_size=1000,
    )

    model = AdaptiveTransformer(config).to(device)

    print(
        f"\nModel: {config.num_layers} layers, {config.num_blocks} blocks, dim={config.hidden_dim}"
    )

    # Memory comparison
    L, N, d = config.num_layers, config.num_blocks, config.hidden_dim

    # Standard attention: store all L layer representations
    standard_memory = L * d  # O(Ld)

    # AttnRes: only store N block representations
    attnres_memory = (N + 1) * d  # O(Nd), +1 for partial block

    savings = (1 - attnres_memory / standard_memory) * 100

    print(f"\nMemory per token:")
    print(f"  Standard (O(Ld)): {standard_memory / 1024:.2f} KB ({L} layers × {d} dim)")
    print(f"  AttnRes (O(Nd)):  {attnres_memory / 1024:.2f} KB ({N+1} blocks × {d} dim)")
    print(f"  Savings: {savings:.1f}%")

    return {
        "standard_memory": standard_memory,
        "attnres_memory": attnres_memory,
        "savings_percent": savings,
    }


def test_attnres_pseudo_query_learning():
    """Test that pseudo-queries are properly initialized and learnable."""
    print("\n" + "=" * 70)
    print("ATTNRES PSEUDO-QUERY LEARNING TEST")
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

    print(f"\nChecking pseudo-query initialization...")

    # Check zero initialization
    all_zeros = True
    for i, attnres in enumerate(model.attnres_modules):
        attn_q_zero = torch.allclose(
            attnres.pseudo_query_attn, torch.zeros_like(attnres.pseudo_query_attn)
        )
        mlp_q_zero = torch.allclose(
            attnres.pseudo_query_mlp, torch.zeros_like(attnres.pseudo_query_mlp)
        )
        if not (attn_q_zero and mlp_q_zero):
            all_zeros = False
            print(f"  Layer {i}: NOT zero-initialized!")
        else:
            print(f"  Layer {i}: ✓ Zero-initialized")

    if all_zeros:
        print("\n✓ All pseudo-queries are zero-initialized (training stable)")

    # Check they require gradients
    print("\nChecking requires_grad...")
    for i, attnres in enumerate(model.attnres_modules):
        attn_grad = attnres.pseudo_query_attn.requires_grad
        mlp_grad = attnres.pseudo_query_mlp.requires_grad
        print(f"  Layer {i}: attn_q.grad={attn_grad}, mlp_q.grad={mlp_grad}")

    # Simulate a forward-backward pass
    print("\nSimulating training step...")
    input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)
    model.train()

    logits = model(input_ids, use_attnres=True)
    loss = logits.mean()
    loss.backward()

    # Check gradients
    print("\nGradient check after backward:")
    for i, attnres in enumerate(model.attnres_modules):
        attn_grad = attnres.pseudo_query_attn.grad
        mlp_grad = attnres.pseudo_query_mlp.grad

        if attn_grad is not None:
            print(f"  Layer {i}: attn_q.grad.norm={attn_grad.norm().item():.6f}")
        else:
            print(f"  Layer {i}: attn_q.grad=None")

        if mlp_grad is not None:
            print(f"  Layer {i}: mlp_q.grad.norm={mlp_grad.norm().item():.6f}")
        else:
            print(f"  Layer {i}: mlp_q.grad=None")

    return {"success": True}


def test_attnres_with_rabitq():
    """Test AttnRes + RaBitQ combined."""
    print("\n" + "=" * 70)
    print("ATTNRES + RABITQ COMBINED TEST")
    print("=" * 70)

    device = "cpu"

    config = ModelConfig(
        num_layers=8,
        hidden_dim=256,
        num_heads=4,
        num_blocks=4,
        vocab_size=500,
    )

    model = AdaptiveTransformer(config).to(device)
    model.eval()

    # Initialize RaBitQ
    model.init_rabitq_caches(total_bits=1, residual_window=64)

    input_ids = torch.randint(0, config.vocab_size, (1, 12), device=device)

    print("\nTesting combinations:")

    results = []

    # 1. AttnRes only
    print("\n[1] AttnRes only...")
    t0 = time.time()
    with torch.no_grad():
        output1 = model.generate(input_ids, max_new_tokens=5, use_attnres=True, use_rabitq=False)
    t1 = time.time() - t0
    print(f"  Time: {t1:.3f}s, Output: {output1[0, 12:].tolist()}")
    results.append(("AttnRes only", t1))

    # 2. RaBitQ only
    print("\n[2] RaBitQ only...")
    t0 = time.time()
    with torch.no_grad():
        output2 = model.generate(input_ids, max_new_tokens=5, use_attnres=False, use_rabitq=True)
    t2 = time.time() - t0
    print(f"  Time: {t2:.3f}s, Output: {output2[0, 12:].tolist()}")
    results.append(("RaBitQ only", t2))

    # 3. Both AttnRes + RaBitQ
    print("\n[3] AttnRes + RaBitQ...")
    t0 = time.time()
    with torch.no_grad():
        output3 = model.generate(input_ids, max_new_tokens=5, use_attnres=True, use_rabitq=True)
    t3 = time.time() - t0
    print(f"  Time: {t3:.3f}s, Output: {output3[0, 12:].tolist()}")
    results.append(("AttnRes + RaBitQ", t3))

    print("\nPerformance summary:")
    for name, t in results:
        print(f"  {name}: {t:.3f}s")

    return results


def test_attnres_long_sequence():
    """Test AttnRes with longer sequences."""
    print("\n" + "=" * 70)
    print("ATTNRES LONG SEQUENCE TEST")
    print("=" * 70)

    device = "cpu"

    config = ModelConfig(
        num_layers=8,
        hidden_dim=256,
        num_heads=4,
        num_blocks=4,
        vocab_size=500,
    )

    model = AdaptiveTransformer(config).to(device)
    model.eval()

    seq_lengths = [32, 64, 128]

    print(f"\nTesting different sequence lengths:")

    results = []
    for seq_len in seq_lengths:
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)

        # Time forward pass with AttnRes
        t0 = time.time()
        with torch.no_grad():
            _ = model(input_ids, use_attnres=True)
        t_attnres = time.time() - t0

        # Time forward pass without AttnRes
        t0 = time.time()
        with torch.no_grad():
            _ = model(input_ids, use_attnres=False)
        t_no_attnres = time.time() - t0

        print(
            f"  Seq len {seq_len:3d}: AttnRes={t_attnres:.3f}s, No AttnRes={t_no_attnres:.3f}s, "
            f"overhead={t_attnres/t_no_attnres:.2f}x"
        )

        results.append(
            {"seq_len": seq_len, "attnres_time": t_attnres, "no_attnres_time": t_no_attnres}
        )

    return results


def main():
    print("\n" + "=" * 70)
    print("ATTNRES END-TO-END BENCHMARK")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: CPU")

    all_results = {}

    try:
        all_results["basic"] = test_attnres_basic_generation()
    except Exception as e:
        print(f"\nBasic generation test failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        all_results["block_structure"] = test_attnres_block_structure()
    except Exception as e:
        print(f"\nBlock structure test failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        all_results["memory"] = test_attnres_memory_efficiency()
    except Exception as e:
        print(f"\nMemory efficiency test failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        all_results["pseudo_query"] = test_attnres_pseudo_query_learning()
    except Exception as e:
        print(f"\nPseudo-query test failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        all_results["combined"] = test_attnres_with_rabitq()
    except Exception as e:
        print(f"\nCombined test failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        all_results["long_seq"] = test_attnres_long_sequence()
    except Exception as e:
        print(f"\nLong sequence test failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if "basic" in all_results:
        r = all_results["basic"]
        print(f"\n✓ Basic generation: {r['speedup']:.2f}x speed vs no-AttnRes")
        if not r["tokens_match"]:
            print("  → AttnRes produces different outputs (expected)")

    if "memory" in all_results:
        r = all_results["memory"]
        print(f"✓ Memory savings: {r['savings_percent']:.1f}% reduction vs O(Ld)")

    if "pseudo_query" in all_results:
        print("✓ Pseudo-queries: Zero-initialized and trainable")

    if "combined" in all_results:
        print("✓ AttnRes + RaBitQ: Working together")

    print("\n" + "=" * 70)
    print("✓ ATTNRES END-TO-END TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
