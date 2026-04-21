#!/usr/bin/env python3
"""
TurboQuant V3 Demo - Community-Improved Implementation

Based on: https://github.com/tonbistudio/turboquant-pytorch

Key improvements:
1. MSE-only (no QJL) - Better for softmax attention
2. Asymmetric K/V bits - Keys get more precision
3. Bit-packed storage - Real compression ratios
4. Layer-adaptive - Protect sensitive layers

Results (tonbistudio on Qwen2.5-3B):
- V2 with QJL: 0/27 generation tests passed
- V3 without QJL: 18/18 generation tests passed
- K4/V2: 5.1x compression with perfect retrieval
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from src.turboquant import (
    TurboQuantV3,
    create_v3_k4_v2,
    create_v3_k3_v2,
    create_v3_layer_adaptive,
    V3_RECOMMENDED,
)


def demo_v3_vs_legacy():
    """Compare V3 with legacy approach."""
    print("=" * 70)
    print("TurboQuant V3 vs Legacy (MSE-only vs MSE+QJL)")
    print("=" * 70)
    print()

    print("Key Finding from tonbistudio:")
    print("-" * 70)
    print("QJL hurts for KV cache because:")
    print("  1. QJL provides unbiased inner products")
    print("  2. BUT attention applies SOFTMAX to scores")
    print("  3. Softmax exponentially amplifies variance")
    print("  4. QJL's random noise gets magnified")
    print()
    print("Result:")
    print("  V2 (MSE+QJL): 0/27 generation tests passed")
    print("  V3 (MSE-only): 18/18 generation tests passed")
    print()


def demo_asymmetric_kv():
    """Demonstrate asymmetric K/V bit allocation."""
    print("=" * 70)
    print("Asymmetric K/V Bit Allocation")
    print("=" * 70)
    print()

    print("Why asymmetric allocation?")
    print("-" * 70)
    print("Keys: Decide WHICH tokens to attend to (needs precision)")
    print("Values: Content that gets averaged (errors cancel out)")
    print()
    print("Recommended configs:")
    print()

    configs = [
        ("K4/V2", 4, 2, "5.1x", "Best quality"),
        ("K3/V2", 3, 2, "6.0x", "Good quality"),
        ("K4/V3", 4, 3, "3.8x", "Higher quality values"),
    ]

    print(f"{'Config':<10} {'Key Bits':<10} {'Value Bits':<12} {'Ratio':<8} {'Notes'}")
    print("-" * 70)
    for name, k_bits, v_bits, ratio, notes in configs:
        print(f"{name:<10} {k_bits:<10} {v_bits:<12} {ratio:<8} {notes}")

    print()


def demo_compression():
    """Demonstrate actual compression."""
    print("=" * 70)
    print("V3 Compression Demo")
    print("=" * 70)
    print()

    # Create sample KV cache
    batch_size = 2
    num_heads = 8
    seq_len = 512
    head_dim = 64

    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)

    print(f"Input: {keys.shape}")
    print()

    # Test K4/V2
    print("Testing K4/V2 (4-bit keys, 2-bit values):")
    print("-" * 70)

    v3 = create_v3_k4_v2(head_dim=head_dim)

    # Fit on sample
    v3.fit(keys[:1, :1, :64], values[:1, :1, :64], head_dim=head_dim, layer_idx=0)

    # Compress
    compressed = v3.compress_kv(keys, values, head_dim=head_dim, layer_idx=0)

    # Decompress
    keys_deq, values_deq = v3.decompress_kv(compressed)

    # Calculate metrics
    key_error = (keys - keys_deq).abs().mean().item()
    value_error = (values - values_deq).abs().mean().item()
    ratio = v3.get_compression_ratio(0)

    print(f"  Compression ratio: {ratio:.2f}x")
    print(f"  Key reconstruction error: {key_error:.6f}")
    print(f"  Value reconstruction error: {value_error:.6f}")
    print()


def demo_memory_savings():
    """Show memory savings for long contexts."""
    print("=" * 70)
    print("Memory Savings for Long Contexts")
    print("=" * 70)
    print()

    seq_lengths = [4096, 8192, 16384, 32768, 65536]

    print(f"{'Seq Length':<12} {'FP16':<12} {'V3 K4/V2':<12} {'Saved':<10}")
    print("-" * 70)

    for seq_len in seq_lengths:
        v3 = create_v3_k4_v2()
        stats = v3.memory_stats(
            seq_len=seq_len, num_layers=1, batch_size=1, num_heads=32, head_dim=128
        )

        orig = f"{stats['original_mb']:.1f} MB"
        comp = f"{stats['compressed_mb']:.1f} MB"
        saved = f"{stats['memory_saved_percent']:.0f}%"

        print(f"{seq_len:<12} {orig:<12} {comp:<12} {saved:<10}")

    print()


def demo_layer_adaptive():
    """Demonstrate layer-adaptive compression."""
    print("=" * 70)
    print("Layer-Adaptive Compression")
    print("=" * 70)
    print()

    print("First and last layers get MORE bits (protected)")
    print()

    v3 = create_v3_layer_adaptive(key_bits=3, value_bits=2, protected_layers=2, total_layers=32)

    print(f"{'Layer':<10} {'Key Bits':<10} {'Value Bits':<12} {'Protected'}")
    print("-" * 70)

    for layer_idx in [0, 1, 2, 15, 29, 30, 31]:
        k_bits, v_bits = v3._get_layer_bits(layer_idx)
        protected = "Yes" if (layer_idx < 2 or layer_idx >= 30) else "No"
        print(f"{layer_idx:<10} {k_bits:<10} {v_bits:<12} {protected}")

    print()


def demo_recommended_configs():
    """Show recommended V3 configurations."""
    print("=" * 70)
    print("Recommended V3 Configurations")
    print("=" * 70)
    print()

    print("From tonbistudio testing on Qwen2.5-3B:")
    print()

    configs = [
        ("k4_v2", "K4/V2", "5.1x", "18/18 perfect", "Best overall"),
        ("k3_v2", "K3/V2", "6.0x", "18/18 perfect", "Max compression"),
        ("k4_v2_protected", "K4/V2 Protected", "3.6x", "99% top-1", "Layer-adaptive"),
    ]

    print(f"{'Name':<20} {'Config':<20} {'Ratio':<10} {'Result':<20} {'Notes'}")
    print("-" * 70)
    for name, config, ratio, result, notes in configs:
        print(f"{name:<20} {config:<20} {ratio:<10} {result:<20} {notes}")

    print()


def demo_usage():
    """Show usage example."""
    print("=" * 70)
    print("Usage Example")
    print("=" * 70)
    print()

    code = """
from src.turboquant import create_v3_k4_v2

# Create V3 compressor (recommended: K4/V2)
v3 = create_v3_k4_v2(head_dim=128, device='cuda')

# Fit on sample data for each layer
for layer_idx in range(num_layers):
    v3.fit(
        sample_keys[layer_idx],
        sample_values[layer_idx],
        head_dim=128,
        layer_idx=layer_idx
    )

# Compress during inference
compressed = v3.compress_kv(keys, values, head_dim=128, layer_idx=layer_idx)

# Decompress when needed
keys_deq, values_deq = v3.decompress_kv(compressed)
    """

    print(code)
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TurboQuant V3 Demo")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--v3-vs-legacy", action="store_true", help="V3 vs legacy")
    parser.add_argument("--asymmetric", action="store_true", help="Asymmetric K/V")
    parser.add_argument("--compression", action="store_true", help="Compression demo")
    parser.add_argument("--memory", action="store_true", help="Memory savings")
    parser.add_argument("--layer-adaptive", action="store_true", help="Layer-adaptive")
    parser.add_argument("--configs", action="store_true", help="Recommended configs")
    parser.add_argument("--usage", action="store_true", help="Usage example")

    args = parser.parse_args()

    if not any([getattr(args, attr) for attr in vars(args) if attr != "all"]):
        args.all = True

    if args.all or args.v3_vs_legacy:
        demo_v3_vs_legacy()

    if args.all or args.asymmetric:
        demo_asymmetric_kv()

    if args.all or args.compression:
        demo_compression()

    if args.all or args.memory:
        demo_memory_savings()

    if args.all or args.layer_adaptive:
        demo_layer_adaptive()

    if args.all or args.configs:
        demo_recommended_configs()

    if args.all or args.usage:
        demo_usage()

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print()
    print("Reference: https://github.com/tonbistudio/turboquant-pytorch")


if __name__ == "__main__":
    main()
