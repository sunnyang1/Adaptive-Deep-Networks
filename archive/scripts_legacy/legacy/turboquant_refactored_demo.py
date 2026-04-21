#!/usr/bin/env python3
"""
Refactored TurboQuant Demo

Demonstrates the new unified TurboQuant API.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from src.turboquant import TurboQuant, RECOMMENDED_CONFIGS


def demo_simple_api():
    """Demonstrate the simple, unified API."""
    print("=" * 70)
    print("Refactored TurboQuant - Simple Unified API")
    print("=" * 70)
    print()

    # Create sample data
    batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)

    print(f"Input shape: {keys.shape}")
    print()

    # Test different modes
    modes = ["fp16", "int8", "tq4", "tq3"]

    print(f"{'Mode':<10} {'Ratio':<8} {'Error':<12} {'Status'}")
    print("-" * 70)

    for mode in modes:
        # Create quantizer
        quant = TurboQuant(mode, head_dim=head_dim)

        # Fit (required for TQ modes)
        if mode in ["tq3", "tq4"]:
            quant.fit(keys, values)

        # Compress/decompress
        compressed = quant.compress_kv(keys, values)
        keys_deq, values_deq = quant.decompress_kv(compressed)

        # Calculate error
        error = (keys - keys_deq).abs().mean().item()

        # Get stats
        stats = quant.memory_stats(seq_len=seq_len, batch_size=batch_size, num_heads=num_heads)

        status = "✅" if error < 0.1 else "⚠️"
        print(f"{mode:<10} {stats['compression_ratio']:<8.2f}x {error:<12.6f} {status}")

    print()


def demo_context_manager():
    """Demonstrate context manager usage."""
    print("=" * 70)
    print("Context Manager API")
    print("=" * 70)
    print()

    keys = torch.randn(2, 8, 128, 64)
    values = torch.randn(2, 8, 128, 64)

    # Using context manager
    with TurboQuant("tq4", head_dim=64) as quant:
        quant.fit(keys, values)
        compressed = quant.compress_kv(keys, values)
        keys_deq, values_deq = quant.decompress_kv(compressed)

    print("Compressed and decompressed using context manager")
    print(f"Reconstruction error: {(keys - keys_deq).abs().mean().item():.6f}")
    print()


def demo_memory_stats():
    """Demonstrate memory statistics."""
    print("=" * 70)
    print("Memory Statistics for Long Sequences")
    print("=" * 70)
    print()

    seq_lengths = [4096, 16384, 65536, 131072]
    modes = [("FP16", "fp16"), ("INT8", "int8"), ("TQ4", "tq4")]

    print(f"{'Seq Length':<12} {'Mode':<8} {'Original':<12} {'Compressed':<12} {'Saving'}")
    print("-" * 70)

    for seq_len in seq_lengths:
        for name, mode in modes:
            quant = TurboQuant(mode, head_dim=128)
            stats = quant.memory_stats(seq_len=seq_len, batch_size=1, num_heads=32)

            orig = f"{stats['original_mb']:.1f} MB"
            comp = f"{stats['compressed_mb']:.1f} MB"
            saving = f"{stats['memory_saved']:.0f}%"

            print(f"{seq_len:<12} {name:<8} {orig:<12} {comp:<12} {saving}")
        print()


def demo_recommended_configs():
    """Show recommended configurations."""
    print("=" * 70)
    print("Recommended Configurations")
    print("=" * 70)
    print()

    print(f"{'Config':<20} {'Mode':<12} {'Description'}")
    print("-" * 70)

    descriptions = {
        "small_model": "For models < 4B parameters",
        "balanced": "Best accuracy/compression trade-off",
        "fast": "3x compression for 4B+ models",
        "extreme": "Maximum compression for 4B+ models",
        "flash_attention": "With FlashAttention enabled",
    }

    for name, mode in RECOMMENDED_CONFIGS.items():
        quant = TurboQuant(mode)
        desc = descriptions.get(name, "")
        print(f"{name:<20} {mode:<12} {desc}")

    print()


def demo_flash_attention():
    """Demonstrate FlashAttention modes."""
    print("=" * 70)
    print("FlashAttention Integration")
    print("=" * 70)
    print()

    modes = ["tq4", "tq4_flash", "int8", "int8_flash"]

    print(f"{'Mode':<15} {'FlashAttn':<12} {'Compression'}")
    print("-" * 70)

    for mode in modes:
        quant = TurboQuant(mode, head_dim=128)
        flash = "Yes" if quant.config.flash_attention else "No"
        ratio = f"{quant.config.compression_ratio:.2f}x"
        print(f"{mode:<15} {flash:<12} {ratio}")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Refactored TurboQuant Demo")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--simple", action="store_true", help="Simple API demo")
    parser.add_argument("--context", action="store_true", help="Context manager demo")
    parser.add_argument("--memory", action="store_true", help="Memory stats demo")
    parser.add_argument("--configs", action="store_true", help="Recommended configs")
    parser.add_argument("--flash", action="store_true", help="FlashAttention demo")

    args = parser.parse_args()

    if not any([args.simple, args.context, args.memory, args.configs, args.flash]):
        args.all = True

    if args.all or args.simple:
        demo_simple_api()

    if args.all or args.context:
        demo_context_manager()

    if args.all or args.memory:
        demo_memory_stats()

    if args.all or args.configs:
        demo_recommended_configs()

    if args.all or args.flash:
        demo_flash_attention()

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
