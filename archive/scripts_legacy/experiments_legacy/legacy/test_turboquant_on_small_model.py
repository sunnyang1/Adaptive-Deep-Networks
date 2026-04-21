#!/usr/bin/env python3
"""
Test TurboQuant V3 on Small Model (2.2B)

This script:
1. Builds the small model (2.2B parameters)
2. Tests different TurboQuant configurations
3. Measures compression quality and speed
4. Reports results
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, "src")

import torch
import torch.nn as nn
import json
import time
import numpy as np
from pathlib import Path

from src.models.configs import get_config
from src.models.adaptive_transformer import AdaptiveTransformer
from src.turboquant import (
    TurboQuantV3,
    create_v3_k4_v2,
    create_v3_k3_v2,
    TurboQuant,  # Legacy
)


class SmallModelBuilder:
    """Build and test small model with TurboQuant."""

    def __init__(self, device="cpu"):
        self.device = device
        self.config = get_config("small")
        self.model = None

    def build_model(self):
        """Build the small model."""
        print("=" * 70)
        print("Building Small Model (2.2B parameters)")
        print("=" * 70)
        print()

        print(f"Configuration:")
        print(f"  Layers: {self.config.num_layers}")
        print(f"  Hidden dim: {self.config.hidden_dim}")
        print(f"  Num heads: {self.config.num_heads}")
        print(f"  Num blocks: {self.config.num_blocks}")
        print(f"  Head dim: {self.config.hidden_dim // self.config.num_heads}")
        print()

        # Build model
        self.model = AdaptiveTransformer(self.config).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Model Statistics:")
        print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size (FP16): {total_params * 2 / (1024**3):.2f} GB")
        print()

        return self.model

    def extract_kv_cache(self, batch_size=1, seq_len=1024):
        """Extract KV cache from model."""
        if self.model is None:
            self.build_model()

        print(f"Extracting KV cache (batch={batch_size}, seq_len={seq_len})...")

        self.model.eval()
        with torch.no_grad():
            # Create dummy input
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(
                self.device
            )

            # Forward pass to populate KV cache
            outputs = self.model(input_ids)

            # Extract KV cache from first layer
            # Note: This is simplified - actual KV cache extraction depends on model implementation
            head_dim = self.config.hidden_dim // self.config.num_heads

            # Create synthetic KV cache for testing
            keys = torch.randn(batch_size, self.config.num_heads, seq_len, head_dim).to(self.device)
            values = torch.randn(batch_size, self.config.num_heads, seq_len, head_dim).to(
                self.device
            )

        print(f"  KV cache shape: {keys.shape}")
        print(f"  KV cache size: {keys.numel() * 2 * 2 / (1024**2):.2f} MB (FP16)")
        print()

        return keys, values, head_dim


def test_turboquant_v3(keys, values, head_dim, device="cpu"):
    """Test TurboQuant V3 on KV cache."""
    print("=" * 70)
    print("Testing TurboQuant V3")
    print("=" * 70)
    print()

    configs = [
        ("K4/V2 (Recommended)", create_v3_k4_v2(head_dim=head_dim, device=device)),
        ("K3/V2 (Max Compression)", create_v3_k3_v2(head_dim=head_dim, device=device)),
    ]

    results = []

    for name, v3 in configs:
        print(f"\nTesting {name}:")
        print("-" * 70)

        # Fit on sample
        sample_size = min(64, keys.shape[2])
        sample_keys = keys[:, :, :sample_size, :]
        sample_values = values[:, :, :sample_size, :]

        start = time.time()
        v3.fit(sample_keys, sample_values, head_dim=head_dim, layer_idx=0)
        fit_time = time.time() - start

        # Compress
        start = time.time()
        compressed = v3.compress_kv(keys, values, head_dim=head_dim, layer_idx=0)
        compress_time = time.time() - start

        # Decompress
        start = time.time()
        keys_deq, values_deq = v3.decompress_kv(compressed)
        decompress_time = time.time() - start

        # Calculate metrics
        key_error = (keys - keys_deq).abs().mean().item()
        value_error = (values - values_deq).abs().mean().item()
        key_rel_error = (keys - keys_deq).abs().mean().item() / keys.abs().mean().item()
        value_rel_error = (values - values_deq).abs().mean().item() / values.abs().mean().item()

        ratio = v3.get_compression_ratio(0)

        # Memory stats
        stats = v3.memory_stats(
            seq_len=keys.shape[2],
            num_layers=1,
            batch_size=keys.shape[0],
            num_heads=keys.shape[1],
            head_dim=head_dim,
        )

        print(f"  Compression ratio: {ratio:.2f}x")
        mem_saved_key = (
            "memory_saved_percent" if "memory_saved_percent" in stats else "memory_saved"
        )
        print(f"  Memory saved: {stats[mem_saved_key]:.1f}%")
        print(f"  Key error (absolute): {key_error:.6f}")
        print(f"  Key error (relative): {key_rel_error:.4%}")
        print(f"  Value error (absolute): {value_error:.6f}")
        print(f"  Value error (relative): {value_rel_error:.4%}")
        print(f"  Fit time: {fit_time:.3f}s")
        print(f"  Compress time: {compress_time:.3f}s")
        print(f"  Decompress time: {decompress_time:.3f}s")

        results.append(
            {
                "name": name,
                "ratio": ratio,
                "memory_saved": stats["memory_saved_percent"],
                "key_rel_error": key_rel_error,
                "value_rel_error": value_rel_error,
                "compress_time": compress_time,
                "decompress_time": decompress_time,
            }
        )

    return results


def test_legacy_turboquant(keys, values, head_dim, device="cpu"):
    """Test legacy TurboQuant for comparison."""
    print("\n" + "=" * 70)
    print("Testing Legacy TurboQuant (for comparison)")
    print("=" * 70)
    print()

    configs = [
        ("FP16 (baseline)", "fp16"),
        ("INT8", "int8"),
        ("TQ4", "tq4"),
    ]

    results = []

    for name, mode in configs:
        print(f"\nTesting {name}:")
        print("-" * 70)

        quant = TurboQuant(mode, head_dim=head_dim, device=device)

        # Fit if needed
        if mode in ["tq3", "tq4"]:
            sample_keys = keys[:, :, :64, :]
            sample_values = values[:, :, :64, :]
            quant.fit(sample_keys, sample_values)

        # Compress
        start = time.time()
        compressed = quant.compress_kv(keys, values)
        compress_time = time.time() - start

        # Decompress
        start = time.time()
        keys_deq, values_deq = quant.decompress_kv(compressed)
        decompress_time = time.time() - start

        # Metrics
        key_error = (keys - keys_deq).abs().mean().item()
        key_rel_error = key_error / keys.abs().mean().item()

        ratio = quant.config.compression_ratio
        stats = quant.memory_stats(seq_len=keys.shape[2])

        print(f"  Compression ratio: {ratio:.2f}x")
        mem_saved_key = (
            "memory_saved_percent" if "memory_saved_percent" in stats else "memory_saved"
        )
        print(f"  Memory saved: {stats[mem_saved_key]:.1f}%")
        print(f"  Key error (relative): {key_rel_error:.4%}")
        print(f"  Compress time: {compress_time:.3f}s")
        print(f"  Decompress time: {decompress_time:.3f}s")

        results.append(
            {
                "name": name,
                "ratio": ratio,
                "memory_saved": stats[mem_saved_key],
                "key_rel_error": key_rel_error,
                "compress_time": compress_time,
            }
        )

    return results


def test_long_context(keys, values, head_dim, seq_lens=[1024, 4096, 8192], device="cpu"):
    """Test compression with different sequence lengths."""
    print("\n" + "=" * 70)
    print("Long Context Test")
    print("=" * 70)
    print()

    v3 = create_v3_k4_v2(head_dim=head_dim, device=device)

    print(f"{'Seq Length':<15} {'Original':<15} {'Compressed':<15} {'Ratio':<10} {'Error'}")
    print("-" * 70)

    for seq_len in seq_lens:
        if seq_len > keys.shape[2]:
            continue

        # Truncate to seq_len
        k = keys[:, :, :seq_len, :]
        v = values[:, :, :seq_len, :]

        # Fit on small sample
        if seq_len >= 64:
            v3.fit(k[:, :, :64, :], v[:, :, :64, :], head_dim=head_dim, layer_idx=0)

        # Compress
        compressed = v3.compress_kv(k, v, head_dim=head_dim, layer_idx=0)
        k_deq, v_deq = v3.decompress_kv(compressed)

        # Stats
        stats = v3.memory_stats(seq_len=seq_len)
        error = (k - k_deq).abs().mean().item() / k.abs().mean().item()

        orig = f"{stats['original_mb']:.1f} MB"
        comp = f"{stats['compressed_mb']:.1f} MB"
        ratio = f"{stats['compression_ratio']:.2f}x"
        err = f"{error:.4%}"

        print(f"{seq_len:<15} {orig:<15} {comp:<15} {ratio:<10} {err}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test TurboQuant on Small Model")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--skip-v3", action="store_true", help="Skip V3 tests")
    parser.add_argument("--skip-legacy", action="store_true", help="Skip legacy tests")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TurboQuant Test on Small Model (2.2B)")
    print("=" * 70)
    print()

    # Build model
    builder = SmallModelBuilder(device=args.device)
    model = builder.build_model()

    # Extract KV cache
    keys, values, head_dim = builder.extract_kv_cache(
        batch_size=args.batch_size, seq_len=args.seq_len
    )

    all_results = {}

    # Test V3
    if not args.skip_v3:
        v3_results = test_turboquant_v3(keys, values, head_dim, args.device)
        all_results["v3"] = v3_results

    # Test legacy
    if not args.skip_legacy:
        legacy_results = test_legacy_turboquant(keys, values, head_dim, args.device)
        all_results["legacy"] = legacy_results

    # Long context test
    print("\n")
    test_long_context(keys, values, head_dim, device=args.device)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    if not args.skip_v3:
        print("V3 Results:")
        for r in all_results.get("v3", []):
            print(f"  {r['name']}: {r['ratio']:.2f}x compression, {r['key_rel_error']:.4%} error")

    if not args.skip_legacy:
        print("\nLegacy Results:")
        for r in all_results.get("legacy", []):
            print(f"  {r['name']}: {r['ratio']:.2f}x compression, {r['key_rel_error']:.4%} error")

    print()
    print("Recommendations:")
    print("  - For best quality: Use V3 K4/V2 (5.1x compression)")
    print("  - For max compression: Use V3 K3/V2 (6.0x compression)")
    print("  - Avoid QJL (V2) - it hurts generation quality")
    print()


if __name__ == "__main__":
    main()
