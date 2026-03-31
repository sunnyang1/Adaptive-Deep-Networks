#!/usr/bin/env python3
"""
MNN-Inspired TurboQuant Demo

Demonstrates the improved TurboQuant implementation based on MNN's approach.

Reference: https://github.com/alibaba/MNN/commit/244f5d10df5a95b4f4e6f3d9251c6fe3dc0e7c83

Key improvements:
1. attention_mode encoding: flash_attention * 8 + kv_quant_mode
2. Separate KV quantization modes (TQ3/TQ4)
3. FlashAttention integration
4. Optimized Lloyd-Max codebook
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
from src.turboquant import (
    MNNTurboQuantConfig,
    MNNTurboQuantCompressor,
    AttentionMode,
    KVQuantMode,
    create_mnn_turboquant,
    CONFIG_RECOMMENDATIONS,
)


def demo_attention_modes():
    """Demonstrate different attention modes."""
    print("=" * 70)
    print("MNN TurboQuant Attention Modes")
    print("=" * 70)
    print()
    
    modes = [
        0,   # Standard + FP16
        2,   # Standard + KV-INT8
        8,   # FlashAttention + FP16 (default)
        10,  # FlashAttention + KV-INT8
        12,  # FlashAttention + KV-TQ3
        14,  # FlashAttention + KV-TQ4 (recommended for 4B+)
    ]
    
    print(f"{'Mode':<8} {'FlashAttn':<12} {'KV Mode':<15} {'Description':<30}")
    print("-" * 70)
    
    for mode in modes:
        config = MNNTurboQuantConfig(attention_mode=mode)
        fa = "Yes" if config.flash_attention else "No"
        kv = config.kv_quant_mode.name
        desc = AttentionMode.get_description(mode)
        print(f"{mode:<8} {fa:<12} {kv:<15} {desc:<30}")
    
    print()


def demo_compression_modes():
    """Demonstrate compression ratios."""
    print("=" * 70)
    print("Compression Ratio Comparison")
    print("=" * 70)
    print()
    
    modes = [
        ("FP16 (baseline)", 8),
        ("KV-INT8", 10),
        ("KV-TQ4", 14),
        ("KV-TQ3", 12),
        ("Key-only TQ4", 13),  # Mode 5 + 8
        ("Key-only TQ3", 11),  # Mode 3 + 8
    ]
    
    print(f"{'Configuration':<20} {'Mode':<8} {'Ratio':<10} {'Memory Saved':<15}")
    print("-" * 70)
    
    for name, mode in modes:
        config = MNNTurboQuantConfig(attention_mode=mode)
        ratio = config.compression_ratio
        saved = f"{(1 - 1/ratio) * 100:.1f}%"
        print(f"{name:<20} {mode:<8} {ratio:<10.2f}x {saved:<15}")
    
    print()


def demo_memory_calculation():
    """Calculate memory usage for different configurations."""
    print("=" * 70)
    print("Memory Usage Calculation (seq_len=32768, batch=1, heads=32)")
    print("=" * 70)
    print()
    
    configs = [
        ("FP16", 8),
        ("KV-INT8", 10),
        ("KV-TQ4", 14),
        ("KV-TQ3", 12),
    ]
    
    print(f"{'Config':<15} {'Original':<12} {'Compressed':<12} {'Ratio':<10} {'Saving':<10}")
    print("-" * 70)
    
    for name, mode in configs:
        compressor = create_mnn_turboquant(attention_mode=mode)
        stats = compressor.get_memory_stats(seq_len=32768)
        
        orig = f"{stats['original_mb']:.1f} MB"
        comp = f"{stats['compressed_mb']:.1f} MB"
        ratio = f"{stats['saving_ratio']:.2f}x"
        saving = f"{stats['compression_ratio']*100:.1f}%"
        
        print(f"{name:<15} {orig:<12} {comp:<12} {ratio:<10} {saving:<10}")
    
    print()


def demo_quantization():
    """Demonstrate actual quantization."""
    print("=" * 70)
    print("Live Quantization Demo")
    print("=" * 70)
    print()
    
    # Create sample KV cache
    batch_size = 2
    num_heads = 8
    seq_len = 128
    head_dim = 64
    
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print(f"Input shape: {keys.shape}")
    print(f"Input dtype: {keys.dtype}")
    print()
    
    # Test different modes
    for name, mode in [("FP16", 8), ("KV-TQ4", 14)]:
        print(f"Mode {mode} ({name}):")
        
        config = MNNTurboQuantConfig(attention_mode=mode)
        compressor = MNNTurboQuantCompressor(config, head_dim=head_dim)
        
        # Fit codebooks (for TQ modes)
        if mode in (12, 13, 14, 15):
            print("  Fitting Lloyd-Max codebooks...")
            compressor.fit_codebooks(keys, values)
        
        # Compress
        compressed = compressor.compress_kv(keys, values)
        
        # Calculate sizes
        orig_size = keys.element_size() * keys.nelement() + values.element_size() * values.nelement()
        
        comp_size = 0
        for k, v in compressed.items():
            if v is not None:
                comp_size += v.element_size() * v.nelement()
        
        ratio = orig_size / comp_size
        print(f"  Original size: {orig_size / 1024:.2f} KB")
        print(f"  Compressed size: {comp_size / 1024:.2f} KB")
        print(f"  Compression ratio: {ratio:.2f}x")
        
        # Decompress and check error
        keys_decomp, values_decomp = compressor.decompress_kv(compressed)
        
        key_error = (keys - keys_decomp).abs().mean().item()
        value_error = (values - values_decomp).abs().mean().item()
        
        print(f"  Key reconstruction error: {key_error:.6f}")
        print(f"  Value reconstruction error: {value_error:.6f}")
        print()


def demo_recommendations():
    """Show model-size aware recommendations."""
    print("=" * 70)
    print("Model-Size Aware Recommendations")
    print("=" * 70)
    print()
    
    model_sizes = [
        ("Small (2.2B)", 2.2e9),
        ("Medium (8.7B)", 8.7e9),
        ("Large (27B)", 27e9),
    ]
    
    print(f"{'Model Size':<20} {'Mode 14 (TQ4)':<15} {'Mode 12 (TQ3)':<15}")
    print("-" * 70)
    
    for name, params in model_sizes:
        config_tq4 = MNNTurboQuantConfig(attention_mode=14, min_params_for_tq=4e9)
        config_tq3 = MNNTurboQuantConfig(attention_mode=12, min_params_for_tq=4e9)
        
        rec_tq4 = "✅ Recommended" if config_tq4.is_recommended_for_model_size(params) else "⚠️ Not recommended (<4B)"
        rec_tq3 = "✅ Recommended" if config_tq3.is_recommended_for_model_size(params) else "⚠️ Not recommended (<4B)"
        
        print(f"{name:<20} {rec_tq4:<15} {rec_tq3:<15}")
    
    print()
    print("Note: MNN recommends TQ3/TQ4 only for 4B+ models to avoid accuracy loss")
    print()


def main():
    parser = argparse.ArgumentParser(description="MNN-Inspired TurboQuant Demo")
    parser.add_argument('--all', action='store_true', help='Run all demos')
    parser.add_argument('--modes', action='store_true', help='Show attention modes')
    parser.add_argument('--compression', action='store_true', help='Show compression ratios')
    parser.add_argument('--memory', action='store_true', help='Show memory calculations')
    parser.add_argument('--quantization', action='store_true', help='Run quantization demo')
    parser.add_argument('--recommendations', action='store_true', help='Show recommendations')
    
    args = parser.parse_args()
    
    # If no specific demo selected, run all
    if not any([args.modes, args.compression, args.memory, args.quantization, args.recommendations]):
        args.all = True
    
    if args.all or args.modes:
        demo_attention_modes()
    
    if args.all or args.compression:
        demo_compression_modes()
    
    if args.all or args.memory:
        demo_memory_calculation()
    
    if args.all or args.quantization:
        demo_quantization()
    
    if args.all or args.recommendations:
        demo_recommendations()
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
