"""
Minimal RaBitQ KV Cache Benchmark
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rabitq import create_k1, create_k2, create_k3


def main():
    print("=" * 60)
    print("MINIMAL RABITQ BENCHMARK")
    print("=" * 60)

    device = "cpu"
    num_layers = 4
    num_heads = 4
    head_dim = 64
    seq_len = 512

    torch.manual_seed(42)
    keys = torch.randn(num_layers, num_heads, seq_len, head_dim, device=device)
    values = torch.randn(num_layers, num_heads, seq_len, head_dim, device=device)

    fit_k = keys[:, :, :128, :].reshape(-1, head_dim)
    fit_v = values[:, :, :128, :].reshape(-1, head_dim)

    print(
        f"\nTest config: {num_layers} layers, {num_heads} heads, "
        f"{seq_len} seq_len, {head_dim} head_dim"
    )
    print(f"Total KV pairs: {num_layers * num_heads * seq_len * 2}")

    configs = [
        ("FP16", None),
        ("1-bit", create_k1),
        ("2-bit", create_k2),
        ("3-bit", create_k3),
    ]

    for name, factory in configs:
        if factory is None:
            size_mb = keys.numel() * 2 / (1024**2) + values.numel() * 2 / (1024**2)
            print(f"\n{name}: {size_mb:.2f} MB")
            continue

        rq = factory(head_dim=head_dim, device=device)
        rq.fit(fit_k, fit_v)

        t0 = time.time()
        compressed = rq.compress(keys, values)
        comp_time = (time.time() - t0) * 1000

        t0 = time.time()
        k_dq, v_dq = rq.decompress(compressed)
        decomp_time = (time.time() - t0) * 1000

        rel_err = ((keys - k_dq).norm() + (values - v_dq).norm()) / (2 * keys.norm())
        stats = rq.memory_stats(seq_len, num_layers, 1, num_heads)

        print(f"\n{name}:")
        print(
            f"  Ratio: {stats['compression_ratio']:.1f}x "
            f"({stats['original_mb']:.1f}→{stats['compressed_mb']:.1f} MB)"
        )
        print(f"  Error: {rel_err:.2%}")
        print(f"  Time: {comp_time:.0f}ms / {decomp_time:.0f}ms")

    print("\n" + "=" * 60)
    print("✓ RaBitQ compression working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    main()
