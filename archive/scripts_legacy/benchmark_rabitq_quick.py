"""
Quick Benchmark for RaBitQ KV Cache Compression
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rabitq import create_k1, create_k2, create_k3, RaBitQCache
from src.models.configs import ModelConfig
from src.models.adaptive_transformer import AdaptiveTransformer


def quick_standalone_benchmark():
    """Quick standalone compression test."""
    print("=" * 70)
    print("STANDALONE KV CACHE COMPRESSION")
    print("=" * 70)

    device = "cpu"  # Use CPU for reliability
    num_layers = 8
    num_heads = 8
    head_dim = 64
    seq_len = 1024

    # Generate synthetic KV cache
    torch.manual_seed(42)
    keys = torch.randn(num_layers, num_heads, seq_len, head_dim, device=device)
    values = torch.randn(num_layers, num_heads, seq_len, head_dim, device=device)

    fit_k = keys[:, :, :256, :].reshape(-1, head_dim)
    fit_v = values[:, :, :256, :].reshape(-1, head_dim)

    configs = [
        ("FP16 Baseline", None),
        ("RaBitQ 1-bit", create_k1),
        ("RaBitQ 2-bit", create_k2),
        ("RaBitQ 3-bit", create_k3),
    ]

    results = []

    for name, factory in configs:
        if factory is None:
            original_size = (keys.numel() + values.numel()) * 2 / (1024**2)
            print(f"\n{name}:")
            print(f"  Size: {original_size:.2f} MB (uncompressed)")
            results.append({"name": name, "ratio": 1.0, "size_mb": original_size, "error": 0.0})
            continue

        rq = factory(head_dim=head_dim, device=device)
        rq.fit(fit_k, fit_v)

        # Compress
        t0 = time.perf_counter()
        compressed = rq.compress(keys, values)
        compress_time = (time.perf_counter() - t0) * 1000

        # Decompress
        t0 = time.perf_counter()
        keys_dq, values_dq = rq.decompress(compressed)
        decompress_time = (time.perf_counter() - t0) * 1000

        # Error metrics
        max_error = max(
            (keys - keys_dq).abs().max().item(), (values - values_dq).abs().max().item()
        )
        rel_error = ((keys - keys_dq).norm().item() + (values - values_dq).norm().item()) / (
            2 * keys.norm().item()
        )

        stats = rq.memory_stats(seq_len, num_layers, 1, num_heads)

        print(f"\n{name}:")
        print(
            f"  Compression: {stats['compression_ratio']:.2f}x "
            f"({stats['original_mb']:.2f}MB → {stats['compressed_mb']:.2f}MB)"
        )
        print(f"  Max error: {max_error:.4f}, Relative error: {rel_error:.2%}")
        print(f"  Compress: {compress_time:.1f}ms, Decompress: {decompress_time:.1f}ms")

        results.append(
            {
                "name": name,
                "ratio": stats["compression_ratio"],
                "size_mb": stats["compressed_mb"],
                "error": rel_error,
                "compress_ms": compress_time,
                "decompress_ms": decompress_time,
            }
        )

    return results


def quick_cache_benchmark():
    """Quick cache benchmark."""
    print("\n" + "=" * 70)
    print("RABITQ CACHE BENCHMARK")
    print("=" * 70)

    device = "cpu"
    cache_1bit = RaBitQCache(total_bits=1, head_dim=64, residual_window=128, device=device)

    batch_size, num_heads, head_dim = 1, 8, 64

    # Simulate generation
    prompt_len = 256
    gen_tokens = 64

    # Initial prompt
    k_prompt = torch.randn(batch_size, num_heads, prompt_len, head_dim, device=device)
    v_prompt = torch.randn(batch_size, num_heads, prompt_len, head_dim, device=device)

    t0 = time.perf_counter()
    full_k, full_v = cache_1bit.update(k_prompt, v_prompt, layer_idx=0)
    init_time = (time.perf_counter() - t0) * 1000

    # Incremental generation
    gen_times = []
    for i in range(gen_tokens):
        k_new = torch.randn(batch_size, num_heads, 1, head_dim, device=device)
        v_new = torch.randn(batch_size, num_heads, 1, head_dim, device=device)

        t0 = time.perf_counter()
        full_k, full_v = cache_1bit.update(k_new, v_new, layer_idx=0)
        gen_times.append((time.perf_counter() - t0) * 1000)

    avg_gen_time = sum(gen_times) / len(gen_times)

    print(f"\nRaBitQ 1-bit Cache:")
    print(f"  Prompt processing ({prompt_len} tokens): {init_time:.1f}ms")
    print(f"  Per-token generation: {avg_gen_time:.2f}ms")
    print(f"  Generation throughput: {1000/avg_gen_time:.1f} tokens/sec")
    print(f"  Final cache length: {cache_1bit.get_seq_length(0)}")


def quick_transformer_benchmark():
    """Quick transformer benchmark."""
    print("\n" + "=" * 70)
    print("ADAPTIVE TRANSFORMER + RABITQ")
    print("=" * 70)

    device = "cpu"

    config = ModelConfig(num_layers=4, hidden_dim=256, num_heads=4, num_blocks=2, vocab_size=1000)

    model = AdaptiveTransformer(config).to(device)
    model.eval()

    seq_lens = [64, 128, 256]

    for seq_len in seq_lens:
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)

        # Baseline
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids, use_rabitq=False)
        baseline_ms = (time.perf_counter() - t0) * 1000

        # With RaBitQ
        model.init_rabitq_caches(total_bits=1, residual_window=64)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids, use_rabitq=True)
        rabitq_ms = (time.perf_counter() - t0) * 1000

        overhead = (rabitq_ms / baseline_ms - 1) * 100

        print(f"\nSeqLen={seq_len}:")
        print(f"  FP16: {baseline_ms:.1f}ms")
        print(f"  RaBitQ 1-bit: {rabitq_ms:.1f}ms ({overhead:+.1f}% overhead)")


def main():
    print("RaBitQ KV Cache Compression - Quick Benchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: CPU (macOS compatible)")

    try:
        quick_standalone_benchmark()
    except Exception as e:
        print(f"Standalone benchmark error: {e}")

    try:
        quick_cache_benchmark()
    except Exception as e:
        print(f"Cache benchmark error: {e}")

    try:
        quick_transformer_benchmark()
    except Exception as e:
        print(f"Transformer benchmark error: {e}")

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
