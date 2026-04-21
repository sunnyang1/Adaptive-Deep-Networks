"""
End-to-End Benchmark for RaBitQ KV Cache Compression

Tests compression ratio, reconstruction quality, and throughput
for various configurations and sequence lengths.
"""

import torch
import time
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rabitq import (
    RaBitQ,
    RaBitQConfig,
    create_k1,
    create_k2,
    create_k3,
    quantize_vector,
    RabitqConfig,
    reconstruct_vector,
    RaBitQCache,
)
from src.models.configs import ModelConfig
from src.models.adaptive_transformer import AdaptiveTransformer


@dataclass
class BenchmarkResult:
    config_name: str
    total_bits: int
    seq_len: int
    num_layers: int
    num_heads: int
    head_dim: int

    # Compression metrics
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float

    # Quality metrics
    max_abs_error: float
    mean_abs_error: float
    relative_error: float

    # Performance metrics
    compress_time_ms: float
    decompress_time_ms: float
    throughput_tokens_per_sec: float

    # Memory metrics
    peak_memory_mb: float


def measure_memory():
    """Measure current GPU/CPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0


def run_compression_benchmark(
    keys: torch.Tensor,
    values: torch.Tensor,
    rq: RaBitQ,
    config_name: str,
    num_warmup: int = 3,
    num_trials: int = 10,
) -> BenchmarkResult:
    """
    Run benchmark for a single configuration.

    Args:
        keys: [num_layers, num_heads, seq_len, head_dim]
        values: [num_layers, num_heads, seq_len, head_dim]
        rq: RaBitQ instance
        config_name: name of the configuration
    """
    num_layers, num_heads, seq_len, head_dim = keys.shape

    # Warmup
    for _ in range(num_warmup):
        compressed = rq.compress(keys, values)
        _ = rq.decompress(compressed)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark compression
    compress_times = []
    decompress_times = []

    for _ in range(num_trials):
        # Compression
        t0 = time.perf_counter()
        compressed = rq.compress(keys, values)
        t1 = time.perf_counter()
        compress_times.append((t1 - t0) * 1000)  # ms

        # Decompression
        t0 = time.perf_counter()
        keys_dq, values_dq = rq.decompress(compressed)
        t1 = time.perf_counter()
        decompress_times.append((t1 - t0) * 1000)  # ms

    # Compute quality metrics
    max_abs_error = max(
        (keys - keys_dq).abs().max().item(), (values - values_dq).abs().max().item()
    )
    mean_abs_error = (
        (keys - keys_dq).abs().mean().item() + (values - values_dq).abs().mean().item()
    ) / 2

    keys_norm = keys.norm().item()
    relative_error = ((keys - keys_dq).norm().item() + (values - values_dq).norm().item()) / (
        2 * keys_norm
    )

    # Memory stats
    memory_stats = rq.memory_stats(seq_len, num_layers, batch_size=1, num_heads=num_heads)
    peak_memory = measure_memory()

    # Throughput
    total_tokens = num_layers * num_heads * seq_len * 2  # K + V
    avg_compress_time = sum(compress_times) / len(compress_times)
    throughput = total_tokens / (avg_compress_time / 1000)

    return BenchmarkResult(
        config_name=config_name,
        total_bits=rq.config.total_bits,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        original_size_mb=memory_stats["original_mb"],
        compressed_size_mb=memory_stats["compressed_mb"],
        compression_ratio=memory_stats["compression_ratio"],
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        relative_error=relative_error,
        compress_time_ms=avg_compress_time,
        decompress_time_ms=sum(decompress_times) / len(decompress_times),
        throughput_tokens_per_sec=throughput,
        peak_memory_mb=peak_memory,
    )


def benchmark_standalone_compression():
    """Benchmark standalone RaBitQ compression (no transformer)."""
    print("=" * 80)
    print("STANDALONE KV CACHE COMPRESSION BENCHMARK")
    print("=" * 80)

    configs = [
        ("FP16 Baseline", None),
        ("RaBitQ 1-bit", create_k1),
        ("RaBitQ 2-bit", create_k2),
        ("RaBitQ 3-bit", create_k3),
    ]

    # Test different sequence lengths (reduced for macOS testing)
    seq_lengths = [128, 512, 2048]

    # Fixed model config (Small model)
    num_layers = 32
    num_heads = 8
    head_dim = 64

    results: List[BenchmarkResult] = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")

        # Generate synthetic KV cache
        torch.manual_seed(42)
        keys = torch.randn(num_layers, num_heads, seq_len, head_dim, device=device)
        values = torch.randn(num_layers, num_heads, seq_len, head_dim, device=device)

        # Fit on subset
        fit_size = min(1024, seq_len)
        rq_fit_k = keys[:, :, :fit_size, :].reshape(-1, head_dim)
        rq_fit_v = values[:, :, :fit_size, :].reshape(-1, head_dim)

        for config_name, factory in configs:
            if factory is None:
                # FP16 baseline
                original_size = (keys.numel() + values.numel()) * 2 / (1024**2)
                print(f"  {config_name}: {original_size:.2f} MB (no compression)")
                continue

            rq = factory(head_dim=head_dim, device=device)
            rq.fit(rq_fit_k, rq_fit_v)

            result = run_compression_benchmark(keys, values, rq, config_name)
            results.append(result)

            print(f"  {config_name}:")
            print(
                f"    Compression: {result.compression_ratio:.2f}x "
                f"({result.original_size_mb:.2f}MB → {result.compressed_size_mb:.2f}MB)"
            )
            print(
                f"    Quality: max_error={result.max_abs_error:.4f}, "
                f"rel_error={result.relative_error:.2%}"
            )
            print(
                f"    Speed: compress={result.compress_time_ms:.1f}ms, "
                f"decompress={result.decompress_time_ms:.1f}ms"
            )
            print(f"    Throughput: {result.throughput_tokens_per_sec:.0f} tokens/sec")

    return results


def benchmark_transformer_integration():
    """Benchmark RaBitQ integrated with AdaptiveTransformer."""
    print("\n" + "=" * 80)
    print("ADAPTIVE TRANSFORMER + RABITQ INTEGRATION BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Small model config for testing
    config = ModelConfig(
        num_layers=8, hidden_dim=512, num_heads=8, num_blocks=4, vocab_size=1000, mlp_ratio=4
    )

    model = AdaptiveTransformer(config).to(device)
    model.eval()

    seq_lengths = [128, 512, 2048]

    results = []

    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)

        # Baseline: FP16 forward
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids, use_rabitq=False)
        t1 = time.perf_counter()
        baseline_time = (t1 - t0) * 1000
        print(f"  FP16 forward: {baseline_time:.1f}ms")

        # RaBitQ forward
        model.init_rabitq_caches(total_bits=1, residual_window=128)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids, use_rabitq=True)
        t1 = time.perf_counter()
        rabitq_time = (t1 - t0) * 1000
        print(
            f"  RaBitQ 1-bit forward: {rabitq_time:.1f}ms "
            f"({baseline_time/rabitq_time:.2f}x of FP16)"
        )

        # Generation test
        prompt_len = 32
        max_new_tokens = 16
        prompt_ids = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(prompt_ids, max_new_tokens=max_new_tokens, use_rabitq=True)
        t1 = time.perf_counter()
        gen_time = t1 - t0

        tokens_per_sec = max_new_tokens / gen_time
        print(
            f"  Generation: {tokens_per_sec:.1f} tokens/sec "
            f"({max_new_tokens} tokens in {gen_time:.2f}s)"
        )

        results.append(
            {
                "seq_len": seq_len,
                "baseline_ms": baseline_time,
                "rabitq_ms": rabitq_time,
                "gen_tokens_per_sec": tokens_per_sec,
            }
        )

    return results


def benchmark_rabitq_cache():
    """Benchmark RaBitQCache HF-compatible cache."""
    print("\n" + "=" * 80)
    print("RABITQ CACHE (HF-COMPATIBLE) BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = [
        ("1-bit, window=128", 1, 128),
        ("1-bit, window=512", 1, 512),
        ("2-bit, window=128", 2, 128),
        ("3-bit, window=128", 3, 128),
    ]

    batch_size = 1
    num_heads = 8
    head_dim = 64
    seq_len = 8192
    num_layers = 8

    results = []

    for config_name, total_bits, window in configs:
        cache = RaBitQCache(
            total_bits=total_bits, head_dim=head_dim, residual_window=window, device=device
        )

        # Simulate incremental cache updates
        chunk_size = 128
        num_chunks = seq_len // chunk_size

        compress_times = []

        t0 = time.perf_counter()
        for chunk_idx in range(num_chunks):
            k = torch.randn(batch_size, num_heads, chunk_size, head_dim, device=device)
            v = torch.randn(batch_size, num_heads, chunk_size, head_dim, device=device)

            t_c0 = time.perf_counter()
            full_k, full_v = cache.update(k, v, layer_idx=0)
            t_c1 = time.perf_counter()
            compress_times.append((t_c1 - t_c0) * 1000)

        total_time = (time.perf_counter() - t0) * 1000
        avg_time = sum(compress_times) / len(compress_times)

        # Memory stats
        memory_saved = (1 - (cache.get_seq_length(0) / seq_len)) * 100

        print(f"  {config_name}:")
        print(f"    Total time: {total_time:.1f}ms")
        print(f"    Avg per chunk: {avg_time:.2f}ms")
        print(f"    Cache length: {cache.get_seq_length(0)}")

        results.append(
            {
                "config": config_name,
                "total_time_ms": total_time,
                "avg_chunk_ms": avg_time,
                "cache_len": cache.get_seq_length(0),
            }
        )

    return results


def print_summary_table(results: List[BenchmarkResult]):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'Config':<15} {'SeqLen':<8} {'Ratio':<8} {'MaxErr':<8} "
        f"{'Cmp(ms)':<8} {'Dcmp(ms)':<8} {'Thrput':<10}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r.config_name:<15} {r.seq_len:<8} {r.compression_ratio:<8.1f}x "
            f"{r.max_abs_error:<8.4f} {r.compress_time_ms:<8.1f} "
            f"{r.decompress_time_ms:<8.1f} {r.throughput_tokens_per_sec:<10.0f}"
        )


def main():
    print("RaBitQ KV Cache Compression Benchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = {}

    # Run benchmarks
    try:
        standalone_results = benchmark_standalone_compression()
        all_results["standalone"] = standalone_results
        print_summary_table(standalone_results)
    except Exception as e:
        print(f"Standalone benchmark failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        transformer_results = benchmark_transformer_integration()
        all_results["transformer"] = transformer_results
    except Exception as e:
        print(f"Transformer benchmark failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        cache_results = benchmark_rabitq_cache()
        all_results["cache"] = cache_results
    except Exception as e:
        print(f"Cache benchmark failed: {e}")
        import traceback

        traceback.print_exc()

    # Save results
    output_path = Path("results/rabitq_benchmark.json")
    output_path.parent.mkdir(exist_ok=True)

    # Convert to serializable format
    serializable = {}
    for key, val in all_results.items():
        if key == "standalone":
            serializable[key] = [
                {
                    "config_name": r.config_name,
                    "total_bits": r.total_bits,
                    "seq_len": r.seq_len,
                    "compression_ratio": r.compression_ratio,
                    "max_abs_error": r.max_abs_error,
                    "mean_abs_error": r.mean_abs_error,
                    "relative_error": r.relative_error,
                    "compress_time_ms": r.compress_time_ms,
                    "decompress_time_ms": r.decompress_time_ms,
                    "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
                }
                for r in val
            ]
        else:
            serializable[key] = val

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if "standalone" in all_results:
        results = all_results["standalone"]
        k1_results = [r for r in results if r.total_bits == 1]
        if k1_results:
            avg_ratio = sum(r.compression_ratio for r in k1_results) / len(k1_results)
            avg_error = sum(r.relative_error for r in k1_results) / len(k1_results)
            print(f"✓ RaBitQ 1-bit achieves {avg_ratio:.1f}x compression")
            print(f"✓ Average relative error: {avg_error:.2%}")
            print(f"✓ Throughput: {k1_results[-1].throughput_tokens_per_sec:.0f} tokens/sec")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
