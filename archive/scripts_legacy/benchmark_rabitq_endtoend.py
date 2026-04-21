"""
End-to-End RaBitQ KV Cache Compression Benchmark

Tests the full pipeline with quality verification.
"""

import torch
import time
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rabitq import (
    RaBitQ,
    create_k1,
    create_k2,
    create_k3,
    quantize_vector,
    RabitqConfig,
    reconstruct_vector,
)
from src.rabitq.estimator import make_full_single_query, full_est_dist


def test_compression_quality():
    """Test compression quality on synthetic data."""
    print("=" * 70)
    print("COMPRESSION QUALITY TEST")
    print("=" * 70)

    dim = 64
    num_vectors = 100

    # Generate test data with different distributions
    torch.manual_seed(42)
    test_cases = [
        ("Normal", torch.randn(num_vectors, dim)),
        ("Uniform", torch.rand(num_vectors, dim) * 2 - 1),
        ("Sparse", torch.randn(num_vectors, dim) * (torch.rand(num_vectors, dim) > 0.8).float()),
    ]

    configs = [
        ("1-bit", 1),
        ("2-bit", 2),
        ("3-bit", 3),
    ]

    results = []

    for dist_name, data in test_cases:
        print(f"\nDistribution: {dist_name}")
        centroid = data.mean(dim=0)

        for config_name, total_bits in configs:
            config = RabitqConfig(total_bits=total_bits, metric_type="ip")

            # Quantize all vectors
            reconstructed = []
            for i in range(min(20, num_vectors)):  # Sample for speed
                qv = quantize_vector(data[i], centroid, config)
                recon = reconstruct_vector(centroid, qv)
                reconstructed.append(recon)

            reconstructed = torch.stack(reconstructed)
            sample_data = data[: len(reconstructed)]

            # Quality metrics
            abs_errors = (sample_data - reconstructed).abs()
            mae = abs_errors.mean().item()
            max_err = abs_errors.max().item()
            rel_err = (sample_data - reconstructed).norm() / sample_data.norm()

            # Cosine similarity preservation
            orig_sim = (
                torch.nn.functional.cosine_similarity(sample_data[:-1], sample_data[1:], dim=-1)
                .mean()
                .item()
            )
            recon_sim = (
                torch.nn.functional.cosine_similarity(reconstructed[:-1], reconstructed[1:], dim=-1)
                .mean()
                .item()
            )
            sim_preserved = abs(orig_sim - recon_sim) < 0.1

            print(
                f"  {config_name}: MAE={mae:.4f}, Max={max_err:.4f}, "
                f"Rel={rel_err:.2%}, SimPreserved={sim_preserved}"
            )

            results.append(
                {
                    "distribution": dist_name,
                    "bits": total_bits,
                    "mae": mae,
                    "max_error": max_err,
                    "relative_error": rel_err.item(),
                    "similarity_preserved": sim_preserved,
                }
            )

    return results


def test_kv_cache_pipeline():
    """Test full KV cache compression pipeline."""
    print("\n" + "=" * 70)
    print("KV CACHE PIPELINE TEST")
    print("=" * 70)

    device = "cpu"
    batch_size = 1
    num_layers = 8
    num_heads = 8
    head_dim = 64
    seq_len = 256

    print(f"\nConfig: {num_layers}L x {num_heads}H x {seq_len}T x {head_dim}D")

    # Simulate KV cache from transformer
    torch.manual_seed(42)
    keys = torch.randn(num_layers, num_heads, seq_len, head_dim, device=device)
    values = torch.randn(num_layers, num_heads, seq_len, head_dim, device=device)

    # Test configurations
    configs = [
        ("RaBitQ-1", create_k1),
        ("RaBitQ-2", create_k2),
        ("RaBitQ-3", create_k3),
    ]

    results = []

    for name, factory in configs:
        rq = factory(head_dim=head_dim, device=device)

        # Fit on subset
        fit_k = keys[:, :, :128, :].reshape(-1, head_dim)
        fit_v = values[:, :, :128, :].reshape(-1, head_dim)
        rq.fit(fit_k, fit_v)

        # Full compression
        t0 = time.time()
        compressed = rq.compress(keys, values)
        compress_time = time.time() - t0

        # Decompression
        t0 = time.time()
        keys_dq, values_dq = rq.decompress(compressed)
        decompress_time = time.time() - t0

        # Quality
        k_error = (keys - keys_dq).abs().mean().item()
        v_error = (values - values_dq).abs().mean().item()

        # Memory stats
        stats = rq.memory_stats(seq_len, num_layers, batch_size, num_heads)

        # Attention simulation - check if compressed KV gives similar attention pattern
        # Simulate one attention head
        query = torch.randn(1, head_dim)

        # Original attention
        scores_orig = torch.matmul(query, keys[0, 0].T) / (head_dim**0.5)
        attn_orig = torch.softmax(scores_orig, dim=-1)

        # Compressed attention
        scores_comp = torch.matmul(query, keys_dq[0, 0].T) / (head_dim**0.5)
        attn_comp = torch.softmax(scores_comp, dim=-1)

        attn_diff = (attn_orig - attn_comp).abs().mean().item()

        print(f"\n{name}:")
        print(f"  Compression: {stats['compression_ratio']:.1f}x")
        print(f"  Size: {stats['original_mb']:.2f}MB → {stats['compressed_mb']:.2f}MB")
        print(f"  KV Error: K={k_error:.4f}, V={v_error:.4f}")
        print(f"  Attention diff: {attn_diff:.4f}")
        print(f"  Time: {compress_time:.2f}s compress, {decompress_time:.2f}s decompress")

        results.append(
            {
                "name": name,
                "compression_ratio": stats["compression_ratio"],
                "original_mb": stats["original_mb"],
                "compressed_mb": stats["compressed_mb"],
                "k_error": k_error,
                "v_error": v_error,
                "attention_diff": attn_diff,
                "compress_time": compress_time,
                "decompress_time": decompress_time,
            }
        )

    return results


def test_estimator_accuracy():
    """Test distance estimator accuracy."""
    print("\n" + "=" * 70)
    print("DISTANCE ESTIMATOR ACCURACY")
    print("=" * 70)

    dim = 64
    num_queries = 20
    num_data = 20

    torch.manual_seed(42)
    queries = torch.randn(num_queries, dim)
    data = torch.randn(num_data, dim)
    centroid = data.mean(dim=0)

    configs = [
        ("1-bit IP", 1, "ip"),
        ("2-bit IP", 2, "ip"),
        ("3-bit IP", 3, "ip"),
        ("1-bit L2", 1, "l2"),
    ]

    results = []

    for name, total_bits, metric_type in configs:
        config = RabitqConfig(total_bits=total_bits, metric_type=metric_type)

        # Quantize data
        quantized = []
        for i in range(num_data):
            qv = quantize_vector(data[i], centroid, config)
            quantized.append(qv)

        # Test estimation vs ground truth
        errors = []
        for q_idx in range(min(10, num_queries)):
            query = queries[q_idx]
            q_obj = make_full_single_query(query, centroid, metric_type, total_bits)

            for d_idx in range(min(10, num_data)):
                # True distance
                if metric_type == "ip":
                    true_dist = -torch.dot(query, data[d_idx]).item()
                else:
                    true_dist = ((query - data[d_idx]) ** 2).sum().item()

                # Estimated distance
                qv = quantized[d_idx]
                from src.rabitq.packing import unpack_binary_code

                bin_code = unpack_binary_code(qv.binary_code_packed, qv.dim)
                ip = float((q_obj.rotated_query * bin_code.float()).sum().item())
                est_dist, _ = full_est_dist(ip, qv.f_add, qv.f_rescale, q_obj)

                rel_err = abs(est_dist - true_dist) / (abs(true_dist) + 1e-6)
                errors.append(rel_err)

        avg_error = sum(errors) / len(errors)
        max_error = max(errors)

        print(f"  {name}: AvgRelErr={avg_error:.2%}, MaxRelErr={max_error:.2%}")

        results.append(
            {"name": name, "avg_relative_error": avg_error, "max_relative_error": max_error}
        )

    return results


def main():
    print("\n" + "=" * 70)
    print("END-TO-END RABITQ BENCHMARK")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: CPU")

    all_results = {}

    try:
        all_results["quality"] = test_compression_quality()
    except Exception as e:
        print(f"Quality test error: {e}")
        import traceback

        traceback.print_exc()

    try:
        all_results["kv_pipeline"] = test_kv_cache_pipeline()
    except Exception as e:
        print(f"KV pipeline test error: {e}")
        import traceback

        traceback.print_exc()

    try:
        all_results["estimator"] = test_estimator_accuracy()
    except Exception as e:
        print(f"Estimator test error: {e}")
        import traceback

        traceback.print_exc()

    # Save results
    output_path = Path("results/rabitq_endtoend_benchmark.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if "kv_pipeline" in all_results and all_results["kv_pipeline"]:
        for r in all_results["kv_pipeline"]:
            print(
                f"• {r['name']}: {r['compression_ratio']:.1f}x compression, "
                f"attention diff {r['attention_diff']:.4f}"
            )

    print("\n✓ End-to-end benchmark complete!")


if __name__ == "__main__":
    main()
