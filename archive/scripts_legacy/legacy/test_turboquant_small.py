#!/usr/bin/env python3
"""
TurboQuant Tests on Small Model

Based on Adaptive_Deep_Networks_TurboQuant.md paper:
1. Compression ratio validation (target: 6×+ memory reduction)
2. KV Cache compression efficiency (target: 5.7× reduction)
3. Attention accuracy preservation (target: zero accuracy loss)
4. Throughput improvement measurement
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torch.nn as nn
import json
import time
import numpy as np
from datetime import datetime

from models.configs import get_config
from models.adaptive_transformer import AdaptiveTransformer
from turboquant import PolarQuant, QJLCompressor


class TurboQuantTester:
    """Test suite for TurboQuant on Small Model."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.config = model.config
        self.results = {}

        # Use head_dim for per-head compression
        self.head_dim = self.config.hidden_dim // self.config.num_heads  # 64 for small model
        self.angle_bits = 3  # 3-bit for angles, 1-bit for QJL = 4-bit total
        self.qjl_proj_dim = 64  # Projection dimension for QJL

    def test_1_compression_ratio(self):
        """Test 1: Validate compression ratio of TurboQuant components."""
        print("\n" + "=" * 70)
        print("TEST 1: Compression Ratio Analysis")
        print("=" * 70)

        results = {"polar_quant": {}, "turboquant_full": {}, "theoretical": {}}

        # Test PolarQuant compression on head_dim
        dim = self.head_dim  # 64 for small model
        pq = PolarQuant(dim, angle_bits=self.angle_bits, device=self.device)

        polar_ratio = pq.get_compression_ratio()
        results["polar_quant"]["compression_ratio"] = polar_ratio
        results["polar_quant"]["original_bits"] = dim * 16  # FP16
        results["polar_quant"]["compressed_bits"] = 16 + (dim - 1) * self.angle_bits
        results["polar_quant"]["target_dim"] = dim

        print(f"\nPolarQuant ({self.angle_bits}-bit angles) on head_dim={dim}:")
        print(f"  Original: {dim * 16} bits")
        print(
            f"  Compressed: 16 + {dim-1}×{self.angle_bits} = {results['polar_quant']['compressed_bits']} bits"
        )
        print(f"  Compression ratio: {polar_ratio:.2f}×")

        # Full TurboQuant (PolarQuant + QJL)
        original_bits = dim * 16  # FP16
        polar_bits = 16 + (dim - 1) * self.angle_bits
        qjl_bits = self.qjl_proj_dim  # 1-bit signs

        turbo_ratio = original_bits / (polar_bits + qjl_bits)
        results["turboquant_full"]["compression_ratio"] = turbo_ratio
        results["turboquant_full"]["original_bits"] = original_bits
        results["turboquant_full"]["compressed_bits"] = polar_bits + qjl_bits
        results["turboquant_full"]["breakdown"] = {
            "polar_radius": 16,
            "polar_angles": (dim - 1) * self.angle_bits,
            "qjl_signs": self.qjl_proj_dim,
        }

        print(f"\nFull TurboQuant (PolarQuant + QJL):")
        print(f"  Original: {original_bits} bits")
        print(f"  Compressed: {polar_bits} + {qjl_bits} = {polar_bits + qjl_bits} bits")
        print(f"  Compression ratio: {turbo_ratio:.2f}×")
        print(f"  Paper target: 6×+ memory reduction")
        print(f"  Status: {'✅ PASS' if turbo_ratio >= 6.0 else '⚠️  BELOW TARGET'}")

        # Theoretical analysis
        results["theoretical"] = {
            "fp16_size_bytes": dim * 2,
            "compressed_size_bytes": (polar_bits + qjl_bits) / 8,
            "memory_reduction_factor": turbo_ratio,
        }

        self.results["compression_ratio"] = results
        return results

    def test_2_kv_cache_compression(self):
        """Test 2: KV Cache compression efficiency."""
        print("\n" + "=" * 70)
        print("TEST 2: KV Cache Compression")
        print("=" * 70)

        # Simulate KV cache for different sequence lengths
        seq_lengths = [1024, 2048, 4096]
        batch_size = 1
        num_heads = self.config.num_heads
        head_dim = self.head_dim

        results = {
            "by_sequence_length": [],
            "target_reduction": 5.7,  # From paper: 5.7× reduction (16GB -> 2.8GB)
        }

        print(f"\nKV Cache Compression Analysis:")
        print(f"  Batch size: {batch_size}")
        print(f"  Num heads: {num_heads}")
        print(f"  Head dim: {head_dim}")

        # Bits per element
        polar_bits = 16 + (head_dim - 1) * self.angle_bits
        qjl_bits = self.qjl_proj_dim
        bits_per_element = polar_bits + qjl_bits

        for seq_len in seq_lengths:
            # Original KV cache size (FP16)
            elements = 2 * batch_size * num_heads * seq_len  # K + V elements
            kv_original_bytes = elements * head_dim * 2  # FP16 = 2 bytes

            # Compressed KV cache (TurboQuant)
            kv_compressed_bytes = elements * bits_per_element / 8

            reduction = kv_original_bytes / kv_compressed_bytes

            result = {
                "seq_len": seq_len,
                "original_mb": kv_original_bytes / (1024**2),
                "compressed_mb": kv_compressed_bytes / (1024**2),
                "reduction_factor": reduction,
            }
            results["by_sequence_length"].append(result)

            print(f"\n  Sequence length: {seq_len}")
            print(f"    Original: {result['original_mb']:.2f} MB")
            print(f"    Compressed: {result['compressed_mb']:.2f} MB")
            print(f"    Reduction: {reduction:.2f}×")

        avg_reduction = np.mean([r["reduction_factor"] for r in results["by_sequence_length"]])
        results["average_reduction"] = avg_reduction

        print(f"\n  Average reduction: {avg_reduction:.2f}×")
        print(f"  Paper target: 5.7×")
        print(f"  Status: {'✅ PASS' if avg_reduction >= 5.0 else '⚠️  BELOW TARGET'}")

        self.results["kv_cache_compression"] = results
        return results

    def test_3_attention_accuracy(self):
        """Test 3: Attention accuracy preservation after compression."""
        print("\n" + "=" * 70)
        print("TEST 3: Attention Accuracy Preservation")
        print("=" * 70)

        batch_size = 1
        num_heads = self.config.num_heads
        head_dim = self.head_dim
        seq_len = 128  # Shorter for testing

        print(f"\nGenerating test tensors...")
        print(f"  Shape: [{batch_size}, {num_heads}, {seq_len}, {head_dim}]")

        # Generate test tensors
        torch.manual_seed(42)
        queries = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)

        # Compute reference attention (full precision)
        print("Computing full-precision attention...")
        with torch.no_grad():
            scores_ref = torch.matmul(queries, keys.transpose(-2, -1)) / (head_dim**0.5)
            attn_weights_ref = torch.softmax(scores_ref, dim=-1)
            output_ref = torch.matmul(attn_weights_ref, values)

        # Compress with PolarQuant only (simpler test)
        print("Compressing keys with PolarQuant...")
        pq = PolarQuant(head_dim, angle_bits=self.angle_bits, device=self.device)

        # Reshape for compression: [B, H, T, d] -> [B*H*T, d]
        keys_flat = keys.reshape(-1, head_dim)
        values_flat = values.reshape(-1, head_dim)

        # Compress and decompress keys
        r_k, theta_k, _ = pq.compress(keys_flat)
        keys_compressed = pq.decompress(r_k, theta_k).reshape_as(keys)

        # Compress and decompress values
        r_v, theta_v, _ = pq.compress(values_flat)
        values_compressed = pq.decompress(r_v, theta_v).reshape_as(values)

        # Compute attention with compressed KV
        print("Computing attention with compressed KV...")
        with torch.no_grad():
            scores_comp = torch.matmul(queries, keys_compressed.transpose(-2, -1)) / (head_dim**0.5)
            attn_weights_comp = torch.softmax(scores_comp, dim=-1)
            output_comp = torch.matmul(attn_weights_comp, values_compressed)

        # Compute metrics
        mse = torch.mean((output_ref - output_comp) ** 2).item()
        relative_error = torch.norm(output_ref - output_comp) / torch.norm(output_ref)
        relative_error = relative_error.item()

        # Cosine similarity
        cos_sim = nn.CosineSimilarity(dim=-1)
        cos_sim_mean = cos_sim(output_ref.flatten(0, -2), output_comp.flatten(0, -2)).mean().item()

        results = {
            "mse": mse,
            "relative_error": relative_error,
            "cosine_similarity": cos_sim_mean,
            "shape": list(output_ref.shape),
            "target_relative_error": 0.05,  # 5% relative error threshold
        }

        print(f"\nAccuracy Metrics:")
        print(f"  MSE: {mse:.6f}")
        print(f"  Relative error: {relative_error:.4f} ({relative_error*100:.2f}%)")
        print(f"  Cosine similarity: {cos_sim_mean:.4f}")
        print(f"  Target relative error: < 5%")
        print(f"  Status: {'✅ PASS' if relative_error < 0.05 else '⚠️  HIGH ERROR'}")

        self.results["attention_accuracy"] = results
        return results

    def test_4_throughput_comparison(self):
        """Test 4: Throughput comparison (compressed vs full precision)."""
        print("\n" + "=" * 70)
        print("TEST 4: Throughput Comparison")
        print("=" * 70)

        batch_size = 1
        num_heads = self.config.num_heads
        head_dim = self.head_dim
        seq_len = 256
        num_runs = 5

        print(f"\nTest configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Num runs: {num_runs}")

        # Generate test tensors
        torch.manual_seed(42)
        queries = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)

        # Warmup
        with torch.no_grad():
            _ = torch.matmul(queries, keys.transpose(-2, -1))

        # Full precision attention timing
        print("\nTiming full-precision attention...")
        times_fp = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                scores = torch.matmul(queries, keys.transpose(-2, -1)) / (head_dim**0.5)
                attn = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn, values)
            if self.device.type == "mps":
                torch.mps.synchronize()
            times_fp.append(time.time() - start)

        avg_time_fp = np.mean(times_fp)

        # Compress keys/values
        pq = PolarQuant(head_dim, angle_bits=self.angle_bits, device=self.device)
        keys_flat = keys.reshape(-1, head_dim)
        values_flat = values.reshape(-1, head_dim)

        r_k, theta_k, _ = pq.compress(keys_flat)
        r_v, theta_v, _ = pq.compress(values_flat)

        # Compressed attention timing
        print("Timing PolarQuant attention...")
        times_turbo = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                # Decompress
                keys_comp = pq.decompress(r_k, theta_k).reshape_as(keys)
                values_comp = pq.decompress(r_v, theta_v).reshape_as(values)
                # Compute attention
                scores = torch.matmul(queries, keys_comp.transpose(-2, -1)) / (head_dim**0.5)
                attn = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn, values_comp)
            if self.device.type == "mps":
                torch.mps.synchronize()
            times_turbo.append(time.time() - start)

        avg_time_turbo = np.mean(times_turbo)

        # Compute throughput
        elements_per_run = batch_size * num_heads * seq_len * head_dim
        throughput_fp = elements_per_run / avg_time_fp / 1e6  # M elements/s
        throughput_turbo = elements_per_run / avg_time_turbo / 1e6

        speedup = avg_time_fp / avg_time_turbo

        results = {
            "full_precision": {
                "avg_time_ms": avg_time_fp * 1000,
                "throughput_melements_per_sec": throughput_fp,
            },
            "turboquant": {
                "avg_time_ms": avg_time_turbo * 1000,
                "throughput_melements_per_sec": throughput_turbo,
            },
            "speedup": speedup,
            "target_speedup": 8.0,  # Paper claims 8× throughput on Tensor Cores
        }

        print(f"\nResults:")
        print(f"  Full precision: {avg_time_fp*1000:.2f} ms ({throughput_fp:.1f} M elements/s)")
        print(f"  PolarQuant: {avg_time_turbo*1000:.2f} ms ({throughput_turbo:.1f} M elements/s)")
        print(f"  Overhead: {speedup:.2f}× ({'slower' if speedup < 1 else 'faster'})")
        print(f"  Paper target (Tensor Cores): 8× speedup")
        print(f"  Note: Decompression overhead on CPU/MPS; Tensor Cores needed for speedup")

        self.results["throughput_comparison"] = results
        return results

    def test_5_polar_quant_accuracy(self):
        """Test 5: PolarQuant reconstruction accuracy."""
        print("\n" + "=" * 70)
        print("TEST 5: PolarQuant Reconstruction Accuracy")
        print("=" * 70)

        dim = self.head_dim  # 64
        num_samples = 100

        # Generate random vectors
        torch.manual_seed(42)
        x = torch.randn(num_samples, dim, device=self.device)

        # PolarQuant with different bit widths
        bit_configs = [2, 3, 4, 5]
        results = {"by_bit_width": []}

        print(f"\nTesting {num_samples} random vectors (dim={dim})")

        for bits in bit_configs:
            pq = PolarQuant(dim, angle_bits=bits, device=self.device)

            # Compress and decompress
            r, theta_indices, _ = pq.compress(x)
            x_reconstructed = pq.decompress(r, theta_indices)

            # Compute metrics
            mse = torch.mean((x - x_reconstructed) ** 2).item()
            relative_error = torch.norm(x - x_reconstructed) / torch.norm(x)
            relative_error = relative_error.item()
            snr = 10 * np.log10(torch.mean(x**2).item() / mse)

            compression_ratio = pq.get_compression_ratio()

            result = {
                "angle_bits": bits,
                "compression_ratio": compression_ratio,
                "mse": mse,
                "relative_error": relative_error,
                "snr_db": snr,
            }
            results["by_bit_width"].append(result)

            print(f"\n  {bits}-bit angles:")
            print(f"    Compression: {compression_ratio:.2f}×")
            print(f"    MSE: {mse:.6f}")
            print(f"    Relative error: {relative_error:.4f}")
            print(f"    SNR: {snr:.2f} dB")

        self.results["polar_quant_accuracy"] = results
        return results

    def test_6_qjl_accuracy(self):
        """Test 6: QJL (Quantized Johnson-Lindenstrauss) accuracy."""
        print("\n" + "=" * 70)
        print("TEST 6: QJL Residual Correction Accuracy")
        print("=" * 70)

        dim = self.head_dim
        proj_dims = [32, 64, 128, 256]
        num_samples = 100

        torch.manual_seed(42)
        queries = torch.randn(num_samples, dim, device=self.device)
        keys = torch.randn(num_samples, dim, device=self.device)

        # Compute true dot products
        true_dots = (queries * keys).sum(dim=-1)

        results = {"by_proj_dim": []}

        print(f"\nTesting QJL with {num_samples} query-key pairs (dim={dim})")

        for proj_dim in proj_dims:
            qjl = QJLCompressor(dim, proj_dim, device=self.device)

            # Compress keys
            signs = qjl.compress(keys)

            # Estimate dot products
            estimated_dots = qjl.decompress_for_dot_product(signs, queries).squeeze(-1)

            # Compute error
            mse = torch.mean((true_dots - estimated_dots) ** 2).item()
            relative_error = torch.norm(true_dots - estimated_dots) / torch.norm(true_dots)
            relative_error = relative_error.item()

            # Check unbiased property
            bias = torch.mean(estimated_dots - true_dots).item()

            result = {
                "proj_dim": proj_dim,
                "compression_ratio": (dim * 16) / proj_dim,  # FP16 vs 1-bit
                "mse": mse,
                "relative_error": relative_error,
                "bias": bias,
            }
            results["by_proj_dim"].append(result)

            print(f"\n  Projection dim: {proj_dim}")
            print(f"    Compression: {(dim * 16) / proj_dim:.2f}×")
            print(f"    MSE: {mse:.6f}")
            print(f"    Relative error: {relative_error:.4f}")
            print(f"    Bias: {bias:.6f} (should be ~0 for unbiased estimator)")

        self.results["qjl_accuracy"] = results
        return results

    def save_results(self, output_dir="results"):
        """Save all test results to JSON."""
        os.makedirs(output_dir, exist_ok=True)

        self.results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "model_size": "small",
            "model_params": self.model.count_parameters(),
            "device": str(self.device),
            "turboquant_config": {
                "head_dim": self.head_dim,
                "angle_bits": self.angle_bits,
                "qjl_proj_dim": self.qjl_proj_dim,
                "total_bits_per_element": self.angle_bits + 1,
            },
        }

        output_file = os.path.join(output_dir, "turboquant_small_model_tests.json")
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_file}")
        return output_file

    def generate_report(self):
        """Generate human-readable report."""
        report = []
        report.append("=" * 70)
        report.append("TURBOQUANT SMALL MODEL TEST REPORT")
        report.append("=" * 70)
        report.append(f"\nTimestamp: {datetime.now().isoformat()}")
        report.append(f"Device: {self.device}")
        report.append(f"Model: Small (2.2B params, head_dim={self.head_dim})")

        # Test 1: Compression Ratio
        if "compression_ratio" in self.results:
            cr = self.results["compression_ratio"]
            report.append("\n" + "-" * 70)
            report.append("TEST 1: Compression Ratio")
            report.append("-" * 70)
            report.append(f"Full TurboQuant: {cr['turboquant_full']['compression_ratio']:.2f}×")
            report.append(f"Paper target: 6×+")
            status = "PASS" if cr["turboquant_full"]["compression_ratio"] >= 6.0 else "REVIEW"
            report.append(f"Status: {status}")

        # Test 2: KV Cache
        if "kv_cache_compression" in self.results:
            kv = self.results["kv_cache_compression"]
            report.append("\n" + "-" * 70)
            report.append("TEST 2: KV Cache Compression")
            report.append("-" * 70)
            report.append(f"Average reduction: {kv['average_reduction']:.2f}×")
            report.append(f"Paper target: 5.7×")
            status = "PASS" if kv["average_reduction"] >= 5.0 else "REVIEW"
            report.append(f"Status: {status}")

        # Test 3: Attention Accuracy
        if "attention_accuracy" in self.results:
            acc = self.results["attention_accuracy"]
            report.append("\n" + "-" * 70)
            report.append("TEST 3: Attention Accuracy")
            report.append("-" * 70)
            report.append(f"Relative error: {acc['relative_error']*100:.2f}%")
            report.append(f"Cosine similarity: {acc['cosine_similarity']:.4f}")
            report.append(f"Target error: < 5%")
            status = "PASS" if acc["relative_error"] < 0.05 else "REVIEW"
            report.append(f"Status: {status}")

        # Test 4: Throughput
        if "throughput_comparison" in self.results:
            tp = self.results["throughput_comparison"]
            report.append("\n" + "-" * 70)
            report.append("TEST 4: Throughput")
            report.append("-" * 70)
            report.append(f"Overhead factor: {tp['speedup']:.2f}×")
            report.append(f"Note: Decompression overhead; Tensor Cores needed for speedup")

        # Test 5: PolarQuant Accuracy
        if "polar_quant_accuracy" in self.results:
            pq = self.results["polar_quant_accuracy"]
            report.append("\n" + "-" * 70)
            report.append("TEST 5: PolarQuant Accuracy by Bit Width")
            report.append("-" * 70)
            for r in pq["by_bit_width"]:
                report.append(
                    f"{r['angle_bits']}-bit: {r['compression_ratio']:.1f}× compression, "
                    f"{r['relative_error']*100:.2f}% error, "
                    f"{r['snr_db']:.1f} dB SNR"
                )

        # Test 6: QJL Accuracy
        if "qjl_accuracy" in self.results:
            qjl = self.results["qjl_accuracy"]
            report.append("\n" + "-" * 70)
            report.append("TEST 6: QJL Accuracy by Projection Dim")
            report.append("-" * 70)
            for r in qjl["by_proj_dim"]:
                report.append(
                    f"proj_dim={r['proj_dim']}: {r['compression_ratio']:.1f}× compression, "
                    f"{r['relative_error']*100:.2f}% error, "
                    f"bias={r['bias']:.4f}"
                )

        report.append("\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)

        return "\n".join(report)


def main():
    print("=" * 70)
    print("TurboQuant Tests on Small Model")
    print("=" * 70)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\nDevice: {device}")

    # Build small model
    print("\nBuilding Small Model...")
    config = get_config("small")
    model = AdaptiveTransformer(config).to(device)
    model.eval()

    print(f"Model parameters: {model.count_parameters() / 1e9:.2f}B")
    print(f"Hidden dimension: {config.hidden_dim}")
    print(f"Head dimension: {config.hidden_dim // config.num_heads}")

    # Run tests
    tester = TurboQuantTester(model, device)

    tester.test_1_compression_ratio()
    tester.test_2_kv_cache_compression()
    tester.test_3_attention_accuracy()
    tester.test_4_throughput_comparison()
    tester.test_5_polar_quant_accuracy()
    tester.test_6_qjl_accuracy()

    # Save and report
    tester.save_results()
    report = tester.generate_report()

    print("\n" + report)

    # Save report
    with open("results/turboquant_small_model_report.txt", "w") as f:
        f.write(report)

    print("\n📄 Report saved to: results/turboquant_small_model_report.txt")


if __name__ == "__main__":
    main()
