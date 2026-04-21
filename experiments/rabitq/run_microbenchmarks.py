#!/usr/bin/env python3
"""
RaBitQ Microbenchmarks

Profiles latency for RaBitQ operations: fit, compress, decompress, and
end-to-end pipeline across multiple configurations and sequence lengths.

Usage:
    python experiments/rabitq/run_microbenchmarks.py --quick
    python experiments/rabitq/run_microbenchmarks.py --device mps
    python experiments/rabitq/run_microbenchmarks.py --output-dir results/
"""

import sys
import os
import json
import argparse
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import time

from src.rabitq import (
    create_k4_v2,
    create_k3_v2,
    create_k2_v2,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')


def load_config():
    """Load experiment configuration."""
    import yaml
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def generate_kv_data(batch_size, num_heads, seq_len, head_dim, device):
    """Generate synthetic KV cache data."""
    shape = (batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(shape, device=device)
    values = torch.randn(shape, device=device)
    return keys, values


def benchmark_op(op_fn, warmup=3, repeats=10):
    """
    Benchmark an operation with warmup and multiple repeats.

    Returns dict with mean, median, std, min, max in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        op_fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.synchronize()

    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        op_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)  # ms

    return {
        'mean_ms': round(statistics.mean(timings), 3),
        'median_ms': round(statistics.median(timings), 3),
        'std_ms': round(statistics.stdev(timings) if len(timings) > 1 else 0, 3),
        'min_ms': round(min(timings), 3),
        'max_ms': round(max(timings), 3),
        'repeats': repeats,
    }


def benchmark_config(config_name, rq_factory, cfg, device):
    """Benchmark all operations for a single configuration."""
    test_cfg = cfg['test']
    seq_lengths = test_cfg['seq_lengths']['quick'] if cfg.get('quick') else test_cfg['seq_lengths']['full']
    batch_size = test_cfg['batch_size']
    num_heads = test_cfg['num_heads']
    head_dim = test_cfg['head_dim']

    results = {
        'config': config_name,
        'seq_length_results': [],
    }

    for seq_len in seq_lengths:
        print(f"    seq_len={seq_len}...", flush=True)
        rq = rq_factory()
        keys, values = generate_kv_data(batch_size, num_heads, seq_len, head_dim, device)

        sample_keys = keys[:, :, :min(64, seq_len), :]
        sample_values = values[:, :, :min(64, seq_len), :]

        # Fit latency
        fit_stats = benchmark_op(lambda: rq.fit(sample_keys, sample_values), warmup=1, repeats=5)

        # Compress latency
        # Pre-create rq and fit once for compress benchmark
        rq_comp = rq_factory()
        rq_comp.fit(sample_keys, sample_values)
        compress_stats = benchmark_op(lambda: rq_comp.compress(keys, values), warmup=2, repeats=10)

        # Decompress latency
        compressed = rq_comp.compress(keys, values)
        decompress_stats = benchmark_op(lambda: rq_comp.decompress(compressed), warmup=2, repeats=10)

        # End-to-end latency (fit + compress + decompress on fresh rq)
        def e2e_op():
            rq_e2e = rq_factory()
            rq_e2e.fit(sample_keys, sample_values)
            comp = rq_e2e.compress(keys, values)
            rq_e2e.decompress(comp)
        e2e_stats = benchmark_op(e2e_op, warmup=1, repeats=5)

        seq_result = {
            'seq_len': seq_len,
            'fit': fit_stats,
            'compress': compress_stats,
            'decompress': decompress_stats,
            'end_to_end': e2e_stats,
            'total_elements': batch_size * num_heads * seq_len * head_dim * 2,
        }
        results['seq_length_results'].append(seq_result)

        print(f"      fit={fit_stats['median_ms']:.1f}ms, "
              f"compress={compress_stats['median_ms']:.1f}ms, "
              f"decompress={decompress_stats['median_ms']:.1f}ms, "
              f"e2e={e2e_stats['median_ms']:.1f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description='RaBitQ Microbenchmarks')
    parser.add_argument('--quick', action='store_true', help='Quick mode (seq 128,512)')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu, cuda, mps')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results.json')
    args = parser.parse_args()

    cfg = load_config()
    cfg['quick'] = args.quick

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = 'cpu'

    print(f"RaBitQ Microbenchmarks (device={device}, quick={args.quick})")
    print("=" * 60)

    configs = {
        'k4_v2': lambda: create_k4_v2(head_dim=cfg['test']['head_dim'], device=device),
        'k3_v2': lambda: create_k3_v2(head_dim=cfg['test']['head_dim'], device=device),
        'k2_v2': lambda: create_k2_v2(head_dim=cfg['test']['head_dim'], device=device),
    }

    all_results = {
        'experiment': 'rabitq_microbenchmarks',
        'device': device,
        'quick_mode': args.quick,
        'configurations': {},
    }

    for config_name, rq_factory in configs.items():
        print(f"\n[{config_name}]")
        result = benchmark_config(config_name, rq_factory, cfg, device)
        all_results['configurations'][config_name] = result

    print(f"\n{'=' * 60}")

    # Save results
    output_dir = args.output_dir or os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'microbenchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
