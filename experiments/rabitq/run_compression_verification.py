#!/usr/bin/env python3
"""
RaBitQ Compression Verification

Verifies compression ratio, reconstruction quality (MSE, cosine similarity)
for k4_v2, k3_v2, and k2_v2 configurations across multiple sequence lengths.

Usage:
    python experiments/rabitq/run_compression_verification.py --quick
    python experiments/rabitq/run_compression_verification.py --device mps
    python experiments/rabitq/run_compression_verification.py --output-dir results/
"""

import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
import yaml
import numpy as np

from src.rabitq import (
    RaBitQ,
    RaBitQConfig,
    create_k4_v2,
    create_k3_v2,
    create_k2_v2,
)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')


def load_config():
    """Load experiment configuration."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def generate_kv_data(batch_size, num_heads, seq_len, head_dim, device):
    """Generate synthetic KV cache data for testing."""
    shape = (batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(shape, device=device)
    values = torch.randn(shape, device=device)
    return keys, values


def compute_cosine_similarity(original, reconstructed):
    """Compute mean cosine similarity across all vectors."""
    orig_flat = original.reshape(-1, original.shape[-1])
    recon_flat = reconstructed.reshape(-1, reconstructed.shape[-1])
    cos_sim = F.cosine_similarity(orig_flat, recon_flat, dim=-1)
    return cos_sim.mean().item()


def compute_mse(original, reconstructed):
    """Compute mean squared error."""
    return F.mse_loss(original, reconstructed).item()


def measure_compression_ratio(keys, values, rq):
    """Measure actual compression ratio by comparing tensor sizes."""
    # Original size (fp16 for K + V)
    original_elements = keys.numel() + values.numel()
    original_bytes = original_elements * 2  # fp16 = 2 bytes

    # Compress
    compressed = rq.compress(keys, values)

    # New CompressedKV structure: list of QuantizedVector objects
    # We count only reconstruction-effective storage (binary + ex + delta + vl)
    compressed_bytes = 0
    for key in ('keys', 'values'):
        ck = compressed[key]
        for qv in ck.quantized_vectors:
            compressed_bytes += qv.binary_code_packed.numel() * qv.binary_code_packed.element_size()
            compressed_bytes += qv.ex_code_packed.numel() * qv.ex_code_packed.element_size()
            compressed_bytes += 2 * 4  # delta + vl only

    ratio = original_bytes / max(compressed_bytes, 1)
    return ratio, original_bytes, compressed_bytes


def test_config(config_name, rq, cfg, device):
    """Run verification for a single RaBitQ configuration."""
    test_cfg = cfg['test']
    seq_lengths = test_cfg['seq_lengths']['quick'] if cfg.get('quick') else test_cfg['seq_lengths']['full']
    batch_size = test_cfg['batch_size']
    num_heads = test_cfg['num_heads']
    head_dim = test_cfg['head_dim']

    target_cfg = cfg['targets'].get(config_name, {})
    expected_ratio = target_cfg.get('expected_ratio', 0)
    tolerance = target_cfg.get('tolerance', 1.0)

    results = {
        'config': config_name,
        'expected_ratio': expected_ratio,
        'tolerance': tolerance,
        'seq_length_results': [],
        'passed': True,
    }

    for seq_len in seq_lengths:
        print(f"    seq_len={seq_len}...", end=' ', flush=True)

        # Generate data
        keys, values = generate_kv_data(batch_size, num_heads, seq_len, head_dim, device)

        # Fit on first batch, then measure
        sample_keys = keys[:, :, :min(64, seq_len), :]
        sample_values = values[:, :, :min(64, seq_len), :]
        rq.fit(sample_keys, sample_values)

        # Compress and decompress
        compressed = rq.compress(keys, values)
        keys_recon, values_recon = rq.decompress(compressed)

        # Compute metrics
        key_mse = compute_mse(keys.float(), keys_recon.float())
        val_mse = compute_mse(values.float(), values_recon.float())
        key_cosine = compute_cosine_similarity(keys.float(), keys_recon.float())
        val_cosine = compute_cosine_similarity(values.float(), values_recon.float())
        ratio, orig_bytes, comp_bytes = measure_compression_ratio(keys, values, rq)

        seq_result = {
            'seq_len': seq_len,
            'compression_ratio': round(ratio, 3),
            'original_bytes': orig_bytes,
            'compressed_bytes': comp_bytes,
            'key_mse': round(key_mse, 6),
            'val_mse': round(val_mse, 6),
            'key_cosine_similarity': round(key_cosine, 4),
            'val_cosine_similarity': round(val_cosine, 4),
            'ratio_within_tolerance': abs(ratio - expected_ratio) <= tolerance,
        }
        results['seq_length_results'].append(seq_result)

        if not seq_result['ratio_within_tolerance']:
            results['passed'] = False

        print(f"ratio={ratio:.2f} (target={expected_ratio}±{tolerance}), "
              f"k_cos={key_cosine:.4f}, v_cos={val_cosine:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='RaBitQ Compression Verification')
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

    print(f"RaBitQ Compression Verification (device={device}, quick={args.quick})")
    print("=" * 60)

    configs = {
        'k4_v2': lambda: create_k4_v2(head_dim=cfg['test']['head_dim'], device=device),
        'k3_v2': lambda: create_k3_v2(head_dim=cfg['test']['head_dim'], device=device),
        'k2_v2': lambda: create_k2_v2(head_dim=cfg['test']['head_dim'], device=device),
    }

    all_results = {
        'experiment': cfg['experiment']['name'],
        'description': cfg['experiment']['description'],
        'device': device,
        'quick_mode': args.quick,
        'configurations': {},
        'summary': {
            'total_configs': len(configs),
            'passed_configs': 0,
            'failed_configs': 0,
        }
    }

    for config_name, rq_factory in configs.items():
        print(f"\n[{config_name}]")
        rq = rq_factory()
        result = test_config(config_name, rq, cfg, device)
        all_results['configurations'][config_name] = result
        if result['passed']:
            all_results['summary']['passed_configs'] += 1
            print(f"  ✅ PASSED")
        else:
            all_results['summary']['failed_configs'] += 1
            print(f"  ❌ FAILED")

    print(f"\n{'=' * 60}")
    print(f"Summary: {all_results['summary']['passed_configs']}/{all_results['summary']['total_configs']} passed")

    # Save results
    output_dir = args.output_dir or os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'compression_verification_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")

    return all_results['summary']['failed_configs'] == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
