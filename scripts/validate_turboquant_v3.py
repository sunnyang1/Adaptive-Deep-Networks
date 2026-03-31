"""
TurboQuant V3 Validation Script

Validates compression quality by comparing attention scores on real model KV cache.
Tests multiple configurations and reports comprehensive metrics.

Based on:
- https://github.com/tonbistudio/turboquant-pytorch/validate.py
- https://github.com/tonbistudio/turboquant-pytorch/validate_v3.py

Usage:
    python scripts/validate_turboquant_v3.py --model Qwen/Qwen2.5-0.5B --seq-len 1024
"""

import torch
import torch.nn.functional as F
import time
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from turboquant import create_k4_v2, create_k3_v2, create_k2_v2, TurboQuantV3
from turboquant.api import TurboQuantConfig


# =============================================================================
# Configuration
# =============================================================================

NEEDLE = "The secret project code name is AURORA-7749."
QUESTION = "What is the secret project code name?"
FILLER = """The quarterly financial review meeting covered several topics including
budget allocations for the upcoming fiscal year, departmental spending reports, and projected
revenue streams from various business units. The committee discussed infrastructure upgrades
planned for the western regional offices and noted that maintenance schedules should be
coordinated with the facilities management team. Several action items were assigned to team
leads for follow-up before the next meeting cycle.\n\n"""


@dataclass
class ValidationResult:
    """Results from validating a compression configuration."""
    label: str
    compression_ratio: float
    cosine_sim: float
    top1_pct: float
    top5_pct: float
    mse_error: float
    compressed_mb: float
    original_mb: float
    latency_ms: float


def build_needle_prompt(tokenizer, target_tokens: int = 2048, needle_pos: float = 0.5) -> str:
    """Build needle-in-haystack prompt."""
    filler_len = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)
    
    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Memo ---\n{NEEDLE}\n--- End ---\n\n")
        parts.append(FILLER)
    
    haystack = "".join(parts)
    return (
        f"<|im_start|>user\n{haystack}\nQuestion: {QUESTION}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def find_needle_position(input_ids: torch.Tensor, tokenizer, needle: str = NEEDLE) -> Optional[int]:
    """Find needle token position in input."""
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
    input_ids_list = input_ids[0].tolist()
    
    # Try full match
    for i in range(len(input_ids_list) - len(needle_tokens) + 1):
        if input_ids_list[i:i + len(needle_tokens)] == needle_tokens:
            return i
    
    # Fallback: partial match
    for width in range(len(needle_tokens), 0, -1):
        sub = needle_tokens[:width]
        for i in range(len(input_ids_list) - width + 1):
            if input_ids_list[i:i + width] == sub:
                return i
    
    return None


def compute_attention_metrics(
    real_scores: torch.Tensor,
    compressed_scores: torch.Tensor,
    needle_pos: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute metrics comparing real vs compressed attention scores.
    
    Args:
        real_scores: Real attention scores [batch, heads, seq]
        compressed_scores: Compressed attention scores [batch, heads, seq]
        needle_pos: Position of needle token (if known)
        
    Returns:
        Dict with cosine_sim, top1_pct, top5_pct, needle_rank
    """
    B, H, S = real_scores.shape
    
    cosine_sims = []
    top1_matches = 0
    top5_matches = 0
    needle_ranks = []
    
    for h in range(H):
        rs = real_scores[0, h]  # (S,)
        cs = compressed_scores[0, h]
        
        # Cosine similarity
        cos = F.cosine_similarity(rs.unsqueeze(0), cs.unsqueeze(0)).item()
        cosine_sims.append(cos)
        
        # Top-1 match
        real_top1 = rs.argmax().item()
        comp_top1 = cs.argmax().item()
        if real_top1 == comp_top1:
            top1_matches += 1
        
        # Top-5 match (does real top-1 appear in compressed top-5?)
        comp_top5 = cs.topk(5).indices.tolist()
        if real_top1 in comp_top5:
            top5_matches += 1
        
        # Needle rank
        if needle_pos is not None:
            rank = (cs.argsort(descending=True) == needle_pos).nonzero()
            if len(rank) > 0:
                needle_ranks.append(rank[0].item())
    
    result = {
        "cosine_sim": sum(cosine_sims) / len(cosine_sims),
        "top1_pct": 100 * top1_matches / H,
        "top5_pct": 100 * top5_matches / H,
    }
    
    if needle_ranks:
        result["needle_rank"] = sum(needle_ranks) / len(needle_ranks)
    
    return result


def evaluate_compression_config(
    cache,
    compressor_factory,
    label: str,
    needle_pos: Optional[int] = None,
    device: str = 'cpu'
) -> ValidationResult:
    """
    Evaluate a compression configuration on model KV cache.
    
    Args:
        cache: HF model's past_key_values
        compressor_factory: Function that returns a TurboQuantV3 instance
        label: Name of this configuration
        needle_pos: Position of needle token
        device: Device to run on
        
    Returns:
        ValidationResult with all metrics
    """
    n_layers = len(cache)
    head_dim = cache[0][0].shape[-1]
    num_kv_heads = cache[0][0].shape[1]
    
    total_compressed_bytes = 0
    total_original_bytes = 0
    
    all_cosine_sims = []
    all_top1_matches = 0
    all_top5_matches = 0
    all_needle_ranks = []
    all_mse_errors = []
    total_checks = 0
    
    start_time = time.time()
    
    for layer_idx in range(n_layers):
        keys = cache[layer_idx][0]  # (B, H, S, D)
        values = cache[layer_idx][1]
        
        B, H, S, D = keys.shape
        
        # Query = last token (simulates next-token generation)
        query = keys[:, :, -1:, :]  # (B, H, 1, D)
        
        # Real attention scores
        real_scores = torch.matmul(
            query.float(), 
            keys.float().transpose(-2, -1)
        ).squeeze(-2)  # (B, H, S)
        
        # Compress and decompress
        compressor = compressor_factory(D, device)
        
        # Fit on first layer, reuse for others (or fit each layer)
        sample_size = min(64, S)
        compressor.fit(keys[:, :, :sample_size, :], values[:, :, :sample_size, :])
        
        compressed = compressor.compress(keys, values)
        keys_deq, values_deq = compressor.decompress(compressed)
        
        # Compressed attention scores
        compressed_scores = torch.matmul(
            query.float(),
            keys_deq.float().transpose(-2, -1)
        ).squeeze(-2)
        
        # Compute metrics
        metrics = compute_attention_metrics(real_scores, compressed_scores, needle_pos)
        all_cosine_sims.append(metrics["cosine_sim"])
        all_top1_matches += int(metrics["top1_pct"] * H / 100)
        all_top5_matches += int(metrics["top5_pct"] * H / 100)
        total_checks += H
        
        if "needle_rank" in metrics:
            all_needle_ranks.append(metrics["needle_rank"])
        
        # MSE between original and decompressed keys
        mse = ((keys - keys_deq) ** 2).mean().item()
        all_mse_errors.append(mse)
        
        # Memory accounting
        orig_bytes = keys.numel() * 2 + values.numel() * 2  # fp16
        
        # Compressed size
        k_idx_bytes = compressed['keys']['indices'].numel()
        v_idx_bytes = compressed['values']['indices'].numel()
        k_norm_bytes = compressed['keys']['norm'].numel() * 2  # fp16
        v_norm_bytes = compressed['values']['norm'].numel() * 2
        comp_bytes = k_idx_bytes + v_idx_bytes + k_norm_bytes + v_norm_bytes
        
        total_original_bytes += orig_bytes
        total_compressed_bytes += comp_bytes
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    # Aggregate results
    ratio = total_original_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 0
    avg_cosine = sum(all_cosine_sims) / len(all_cosine_sims)
    top1_pct = 100 * all_top1_matches / total_checks
    top5_pct = 100 * all_top5_matches / total_checks
    avg_mse = sum(all_mse_errors) / len(all_mse_errors)
    
    return ValidationResult(
        label=label,
        compression_ratio=ratio,
        cosine_sim=avg_cosine,
        top1_pct=top1_pct,
        top5_pct=top5_pct,
        mse_error=avg_mse,
        compressed_mb=total_compressed_bytes / (1024 ** 2),
        original_mb=total_original_bytes / (1024 ** 2),
        latency_ms=elapsed_ms
    )


def print_results_table(results: List[ValidationResult]):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print(f"{'Config':<25} {'Ratio':>7} {'CosSim':>8} {'Top-1%':>7} {'Top-5%':>7} {'MSE':>10} {'Time(ms)':>9}")
    print("-" * 100)
    
    for r in results:
        print(f"{r.label:<25} {r.compression_ratio:>7.1f}x {r.cosine_sim:>8.4f} "
              f"{r.top1_pct:>7.1f} {r.top5_pct:>7.1f} {r.mse_error:>10.6f} {r.latency_ms:>9.1f}")
    
    print("=" * 100)
    print(f"Memory: {results[0].original_mb:.1f} MB uncompressed")
    print()


def main():
    parser = argparse.ArgumentParser(description="Validate TurboQuant V3 compression")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name (e.g., Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--seq-len", type=int, default=1024,
                       help="Sequence length to test")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cpu/cuda/mps)")
    parser.add_argument("--skip-model", action="store_true",
                       help="Skip model test, use synthetic data")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print("=" * 100)
    print("TurboQuant V3 Validation")
    print("=" * 100)
    print(f"Device: {device}")
    
    # Define configurations to test
    configs = [
        ("FP16 Baseline", lambda d, dev: None, True),
        ("K4/V2 (Rec)", lambda d, dev: create_k4_v2(head_dim=d, device=dev), False),
        ("K3/V2", lambda d, dev: create_k3_v2(head_dim=d, device=dev), False),
        ("K2/V2 (Max)", lambda d, dev: create_k2_v2(head_dim=d, device=dev), False),
    ]
    
    results = []
    
    if args.skip_model or args.model is None:
        # Synthetic test
        print("\nUsing synthetic KV cache data...")
        print(f"Sequence length: {args.seq_len}")
        
        batch, heads, seq, dim = 1, 8, args.seq_len, 64
        
        # Create synthetic cache
        synthetic_cache = []
        for _ in range(4):  # 4 layers
            k = torch.randn(batch, heads, seq, dim, device=device)
            v = torch.randn(batch, heads, seq, dim, device=device)
            synthetic_cache.append((k, v))
        
        needle_pos = seq // 3
        
        for label, factory, is_baseline in configs:
            if is_baseline:
                # Baseline metrics
                results.append(ValidationResult(
                    label=label,
                    compression_ratio=1.0,
                    cosine_sim=1.0,
                    top1_pct=100.0,
                    top5_pct=100.0,
                    mse_error=0.0,
                    compressed_mb=synthetic_cache[0][0].numel() * 2 * 4 * 2 / (1024**2),
                    original_mb=synthetic_cache[0][0].numel() * 2 * 4 * 2 / (1024**2),
                    latency_ms=0.0
                ))
            else:
                result = evaluate_compression_config(
                    synthetic_cache, factory, label, needle_pos, device
                )
                results.append(result)
        
        print_results_table(results)
        
    else:
        # Real model test
        print(f"\nLoading model: {args.model}...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("Error: transformers not installed. Install with: pip install transformers")
            return
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            
            # Load model with appropriate dtype
            if device == 'cuda':
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=torch.float16,
                    device_map='auto',
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                model = model.to(device)
            
            model.eval()
            print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # Build prompt
        prompt = build_needle_prompt(tokenizer, args.seq_len)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                          max_length=args.seq_len + 256)
        
        if device != 'cpu':
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        seq_len = inputs["input_ids"].shape[1]
        print(f"\nContext: {seq_len} tokens")
        
        # Find needle
        needle_pos = find_needle_position(inputs["input_ids"], tokenizer)
        if needle_pos:
            print(f"Needle found at position: {needle_pos}")
        
        # Forward pass to get KV cache
        print("Running forward pass...")
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False)
        
        cache = outputs.past_key_values
        n_layers = len(cache)
        head_dim = cache[0][0].shape[-1]
        num_kv_heads = cache[0][0].shape[1]
        
        print(f"Layers: {n_layers}, Heads: {num_kv_heads}, Head dim: {head_dim}")
        
        # Convert cache to list format if needed
        if hasattr(cache, 'layers'):
            # DynamicCache format
            cache_list = [(layer.key_cache, layer.value_cache) for layer in cache.layers]
        else:
            cache_list = list(cache)
        
        # Evaluate each config
        for label, factory, is_baseline in configs:
            print(f"\nEvaluating: {label}...")
            
            if is_baseline:
                # Calculate baseline memory
                orig_bytes = sum(k.numel() + v.numel() for k, v in cache_list) * 2
                results.append(ValidationResult(
                    label=label,
                    compression_ratio=1.0,
                    cosine_sim=1.0,
                    top1_pct=100.0,
                    top5_pct=100.0,
                    mse_error=0.0,
                    compressed_mb=orig_bytes / (1024**2),
                    original_mb=orig_bytes / (1024**2),
                    latency_ms=0.0
                ))
            else:
                result = evaluate_compression_config(
                    cache_list, factory, label, needle_pos, device
                )
                results.append(result)
                print(f"  Ratio: {result.compression_ratio:.2f}x | "
                      f"CosSim: {result.cosine_sim:.4f} | "
                      f"Top-1: {result.top1_pct:.1f}%")
        
        print_results_table(results)
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    best_quality = max(results[1:], key=lambda r: r.cosine_sim)  # Exclude baseline
    best_compression = max(results[1:], key=lambda r: r.compression_ratio)
    
    print(f"Best Quality:    {best_quality.label} (cos={best_quality.cosine_sim:.4f})")
    print(f"Best Compression: {best_compression.label} ({best_compression.compression_ratio:.1f}x)")
    print()
    print("Recommendations:")
    print("  - Use K4/V2 for best quality with good compression (~4.9x)")
    print("  - Use K3/V2 for balanced quality/compression (~3.0x)")
    print("  - Use K2/V2 only when memory constrained (~7.1x)")
    print("=" * 100)


if __name__ == "__main__":
    main()
