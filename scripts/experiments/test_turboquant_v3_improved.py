"""
Improved TurboQuant V3 Test Suite
Based on tonbistudio/turboquant-pytorch best practices

Tests:
1. Lloyd-Max codebook properties (symmetry, distortion)
2. MSE distortion bounds vs theoretical
3. Compression accuracy with residual window
4. Needle-in-haystack retrieval (compressed vs fp16)
5. End-to-end model generation (if model available)
"""

import torch
import math
import time
import sys
import os
import argparse
from typing import Dict, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from turboquant import create_v3_k4_v2, create_v3_k3_v2, TurboQuantV3
from turboquant.v3_improved import LloydMaxQuantizerV3, RandomRotation


# ============================================================================
# Unit Tests
# ============================================================================

def test_lloyd_max_properties():
    """Verify Lloyd-Max codebook properties."""
    print("\n" + "=" * 70)
    print("TEST 1: Lloyd-Max Codebook Properties")
    print("=" * 70)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    for bits in [2, 3, 4]:
        quantizer = LloydMaxQuantizerV3(num_bits=bits, device=device)
        
        # Fit on Gaussian distributed data (like rotated unit vectors)
        # After random rotation of unit vectors, coordinates follow N(0, 1/d)
        samples = torch.randn(10000, device=device) * (1.0 / math.sqrt(64))
        quantizer.fit(samples)
        
        centroids = quantizer.centroids
        
        # Symmetry check
        centroid_sum = centroids.sum().item()
        is_symmetric = abs(centroid_sum) < 0.01
        
        # Range
        cmin, cmax = centroids.min().item(), centroids.max().item()
        
        # Spacing check (should be roughly uniform for high bits)
        sorted_c = torch.sort(centroids)[0]
        diffs = (sorted_c[1:] - sorted_c[:-1]).mean().item()
        
        status = "PASS" if is_symmetric else "FAIL"
        print(f"  {bits}-bit: {len(centroids)} levels, "
              f"range=[{cmin:+.3f}, {cmax:+.3f}], "
              f"avg_spacing={diffs:.4f}, "
              f"symmetric={is_symmetric} [{status}]")
    
    print()


def test_mse_distortion_bounds():
    """Verify MSE distortion is within theoretical bounds."""
    print("=" * 70)
    print("TEST 2: MSE Distortion Bounds")
    print("=" * 70)
    
    d = 64
    n_vectors = 1000
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    for bits in [2, 3, 4]:
        # Create compressor
        from turboquant.v3_improved import MSECompressor, MSECompressorConfig
        config = MSECompressorConfig(bits=bits, use_rotation=True, pack_bits=False, device=device)
        compressor = MSECompressor(config)
        
        # Generate random unit vectors
        x = torch.randn(n_vectors, d, device=device)
        x_norm = x.norm(dim=-1, keepdim=True)
        x = x / (x_norm + 1e-8)
        
        # Fit and compress
        compressor.fit(x, d)
        compressed = compressor.compress(x, d)
        x_hat = compressor.decompress(compressed)
        
        # Compute MSE
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
        
        # Theoretical bound: D_mse <= sqrt(3)*pi/2 * (1/4^b) for unit vectors
        theoretical = math.sqrt(3) * math.pi / 2 * (1 / (4 ** bits))
        ratio = mse / theoretical
        
        status = "OK" if ratio <= 2.0 else "HIGH"  # Allow 2x slack for finite d
        print(f"  {bits}-bit: MSE={mse:.6f}, theory={theoretical:.6f}, "
              f"ratio={ratio:.2f}x [{status}]")
    
    print()


def test_random_rotation_properties():
    """Test FWHT random rotation properties."""
    print("=" * 70)
    print("TEST 3: Random Rotation (FWHT) Properties")
    print("=" * 70)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    for dim in [64, 128]:
        rotation = RandomRotation(dim, device=device)
        
        # Test orthogonality: H @ H.T should be close to I * n
        x = torch.randn(10, dim, device=device)
        x_rot = rotation.rotate(x)
        x_back = rotation.inverse(x_rot)
        
        # Check reconstruction error
        recon_error = (x - x_back).abs().mean().item()
        
        # Check that rotation preserves norm (approximately)
        norm_orig = x.norm(dim=-1)
        norm_rot = x_rot.norm(dim=-1)
        norm_ratio = (norm_rot / (norm_orig + 1e-8)).mean().item()
        
        print(f"  dim={dim}: recon_error={recon_error:.6f}, "
              f"norm_ratio={norm_ratio:.4f}")
    
    print()


# ============================================================================
# Compression Tests
# ============================================================================

def test_residual_window():
    """Test compression with residual window (recent tokens in fp16)."""
    print("=" * 70)
    print("TEST 4: Residual Window Compression")
    print("=" * 70)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    batch, heads, seq_len, head_dim = 1, 8, 256, 64
    
    # Generate test data
    keys = torch.randn(batch, heads, seq_len, head_dim, device=device)
    values = torch.randn(batch, heads, seq_len, head_dim, device=device)
    sample = keys[:, :, :64, :]
    
    v3 = create_v3_k4_v2(head_dim=head_dim, device=device)
    v3.fit(sample, sample, head_dim=head_dim, layer_idx=0)
    
    residual_windows = [0, 32, 64, 128]
    
    for window in residual_windows:
        # Simulate residual window: compress only [0:seq_len-window]
        if window > 0:
            to_compress_k = keys[:, :, :-window, :]
            to_compress_v = values[:, :, :-window, :]
            recent_k = keys[:, :, -window:, :]
            recent_v = values[:, :, -window:, :]
        else:
            to_compress_k = keys
            to_compress_v = values
            recent_k = None
            recent_v = None
        
        # Compress
        compressed = v3.compress_kv(to_compress_k, to_compress_v, head_dim, layer_idx=0)
        decomp_k, decomp_v = v3.decompress_kv(compressed)
        
        # Reconstruct full cache
        if recent_k is not None:
            full_k = torch.cat([decomp_k, recent_k], dim=2)
            full_v = torch.cat([decomp_v, recent_v], dim=2)
        else:
            full_k = decomp_k
            full_v = decomp_v
        
        # Error on compressed portion only
        key_err = (to_compress_k - decomp_k).abs().mean() / to_compress_k.abs().mean()
        
        # Memory stats
        orig_bytes = keys.numel() * 2 + values.numel() * 2
        comp_bytes = (compressed['keys']['indices'].numel() + 
                     compressed['values']['indices'].numel() +
                     compressed['keys']['norm'].numel() * 2 +
                     compressed['values']['norm'].numel() * 2)
        if recent_k is not None:
            comp_bytes += recent_k.numel() * 2 + recent_v.numel() * 2
        
        ratio = orig_bytes / comp_bytes
        
        print(f"  window={window:>3d}: compressed_err={key_err.item()*100:.2f}%, "
              f"ratio={ratio:.2f}x")
    
    print()


def test_needle_in_haystack():
    """Test retrieval accuracy: can we find a specific token after compression?"""
    print("=" * 70)
    print("TEST 5: Needle-in-Haystack Retrieval")
    print("=" * 70)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    d = 64
    
    configs = [
        ("K4/V2", create_v3_k4_v2(head_dim=d, device=device)),
        ("K3/V2", create_v3_k3_v2(head_dim=d, device=device)),
    ]
    
    for seq_len in [512, 1024]:
        print(f"\n  Sequence length: {seq_len}")
        
        for name, v3 in configs:
            # Generate random keys (normalized like real attention)
            keys = torch.randn(1, 8, seq_len, d, device=device)
            keys = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
            
            # Pick a needle position
            needle_pos = seq_len // 3
            query = keys[:, :, needle_pos, :].clone()  # Query matching needle
            
            # Fit and compress
            sample = keys[:, :, :min(64, seq_len), :]
            v3.fit(sample, sample, head_dim=d, layer_idx=0)
            compressed = v3.compress_kv(keys, keys, head_dim=d, layer_idx=0)
            keys_deq, _ = v3.decompress_kv(compressed)
            keys_deq = keys_deq / (keys_deq.norm(dim=-1, keepdim=True) + 1e-8)
            
            # Compute attention scores (query @ keys.T)
            scores_fp16 = torch.matmul(query, keys.transpose(-2, -1)).squeeze()
            scores_deq = torch.matmul(query, keys_deq.transpose(-2, -1)).squeeze()
            
            # Find top positions
            top1_fp16 = scores_fp16.argmax().item()
            top1_deq = scores_deq.argmax().item()
            
            # Top-5 accuracy
            top5_fp16 = scores_fp16.topk(5).indices.tolist()
            top5_deq = scores_deq.topk(5).indices.tolist()
            
            # Check if needle is found
            found_fp16 = needle_pos in top5_fp16
            found_deq = needle_pos in top5_deq
            
            status = "MATCH" if top1_deq == top1_fp16 else ("TOP5" if found_deq else "MISS")
            
            print(f"    {name:10s}: fp16_top1={top1_fp16:>4d}, "
                  f"deq_top1={top1_deq:>4d}, needle={needle_pos:>4d} "
                  f"[{status}]")
    
    print()


# ============================================================================
# End-to-End Generation Test (if model available)
# ============================================================================

NEEDLE = "The secret project code name is AURORA-7749."
EXPECTED = "AURORA-7749"
FILLER = """The quarterly financial review meeting covered several topics including
budget allocations for the upcoming fiscal year, departmental spending reports, and projected
revenue streams from various business units. The committee discussed infrastructure upgrades
planned for the western regional offices and noted that maintenance schedules should be
coordinated with the facilities management team. Several action items were assigned to team
leads for follow-up before the next meeting cycle.\n\n"""


def build_needle_prompt(tokenizer, target_tokens=1024, needle_pos=0.5):
    """Build needle-in-haystack prompt."""
    filler_tokens = tokenizer.encode(FILLER)
    n_reps = max(1, target_tokens // len(filler_tokens))
    needle_idx = int(n_reps * needle_pos)
    
    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Internal Memo ---\n{NEEDLE}\n--- End Memo ---\n\n")
        parts.append(FILLER)
    
    haystack = "".join(parts)
    
    prompt = (
        f"<|im_start|>system\nYou are a helpful assistant. Answer concisely.<|im_end|>\n"
        f"<|im_start|>user\nRead this document:\n\n{haystack}\n\n"
        f"What is the secret project code name? Answer with just the code name.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


class V3Cache:
    """
    DynamicCache that compresses stored KV with TurboQuant V3.
    On each update: compress the full cache, decompress, return to attention.
    Uses incremental chunk storage to avoid recompression of old tokens.
    
    Adapted from tonbistudio/turboquant-pytorch/generation_test.py
    """
    
    def __init__(self, key_bits=4, value_bits=2, residual_window=128, device='cpu'):
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.residual_window = residual_window
        self.device = device
        
        self._compressors = {}
        self._chunks_k = {}  # layer_idx -> list of compressed key chunks
        self._chunks_v = {}  # layer_idx -> list of compressed value chunks
        self._fp16_recent_k = {}  # layer_idx -> recent fp16 keys
        self._fp16_recent_v = {}  # layer_idx -> recent fp16 values
        self._total_seq = {}
        self._seen_layers = 0
    
    def _get_compressor(self, layer_idx, head_dim):
        if layer_idx not in self._compressors:
            from turboquant import TurboQuantV3, TurboQuantV3Config
            config = TurboQuantV3Config(
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                use_rotation=True,
                pack_bits=True,
                device=self.device
            )
            self._compressors[layer_idx] = TurboQuantV3(config)
        return self._compressors[layer_idx]
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Update cache with new key-value states."""
        B, H, S_new, D = key_states.shape
        device = key_states.device
        comp = self._get_compressor(layer_idx, D)
        
        if layer_idx not in self._chunks_k:
            self._chunks_k[layer_idx] = []
            self._chunks_v[layer_idx] = []
            self._fp16_recent_k[layer_idx] = []
            self._fp16_recent_v[layer_idx] = []
            self._total_seq[layer_idx] = 0
        
        self._total_seq[layer_idx] += S_new
        
        # Add new tokens to fp16 recent buffer
        self._fp16_recent_k[layer_idx].append(key_states)
        self._fp16_recent_v[layer_idx].append(value_states)
        
        # Check if recent buffer exceeds window — compress overflow
        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
        rw = self.residual_window
        
        if recent_k.shape[2] > rw and rw > 0:
            overflow = recent_k.shape[2] - rw
            
            # Compress the overflow portion
            to_compress_k = recent_k[:, :, :overflow, :]
            to_compress_v = recent_v[:, :, :overflow, :]
            
            # Fit on first compress
            if not comp.compressors_k:
                sample_size = min(64, to_compress_k.shape[2])
                comp.fit(to_compress_k[:, :, :sample_size, :], 
                        to_compress_v[:, :, :sample_size, :], 
                        head_dim=D, layer_idx=0)
            
            ck, cv = comp.compress_kv(to_compress_k, to_compress_v, head_dim=D, layer_idx=0)
            self._chunks_k[layer_idx].append(ck)
            self._chunks_v[layer_idx].append(cv)
            
            # Keep only the recent window
            recent_k = recent_k[:, :, overflow:, :]
            recent_v = recent_v[:, :, overflow:, :]
            self._fp16_recent_k[layer_idx] = [recent_k]
            self._fp16_recent_v[layer_idx] = [recent_v]
        
        # Decompress all chunks + concat with fp16 recent
        parts_k = []
        parts_v = []
        
        for ck, cv in zip(self._chunks_k[layer_idx], self._chunks_v[layer_idx]):
            dk, dv = comp.decompress_kv(ck, cv)
            parts_k.append(dk.to(key_states.dtype))
            parts_v.append(dv.to(value_states.dtype))
        
        # Add fp16 recent
        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
        parts_k.append(recent_k)
        parts_v.append(recent_v)
        
        full_k = torch.cat(parts_k, dim=2)
        full_v = torch.cat(parts_v, dim=2)
        
        return full_k, full_v
    
    def get_seq_length(self, layer_idx=0):
        return self._total_seq.get(layer_idx, 0)


def test_generation(model_name=None, device='cpu'):
    """Test generation with compressed KV cache if model available."""
    print("=" * 70)
    print("TEST 6: End-to-End Generation (if model available)")
    print("=" * 70)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  transformers not installed, skipping generation test")
        print()
        return
    
    if model_name is None:
        print("  No model specified, skipping generation test")
        print("  Run with --model <name> to test generation")
        print()
        return
    
    print(f"\n  Loading model: {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device if device != 'mps' else 'cpu',
            low_cpu_mem_usage=True
        )
        model.eval()
        print(f"  Loaded successfully")
    except Exception as e:
        print(f"  Failed to load model: {e}")
        print()
        return
    
    # Test configurations
    configs = [
        {"fp16": True, "label": "FP16 baseline"},
        {"key_bits": 4, "value_bits": 2, "residual_window": 128, "label": "V3 K4/V2"},
    ]
    
    for target_tokens in [1024]:
        print(f"\n  Context: ~{target_tokens} tokens")
        
        for cfg in configs:
            prompt = build_needle_prompt(tokenizer, target_tokens)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                             max_length=target_tokens + 256)
            input_ids = inputs["input_ids"]
            if device != 'cpu' and torch.cuda.is_available():
                input_ids = input_ids.to(device)
            
            label = cfg.get("label", "???")
            print(f"\n    [{label}] {input_ids.shape[1]} tokens...", end=" ", flush=True)
            
            if cfg.get("fp16"):
                cache = None
            else:
                cache = V3Cache(
                    key_bits=cfg["key_bits"],
                    value_bits=cfg["value_bits"],
                    residual_window=cfg.get("residual_window", 128),
                    device='cpu'  # V3 uses CPU for now
                )
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=16,
                    do_sample=False,
                    past_key_values=cache,
                    use_cache=True,
                )
            
            new_tokens = outputs[0][input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            found = EXPECTED.lower() in response.lower()
            
            safe_response = response[:50].encode("ascii", errors="replace").decode("ascii")
            status = "FOUND" if found else "MISS"
            print(f"{status} | \"{safe_response}\"")
    
    print()


# ============================================================================
# Main
# ============================================================================

def run_all_tests(args):
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TurboQuant V3 Improved Test Suite")
    print("=" * 70)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    print()
    
    # Unit tests
    test_lloyd_max_properties()
    test_mse_distortion_bounds()
    test_random_rotation_properties()
    
    # Compression tests
    test_residual_window()
    test_needle_in_haystack()
    
    # Generation test (if requested)
    if args.model:
        test_generation(args.model, device)
    else:
        print("=" * 70)
        print("TEST 6: End-to-End Generation")
        print("=" * 70)
        print("  Skipped (use --model <name> to run)")
        print()
    
    print("=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurboQuant V3 Improved Tests")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name for generation test (e.g., Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--skip-slow", action="store_true",
                       help="Skip slower tests")
    
    args = parser.parse_args()
    run_all_tests(args)
