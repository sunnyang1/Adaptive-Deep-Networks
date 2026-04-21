"""
Benchmark Engram Performance

Compares baseline ADN vs ADN+Engram on:
1. Needle-in-Haystack (long-context retrieval)
2. Training convergence speed
3. Memory and compute overhead
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.configs import AttnResSmallConfig
from src.models.adaptive_transformer import AdaptiveTransformer
from src.engram.integration import AdaptiveTransformerWithEngram, add_engram_to_config
from src.engram.config import EngramSmallConfig, EngramConfig


def create_needle_in_haystack_data(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cuda"
):
    """
    Create synthetic needle-in-haystack data.
    
    The "needle" is a special token pattern that should be retrieved
    after a long context (the "haystack").
    """
    # Create random haystack
    input_ids = torch.randint(
        10, vocab_size,  # Avoid special tokens
        size=(batch_size, seq_len),
        device=device
    )
    
    # Insert needle at random position
    needle_positions = torch.randint(
        seq_len // 4,  # At least 1/4 through
        seq_len - 1,   # Before last token
        size=(batch_size,),
        device=device
    )
    
    # Use token 1 as needle start, token 2 as needle end
    needle_token = 1
    
    for b in range(batch_size):
        pos = needle_positions[b].item()
        input_ids[b, pos] = needle_token
    
    # Target is to predict the needle token
    labels = input_ids.clone()
    
    return input_ids, labels, needle_positions


def measure_needle_retrieval_accuracy(
    model,
    seq_lengths=[1024, 2048, 4096],
    num_trials=10,
    device="cuda"
):
    """
    Measure accuracy of retrieving needle from haystack.
    
    Returns accuracy for each sequence length.
    """
    model.eval()
    results = {}
    
    for seq_len in seq_lengths:
        correct = 0
        total = 0
        
        for _ in range(num_trials):
            input_ids, labels, needle_positions = create_needle_in_haystack_data(
                batch_size=4,
                seq_len=seq_len,
                vocab_size=model.config.vocab_size,
                device=device
            )
            
            with torch.no_grad():
                logits = model(input_ids)
                
                # Check if needle token is in top-k predictions at needle position
                for b in range(4):
                    pos = needle_positions[b].item()
                    pred = logits[b, pos, :].argmax().item()
                    
                    # Needle token is 1
                    if pred == 1:
                        correct += 1
                    total += 1
        
        accuracy = correct / total if total > 0 else 0
        results[seq_len] = accuracy
        print(f"  Seq len {seq_len}: {accuracy:.2%}")
    
    return results


def measure_training_step(
    model,
    batch_size=4,
    seq_len=512,
    device="cuda",
    num_steps=10
):
    """
    Measure training step time and memory.
    
    Returns dict with timing and memory stats.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Warmup
    for _ in range(2):
        input_ids = torch.randint(
            0, model.config.vocab_size,
            size=(batch_size, seq_len),
            device=device
        )
        labels = input_ids.clone()
        
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Measure
    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated(device)
    
    times = []
    for _ in range(num_steps):
        input_ids = torch.randint(
            0, model.config.vocab_size,
            size=(batch_size, seq_len),
            device=device
        )
        labels = input_ids.clone()
        
        torch.cuda.synchronize()
        start = time.time()
        
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    end_mem = torch.cuda.memory_allocated(device)
    
    return {
        "step_time_mean": np.mean(times),
        "step_time_std": np.std(times),
        "memory_mb": (end_mem - start_mem) / 1024 / 1024,
    }


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_benchmark(
    output_dir: str = "results/engram_benchmark",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run complete benchmark comparing baseline vs Engram.
    """
    print("=" * 60)
    print("Engram Performance Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {
        "baseline": {},
        "engram": {},
    }
    
    # ===== Baseline Model =====
    print("[1/4] Testing Baseline ADN...")
    config_baseline = AttnResSmallConfig()
    config_baseline.num_layers = 8  # Smaller for faster testing
    config_baseline.hidden_dim = 512
    
    model_baseline = AdaptiveTransformer(config_baseline).to(device)
    params_baseline = count_parameters(model_baseline)
    print(f"  Parameters: {params_baseline / 1e6:.1f}M")
    
    # Needle in haystack
    print("  Running needle-in-haystack...")
    results["baseline"]["needle"] = measure_needle_retrieval_accuracy(
        model_baseline,
        seq_lengths=[512, 1024],
        num_trials=5,
        device=device
    )
    
    # Training step
    print("  Measuring training performance...")
    results["baseline"]["training"] = measure_training_step(
        model_baseline,
        batch_size=2,
        seq_len=256,
        device=device,
        num_steps=5
    )
    results["baseline"]["parameters"] = params_baseline
    
    del model_baseline
    torch.cuda.empty_cache()
    
    # ===== Engram Model =====
    print("\n[2/4] Testing ADN + Engram...")
    config_engram = AttnResSmallConfig()
    config_engram.num_layers = 8
    config_engram.hidden_dim = 512
    config_engram.use_engram = True
    config_engram.engram_config = EngramConfig(
        enabled=True,
        engram_vocab_size=[10000, 10000],
        max_ngram_size=3,
        n_embed_per_ngram=256,
        n_head_per_ngram=4,
        layer_ids=[1, 4],
        tokenizer_name_or_path="gpt2",
        pad_id=50256,
        seed=42,
    )
    
    model_engram = AdaptiveTransformerWithEngram(config_engram).to(device)
    params_engram = count_parameters(model_engram)
    print(f"  Parameters: {params_engram / 1e6:.1f}M")
    print(f"  Engram overhead: {(params_engram - params_baseline) / params_baseline * 100:.1f}%")
    
    # Needle in haystack
    print("  Running needle-in-haystack...")
    results["engram"]["needle"] = measure_needle_retrieval_accuracy(
        model_engram,
        seq_lengths=[512, 1024],
        num_trials=5,
        device=device
    )
    
    # Training step
    print("  Measuring training performance...")
    results["engram"]["training"] = measure_training_step(
        model_engram,
        batch_size=2,
        seq_len=256,
        device=device,
        num_steps=5
    )
    results["engram"]["parameters"] = params_engram
    
    # ===== Analysis =====
    print("\n[3/4] Analysis...")
    
    # Compare needle accuracy
    print("\nNeedle-in-Haystack Accuracy:")
    for seq_len in [512, 1024]:
        base_acc = results["baseline"]["needle"][seq_len]
        engram_acc = results["engram"]["needle"][seq_len]
        improvement = (engram_acc - base_acc) / base_acc * 100 if base_acc > 0 else 0
        print(f"  Seq len {seq_len}:")
        print(f"    Baseline: {base_acc:.2%}")
        print(f"    Engram:   {engram_acc:.2%}")
        print(f"    Change:   {improvement:+.1f}%")
    
    # Compare training speed
    print("\nTraining Performance:")
    base_time = results["baseline"]["training"]["step_time_mean"]
    engram_time = results["engram"]["training"]["step_time_mean"]
    slowdown = (engram_time - base_time) / base_time * 100
    print(f"  Baseline step time: {base_time * 1000:.1f}ms")
    print(f"  Engram step time:   {engram_time * 1000:.1f}ms")
    print(f"  Slowdown:           {slowdown:+.1f}%")
    
    # Memory overhead
    print("\nMemory:")
    base_mem = results["baseline"]["training"]["memory_mb"]
    engram_mem = results["engram"]["training"]["memory_mb"]
    print(f"  Baseline: {base_mem:.1f}MB")
    print(f"  Engram:   {engram_mem:.1f}MB")
    
    # ===== Save Results =====
    print("\n[4/4] Saving results...")
    output_file = Path(output_dir) / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_file}")
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Engram performance")
    parser.add_argument("--output-dir", default="results/engram_benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    results = run_benchmark(
        output_dir=args.output_dir,
        device=args.device
    )
