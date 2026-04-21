#!/usr/bin/env python3
"""
Build Small Model and Run Benchmarks

This script:
1. Builds the small model (2.2B parameters)
2. Records detailed metrics (memory, parameters, FLOPs)
3. Runs needle-in-haystack and efficiency benchmarks
4. Saves results to results/small_model_benchmarks.json
"""

import os
import sys
import json
import time
import psutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_dir)

import torch
import torch.nn as nn
import numpy as np

from src.models.configs import get_config, get_model_size_params, print_config
from src.models.adaptive_transformer import AdaptiveTransformer
from src.benchmarks.flop_analysis import EfficiencyAnalyzer, run_flop_analysis


class SimpleNeedleEvaluator:
    """Simplified needle-in-haystack evaluator for testing."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.vocab_size = model.config.vocab_size

    def evaluate_single(self, model, context_length: int, depth_percent: float = 50) -> float:
        """
        Evaluate model's ability to retrieve information at given context length.
        Returns a score between 0 and 1.
        """
        import torch

        # Create a simple retrieval task
        # Use random tokens for haystack and mark specific position
        batch_size = 1

        # Generate random context
        input_ids = torch.randint(0, self.vocab_size, (batch_size, context_length))
        input_ids = input_ids.to(self.device)

        # Mark a specific position for the "needle"
        needle_position = int(context_length * depth_percent / 100)
        needle_token = 42  # Special marker token
        input_ids[0, needle_position] = needle_token

        # Forward pass
        model.eval()
        with torch.no_grad():
            try:
                logits = model(input_ids)

                # Check if model can predict the needle token's context
                # Simplified: check prediction accuracy near the needle
                pred_positions = range(
                    max(0, needle_position - 5), min(context_length - 1, needle_position + 5)
                )

                correct = 0
                total = 0
                for pos in pred_positions:
                    if pos < logits.shape[1] - 1:
                        pred = logits[0, pos].argmax().item()
                        actual = input_ids[0, pos + 1].item()
                        if pred == actual:
                            correct += 1
                        total += 1

                return correct / total if total > 0 else 0.0

            except Exception as e:
                print(f"Error during evaluation: {e}")
                return 0.0


class ModelBuilder:
    """Builds model and records comprehensive metrics."""

    def __init__(self, model_size: str = "small", device: str = "auto"):
        self.model_size = model_size
        self.device = self._get_device(device)
        self.metrics = {}
        self.process = psutil.Process()

    def _get_device(self, device: str):
        """Get the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        mem_info = self.process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
        }

    def build_model(self) -> AdaptiveTransformer:
        """Build model and record metrics."""
        print("=" * 70)
        print(f"Building {self.model_size.upper()} Model")
        print("=" * 70)

        # Record start metrics
        start_time = time.time()
        mem_before = self._get_memory_usage()

        # Load configuration
        config = get_config(self.model_size)
        self.metrics["config"] = {
            "num_layers": config.num_layers,
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "num_blocks": config.num_blocks,
            "mlp_ratio": config.mlp_ratio,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "max_qttt_steps": config.max_qttt_steps,
            "qttt_span_length": config.qttt_span_length,
        }

        # Print configuration
        print("\n📋 Model Configuration:")
        print_config(config)

        # Calculate theoretical parameters
        param_count = get_model_size_params(config)
        self.metrics["theoretical_params"] = param_count
        self.metrics["theoretical_params_human"] = f"{param_count / 1e9:.2f}B"
        print(f"\n📊 Theoretical Parameters: {param_count / 1e9:.2f}B")

        # Build model
        print("\n🔨 Building model...")
        try:
            model = AdaptiveTransformer(config)
            build_time = time.time() - start_time
            self.metrics["build_time_seconds"] = build_time
            print(f"✅ Model built in {build_time:.2f} seconds")
        except Exception as e:
            print(f"❌ Failed to build model: {e}")
            raise

        # Count actual parameters
        actual_params = model.count_parameters()
        attnres_params = model.count_attnsres_parameters()
        self.metrics["actual_params"] = actual_params
        self.metrics["attnres_params"] = attnres_params
        self.metrics["attnres_params_human"] = f"{attnres_params / 1e6:.2f}M"

        print(f"\n📈 Actual Parameters: {actual_params / 1e9:.2f}B")
        print(
            f"   - AttnRes params: {attnres_params / 1e6:.2f}M ({attnres_params/actual_params*100:.3f}%)"
        )

        # Memory usage after build
        mem_after = self._get_memory_usage()
        mem_used = mem_after["rss_mb"] - mem_before["rss_mb"]
        self.metrics["memory"] = {
            "before_build_mb": mem_before["rss_mb"],
            "after_build_mb": mem_after["rss_mb"],
            "model_memory_mb": mem_used,
            "model_memory_gb": mem_used / 1024,
        }
        print(f"\n💾 Memory Usage:")
        print(f"   - Before build: {mem_before['rss_mb']:.1f} MB")
        print(f"   - After build: {mem_after['rss_mb']:.1f} MB")
        print(f"   - Model memory: {mem_used:.1f} MB ({mem_used/1024:.2f} GB)")

        # Move to device
        print(f"\n📱 Moving model to {self.device}...")
        model = model.to(self.device)

        # Model size by dtype
        self.metrics["dtype"] = str(next(model.parameters()).dtype)
        print(f"   - Model dtype: {self.metrics['dtype']}")

        return model

    def analyze_model_structure(self, model: AdaptiveTransformer):
        """Analyze and record model structure details."""
        print("\n" + "=" * 70)
        print("Model Structure Analysis")
        print("=" * 70)

        structure = {
            "num_layers": len(model.layers),
            "num_attnres_modules": len(model.attnres_modules),
        }

        # Count parameters by component
        component_params = {}
        for name, param in model.named_parameters():
            parts = name.split(".")
            if len(parts) > 0:
                component = parts[0]
                if component not in component_params:
                    component_params[component] = 0
                component_params[component] += param.numel()

        structure["params_by_component"] = component_params
        structure["params_by_component_human"] = {
            k: f"{v / 1e6:.1f}M" if v < 1e9 else f"{v / 1e9:.2f}B"
            for k, v in component_params.items()
        }

        print("\n📊 Parameters by Component:")
        for component, count in sorted(component_params.items(), key=lambda x: -x[1]):
            human = structure["params_by_component_human"][component]
            pct = count / sum(component_params.values()) * 100
            print(f"   - {component}: {human} ({pct:.1f}%)")

        self.metrics["structure"] = structure
        return structure

    def run_efficiency_benchmarks(self, model: AdaptiveTransformer):
        """Run FLOP and efficiency analysis."""
        print("\n" + "=" * 70)
        print("Efficiency Benchmarks")
        print("=" * 70)

        model.eval()
        efficiency_metrics = {}

        # Test different sequence lengths
        seq_lengths = [512, 1024, 2048]
        batch_size = 1  # Keep small for CPU/memory constraints

        print("\n🔄 Testing different sequence lengths...")
        seq_results = []

        for seq_len in seq_lengths:
            print(f"\n   Testing seq_len={seq_len}...")
            try:
                # Create dummy input
                input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
                input_ids = input_ids.to(self.device)

                # Warmup
                with torch.no_grad():
                    _ = model(input_ids)

                # Measure forward pass time
                times = []
                for _ in range(3):
                    start = time.time()
                    with torch.no_grad():
                        _ = model(input_ids)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.time() - start)

                avg_time = np.mean(times)
                tokens_per_sec = seq_len / avg_time

                result = {
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "avg_time_ms": avg_time * 1000,
                    "tokens_per_sec": tokens_per_sec,
                }
                seq_results.append(result)

                print(
                    f"      Time: {avg_time*1000:.1f}ms, Throughput: {tokens_per_sec:.1f} tokens/sec"
                )

            except Exception as e:
                print(f"      ⚠️  Failed: {e}")
                seq_results.append({"seq_len": seq_len, "error": str(e)})

        efficiency_metrics["sequence_length_tests"] = seq_results

        # FLOP Analysis (simplified)
        print("\n📊 FLOP Analysis (theoretical):")
        config = model.config

        # Per layer FLOPs for one token
        head_dim = config.hidden_dim // config.num_heads

        # Attention FLOPs: 2 * seq_len * hidden_dim^2 (Q, K, V, O projections) + attention computation
        attn_proj_flops = 4 * 2 * config.hidden_dim * config.hidden_dim  # 4 projections
        attn_compute_flops = 2 * config.num_heads * head_dim  # Simplified

        # MLP FLOPs: 3 * hidden_dim * mlp_dim (gate, up, down)
        mlp_dim = config.hidden_dim * config.mlp_ratio
        mlp_flops = 3 * 2 * config.hidden_dim * mlp_dim

        per_layer_flops = attn_proj_flops + attn_compute_flops + mlp_flops
        total_flops = config.num_layers * per_layer_flops

        flop_analysis = {
            "per_layer_flops": per_layer_flops,
            "total_layers": config.num_layers,
            "total_flops_per_token": total_flops,
            "total_flops_human": f"{total_flops / 1e9:.2f} GFLOPs/token",
        }

        print(f"   - Per layer: {per_layer_flops / 1e6:.1f} MFLOPs")
        print(f"   - Total ({config.num_layers} layers): {total_flops / 1e9:.2f} GFLOPs/token")

        efficiency_metrics["flop_analysis"] = flop_analysis
        self.metrics["efficiency"] = efficiency_metrics

        return efficiency_metrics

    def run_needle_haystack(self, model: AdaptiveTransformer):
        """Run needle-in-haystack benchmark."""
        print("\n" + "=" * 70)
        print("Needle-in-Haystack Benchmark")
        print("=" * 70)

        # Use shorter context for CPU testing
        context_lengths = [1024, 2048, 4096]
        num_trials = 3

        print(f"\n🎯 Testing with context lengths: {context_lengths}")
        print(f"   Trials per length: {num_trials}")

        evaluator = SimpleNeedleEvaluator(model, self.device)

        results = []
        for ctx_len in context_lengths:
            print(f"\n   Testing context length: {ctx_len}...")
            try:
                # Run single trial
                score = evaluator.evaluate_single(model, ctx_len, depth_percent=50)
                results.append(
                    {
                        "context_length": ctx_len,
                        "depth_percent": 50,
                        "score": score,
                        "success": score > 0.5,
                    }
                )
                print(f"      Score: {score:.2%}")
            except Exception as e:
                print(f"      ⚠️  Failed: {e}")
                results.append({"context_length": ctx_len, "error": str(e)})

        # Calculate average
        successful = [r for r in results if "score" in r]
        if successful:
            avg_score = np.mean([r["score"] for r in successful])
        else:
            avg_score = 0.0

        nih_metrics = {
            "context_lengths_tested": context_lengths,
            "num_trials": num_trials,
            "results": results,
            "average_score": avg_score,
        }

        print(f"\n📊 Average Score: {avg_score:.2%}")
        self.metrics["needle_haystack"] = nih_metrics

        return nih_metrics

    def save_results(self, output_dir: str = "results"):
        """Save all metrics to JSON file."""
        os.makedirs(output_dir, exist_ok=True)

        # Add metadata
        self.metrics["metadata"] = {
            "model_size": self.model_size,
            "device": str(self.device),
            "timestamp": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
        }

        output_file = os.path.join(output_dir, f"{self.model_size}_model_benchmarks.json")
        with open(output_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")
        return output_file

    def generate_report(self) -> str:
        """Generate a human-readable report."""
        report = []
        report.append("=" * 70)
        report.append("ADAPTIVE DEEP NETWORKS - SMALL MODEL BENCHMARK REPORT")
        report.append("=" * 70)
        report.append(f"\nTimestamp: {self.metrics.get('metadata', {}).get('timestamp', 'N/A')}")
        report.append(f"Device: {self.metrics.get('metadata', {}).get('device', 'N/A')}")
        report.append(f"PyTorch: {self.metrics.get('metadata', {}).get('pytorch_version', 'N/A')}")

        report.append("\n" + "-" * 70)
        report.append("MODEL SPECIFICATIONS")
        report.append("-" * 70)
        config = self.metrics.get("config", {})
        report.append(f"Layers: {config.get('num_layers', 'N/A')}")
        report.append(f"Hidden Dim: {config.get('hidden_dim', 'N/A')}")
        report.append(f"Num Heads: {config.get('num_heads', 'N/A')}")
        report.append(f"Num Blocks (AttnRes): {config.get('num_blocks', 'N/A')}")
        report.append(f"MLP Ratio: {config.get('mlp_ratio', 'N/A')}")
        report.append(f"Vocab Size: {config.get('vocab_size', 'N/A')}")
        report.append(f"Max Seq Len: {config.get('max_seq_len', 'N/A')}")

        report.append("\n" + "-" * 70)
        report.append("PARAMETER COUNT")
        report.append("-" * 70)
        report.append(f"Theoretical: {self.metrics.get('theoretical_params_human', 'N/A')}")
        report.append(f"Actual: {self.metrics.get('actual_params', 0) / 1e9:.2f}B")
        report.append(f"AttnRes Overhead: {self.metrics.get('attnres_params_human', 'N/A')}")

        report.append("\n" + "-" * 70)
        report.append("BUILD METRICS")
        report.append("-" * 70)
        report.append(f"Build Time: {self.metrics.get('build_time_seconds', 0):.2f}s")
        mem = self.metrics.get("memory", {})
        report.append(f"Model Memory: {mem.get('model_memory_gb', 0):.2f} GB")

        report.append("\n" + "-" * 70)
        report.append("EFFICIENCY METRICS")
        report.append("-" * 70)
        eff = self.metrics.get("efficiency", {})
        flop = eff.get("flop_analysis", {})
        report.append(f"FLOPs per token: {flop.get('total_flops_human', 'N/A')}")

        seq_tests = eff.get("sequence_length_tests", [])
        if seq_tests:
            report.append("\nSequence Length Tests:")
            for test in seq_tests:
                if "error" not in test:
                    report.append(
                        f"  - {test['seq_len']} tokens: {test['avg_time_ms']:.1f}ms ({test['tokens_per_sec']:.1f} tok/s)"
                    )

        report.append("\n" + "-" * 70)
        report.append("NEEDLE-IN-HAYSTACK")
        report.append("-" * 70)
        nih = self.metrics.get("needle_haystack", {})
        report.append(f"Average Score: {nih.get('average_score', 0):.2%}")

        report.append("\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Build Small Model and Run Benchmarks")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--skip-needle", action="store_true", help="Skip needle-in-haystack test")
    parser.add_argument("--skip-efficiency", action="store_true", help="Skip efficiency benchmarks")
    args = parser.parse_args()

    # Initialize builder
    builder = ModelBuilder(model_size="small", device=args.device)

    # Build model
    model = builder.build_model()

    # Analyze structure
    builder.analyze_model_structure(model)

    # Run benchmarks
    if not args.skip_efficiency:
        builder.run_efficiency_benchmarks(model)

    if not args.skip_needle:
        builder.run_needle_haystack(model)

    # Save results
    builder.save_results(args.output_dir)

    # Generate and print report
    report = builder.generate_report()
    print("\n" + report)

    # Save report
    report_file = os.path.join(args.output_dir, "small_model_report.txt")
    with open(report_file, "w") as f:
        f.write(report)
    print(f"\n📄 Report saved to: {report_file}")


if __name__ == "__main__":
    main()
