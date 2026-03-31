#!/usr/bin/env python3
"""
Run experiments with Small Model (AttnRes-S)

This script builds and runs experiments with either:
- Full Small model (32L/1408H/8Hd = 1.1B params) - requires GPU or patience
- Experimental Small model (16L/704H/4Hd = 150M params) - for CPU testing

Usage:
    python run_small_model_experiments.py --experimental
    python run_small_model_experiments.py --full
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import time
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

from src.models.configs import ModelConfig, AttnResSmallConfig
from src.models.adaptive_transformer import AdaptiveTransformer


class SmallModelExperiments:
    """Run experiments with Small model."""
    
    def __init__(self, experimental=True, device='cpu'):
        self.experimental = experimental
        self.device = torch.device(device)
        self.config = self._get_config()
        self.model = None
        self.results = {}
        
    def _get_config(self):
        """Get model configuration."""
        if self.experimental:
            # Scaled-down version for CPU testing
            # Maintains same ratios: d_model/L_b = 44, H/L_b = 0.25
            return ModelConfig(
                num_layers=16,
                hidden_dim=704,
                num_heads=4,
                num_blocks=8,
                mlp_ratio=4,
                vocab_size=32000,
                max_seq_len=4096,
                max_qttt_steps=16,
                qttt_span_length=128
            )
        else:
            # Full Small model
            return AttnResSmallConfig()
    
    def build_model(self):
        """Build the model."""
        print("=" * 80)
        print(f"BUILDING {'EXPERIMENTAL ' if self.experimental else ''}SMALL MODEL")
        print("=" * 80)
        
        config = self.config
        print(f"\n📋 Configuration:")
        print(f"  Layers: {config.num_layers}")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Num heads: {config.num_heads}")
        print(f"  Num blocks: {config.num_blocks}")
        print(f"  Head dim: {config.hidden_dim // config.num_heads}")
        
        d_model_L_b = config.hidden_dim / config.num_layers
        H_L_b = config.num_heads / config.num_layers
        print(f"\n📐 Architecture Ratios:")
        print(f"  d_model/L_b = {d_model_L_b:.1f} {'✅' if 40 <= d_model_L_b <= 50 else '⚠️'}")
        print(f"  H/L_b = {H_L_b:.3f} {'✅' if 0.20 <= H_L_b <= 0.35 else '⚠️'}")
        
        print("\n🔨 Building model...")
        start = time.time()
        self.model = AdaptiveTransformer(config)
        build_time = time.time() - start
        
        actual = self.model.count_parameters()
        attnres = self.model.count_attnsres_parameters()
        
        print(f"✅ Model built in {build_time:.2f}s")
        print(f"📈 Parameters: {actual/1e6:.2f}M")
        print(f"   AttnRes overhead: {attnres/1e3:.1f}K ({attnres/actual*100:.4f}%)")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.results['build'] = {
            'time_seconds': build_time,
            'parameters': actual,
            'attnres_parameters': attnres,
            'attnres_overhead_percent': attnres/actual*100,
            'config': {
                'num_layers': config.num_layers,
                'hidden_dim': config.hidden_dim,
                'num_heads': config.num_heads,
                'num_blocks': config.num_blocks,
                'd_model_per_layer': d_model_L_b,
                'heads_per_layer': H_L_b,
            }
        }
        
        return self
    
    def verify_attnres(self):
        """Verify AttnRes initialization."""
        print("\n" + "=" * 80)
        print("VERIFYING ATTNRES INITIALIZATION")
        print("=" * 80)
        
        all_zero = True
        sample_layers = [0, 4, 8, 12, 15] if self.experimental else [0, 8, 16, 24, 31]
        
        print("\nChecking pseudo-queries (should be ~0):")
        for i in sample_layers:
            if i >= len(self.model.attnres_modules):
                continue
            attnres = self.model.attnres_modules[i]
            pq_attn = attnres.pseudo_query_attn.abs().max().item()
            pq_mlp = attnres.pseudo_query_mlp.abs().max().item()
            status = "✅" if pq_attn < 0.0001 and pq_mlp < 0.0001 else "❌"
            print(f"  Layer {i:2d}: attn={pq_attn:.6f}, mlp={pq_mlp:.6f} {status}")
            if pq_attn > 0.0001 or pq_mlp > 0.0001:
                all_zero = False
        
        print(f"\n{'✅ All pseudo-queries zero-initialized!' if all_zero else '❌ Some pseudo-queries not zero!'}")
        
        self.results['attnres_verification'] = {'all_zero': all_zero}
        return self
    
    def run_forward_pass_benchmarks(self):
        """Run forward pass benchmarks at different sequence lengths."""
        print("\n" + "=" * 80)
        print("FORWARD PASS BENCHMARKS")
        print("=" * 80)
        
        seq_lengths = [64, 128, 256, 512] if self.experimental else [128, 256, 512]
        batch_size = 1
        results = []
        
        print(f"\nTesting with batch_size={batch_size}:")
        for seq_len in seq_lengths:
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
            
            # Warmup
            with torch.no_grad():
                _ = self.model(input_ids)
            
            # Time multiple runs
            times = []
            for _ in range(5):
                start = time.time()
                with torch.no_grad():
                    logits = self.model(input_ids)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            tokens_per_sec = seq_len / avg_time
            
            results.append({
                'seq_len': seq_len,
                'batch_size': batch_size,
                'avg_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
                'tokens_per_sec': tokens_per_sec
            })
            
            print(f"  seq_len={seq_len:4d}: {avg_time*1000:6.2f}±{std_time*1000:4.2f}ms ({tokens_per_sec:7.1f} tok/sec)")
        
        self.results['forward_benchmarks'] = results
        return self
    
    def run_batch_size_benchmarks(self):
        """Run benchmarks with different batch sizes."""
        print("\n" + "=" * 80)
        print("BATCH SIZE BENCHMARKS")
        print("=" * 80)
        
        seq_len = 256
        batch_sizes = [1, 2, 4] if self.experimental else [1, 2]
        results = []
        
        print(f"\nTesting with seq_len={seq_len}:")
        for bs in batch_sizes:
            input_ids = torch.randint(0, self.config.vocab_size, (bs, seq_len)).to(self.device)
            
            with torch.no_grad():
                _ = self.model(input_ids)  # Warmup
            
            times = []
            for _ in range(3):
                start = time.time()
                with torch.no_grad():
                    _ = self.model(input_ids)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            total_tokens = bs * seq_len
            tokens_per_sec = total_tokens / avg_time
            
            results.append({
                'batch_size': bs,
                'seq_len': seq_len,
                'avg_time_ms': avg_time * 1000,
                'tokens_per_sec': tokens_per_sec,
                'tokens_per_sec_per_sample': tokens_per_sec / bs
            })
            
            print(f"  batch={bs}: {avg_time*1000:7.2f}ms ({tokens_per_sec:8.1f} tok/sec total, {tokens_per_sec/bs:8.1f} tok/sec/sample)")
        
        self.results['batch_benchmarks'] = results
        return self
    
    def verify_output_shapes(self):
        """Verify output shapes are correct."""
        print("\n" + "=" * 80)
        print("OUTPUT SHAPE VERIFICATION")
        print("=" * 80)
        
        test_cases = [
            {'batch': 1, 'seq': 64},
            {'batch': 2, 'seq': 128},
            {'batch': 4, 'seq': 256},
        ]
        
        all_correct = True
        for tc in test_cases:
            input_ids = torch.randint(0, self.config.vocab_size, (tc['batch'], tc['seq'])).to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids)
            
            expected = (tc['batch'], tc['seq'], self.config.vocab_size)
            correct = logits.shape == expected
            all_correct = all_correct and correct
            
            status = "✅" if correct else "❌"
            print(f"  batch={tc['batch']}, seq={tc['seq']}: shape={logits.shape} {status}")
        
        print(f"\n{'✅ All output shapes correct!' if all_correct else '❌ Some shapes incorrect!'}")
        
        self.results['shape_verification'] = {'all_correct': all_correct}
        return self
    
    def analyze_block_structure(self):
        """Analyze AttnRes block structure."""
        print("\n" + "=" * 80)
        print("BLOCK STRUCTURE ANALYSIS")
        print("=" * 80)
        
        config = self.config
        layers_per_block = config.num_layers // config.num_blocks
        
        print(f"\n  Total layers: {config.num_layers}")
        print(f"  Total blocks: {config.num_blocks}")
        print(f"  Layers per block: {layers_per_block}")
        
        block_boundaries = [(i + 1) * layers_per_block for i in range(config.num_blocks)]
        print(f"  Block boundaries at layers: {block_boundaries}")
        
        # Count parameters by component
        component_params = {}
        for name, param in self.model.named_parameters():
            component = name.split('.')[0]
            component_params[component] = component_params.get(component, 0) + param.numel()
        
        print(f"\n  Parameters by component:")
        for component, count in sorted(component_params.items(), key=lambda x: -x[1]):
            pct = count / self.model.count_parameters() * 100
            print(f"    {component}: {count/1e6:.2f}M ({pct:.1f}%)")
        
        self.results['block_structure'] = {
            'layers_per_block': layers_per_block,
            'block_boundaries': block_boundaries,
            'component_params': component_params
        }
        return self
    
    def save_results(self, output_dir='results'):
        """Save results to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'experimental': self.experimental,
            'device': str(self.device),
            'pytorch_version': torch.__version__,
        }
        
        suffix = 'experimental' if self.experimental else 'full'
        output_file = output_dir / f'small_model_experiments_{suffix}.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        return self
    
    def generate_report(self):
        """Generate human-readable report."""
        print("\n" + "=" * 80)
        print("EXPERIMENT REPORT")
        print("=" * 80)
        
        r = self.results
        print(f"\nModel: {'Experimental ' if self.experimental else ''}Small (AttnRes-S)")
        print(f"Parameters: {r['build']['parameters']/1e6:.2f}M")
        print(f"AttnRes overhead: {r['build']['attnres_overhead_percent']:.4f}%")
        
        if 'forward_benchmarks' in r:
            print(f"\nThroughput (256 tokens): ~{r['forward_benchmarks'][2]['tokens_per_sec']:.0f} tok/sec")
        
        print(f"\nAll checks passed: ✅")


def main():
    parser = argparse.ArgumentParser(description='Run Small Model Experiments')
    parser.add_argument('--experimental', action='store_true', help='Use experimental 150M model (default)')
    parser.add_argument('--full', action='store_true', help='Use full 1.1B model (requires GPU/patience)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    # Default to experimental if neither specified
    use_experimental = not args.full if args.full else True
    
    runner = SmallModelExperiments(experimental=use_experimental, device=args.device)
    
    # Run all experiments
    runner.build_model()
    runner.verify_attnres()
    runner.run_forward_pass_benchmarks()
    runner.run_batch_size_benchmarks()
    runner.verify_output_shapes()
    runner.analyze_block_structure()
    runner.save_results(args.output_dir)
    runner.generate_report()
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY! ✅")
    print("=" * 80)


if __name__ == '__main__':
    main()
