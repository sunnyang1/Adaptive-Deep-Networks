#!/usr/bin/env python3
"""
Main Benchmark Runner for Adaptive Deep Networks Validation

Runs all validation benchmarks:
1. Needle-in-Haystack (long-context retrieval)
2. MATH (mathematical reasoning)
3. FLOP Analysis (efficiency verification)
4. Ablation Studies
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.configs import get_config, print_config
from models.adaptive_transformer import create_adaptive_transformer
from benchmarks.needle_haystack import NeedleHaystackBenchmark
from benchmarks.math_eval import MATHEvaluator
from benchmarks.flop_analysis import run_flop_analysis


def parse_args():
    parser = argparse.ArgumentParser(description='Run ADN validation benchmarks')
    
    parser.add_argument(
        '--model-size',
        type=str,
        default='small',
        choices=['small', 'medium', 'large'],
        help='Model size to validate'
    )
    
    parser.add_argument(
        '--benchmarks',
        type=str,
        nargs='+',
        default=['all'],
        choices=['all', 'needle', 'math', 'flop', 'ablation'],
        help='Benchmarks to run'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    parser.add_argument(
        '--skip-model-tests',
        action='store_true',
        help='Skip tests requiring actual model (for FLOP analysis only)'
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Setup environment and create output directories."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration
    config = {
        'timestamp': timestamp,
        'model_size': args.model_size,
        'device': args.device,
        'benchmarks': args.benchmarks,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        config['cuda_version'] = torch.version.cuda
        config['gpu_count'] = torch.cuda.device_count()
        config['gpu_names'] = [torch.cuda.get_device_name(i) 
                               for i in range(torch.cuda.device_count())]
    
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    return run_dir


def run_needle_haystack(model, tokenizer, output_dir):
    """Run needle-in-haystack benchmark."""
    print("\n" + "=" * 60)
    print("Running Needle-in-Haystack Benchmark")
    print("=" * 60)
    
    benchmark = NeedleHaystackBenchmark(
        model=model,
        tokenizer=tokenizer,
        context_lengths=[1024, 4096, 16384],  # Smaller for testing
        depths_per_length=5,
        num_trials=3
    )
    
    results = benchmark.run(verbose=True)
    
    # Save results
    with open(os.path.join(output_dir, 'needle_haystack.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    benchmark.print_summary()
    
    return results


def run_math_eval(model, tokenizer, output_dir):
    """Run MATH evaluation."""
    print("\n" + "=" * 60)
    print("Running MATH Evaluation")
    print("=" * 60)
    
    evaluator = MATHEvaluator(
        model=model,
        tokenizer=tokenizer,
        max_samples=100,  # Limit for testing
        use_cot=True
    )
    
    results = evaluator.run(verbose=True)
    evaluator.results = results
    
    # Save results
    with open(os.path.join(output_dir, 'math.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    evaluator.print_summary()
    
    return results


def run_flop_analysis_benchmark(output_dir):
    """Run FLOP analysis."""
    print("\n" + "=" * 60)
    print("Running FLOP Analysis")
    print("=" * 60)
    
    results = run_flop_analysis(
        output_path=os.path.join(output_dir, 'flop_analysis.json')
    )
    
    return results


def run_ablation_study(model, tokenizer, output_dir):
    """Run ablation study."""
    print("\n" + "=" * 60)
    print("Running Ablation Study")
    print("=" * 60)
    
    # Define configurations
    configs = [
        ('full', {'use_attnres': True, 'use_gating': True, 'use_qttt': True}),
        ('no_qttt', {'use_attnres': True, 'use_gating': True, 'use_qttt': False}),
        ('no_gating', {'use_attnres': True, 'use_gating': False, 'use_qttt': True}),
        ('no_attnres', {'use_attnres': False, 'use_gating': True, 'use_qttt': True}),
    ]
    
    results = {}
    
    for name, config in configs:
        print(f"\nTesting configuration: {name}")
        # Would run actual tests here
        results[name] = {
            'config': config,
            'placeholder': 'Actual results would be populated here'
        }
    
    # Save results
    with open(os.path.join(output_dir, 'ablation.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def print_summary(all_results, output_dir):
    """Print final summary."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for benchmark, results in all_results.items():
        print(f"\n{benchmark}:")
        if isinstance(results, dict):
            if 'average' in results:
                print(f"  Average accuracy: {results['average'] * 100:.1f}%")
            elif 'overall' in results:
                print(f"  Overall accuracy: {results['overall']['accuracy'] * 100:.1f}%")
            elif 'equivalence' in results:
                eq = results['equivalence']
                print(f"  FLOP equivalence ratio: {eq['ratio']:.3f}")
                print(f"  Verified: {'✓' if eq['is_equivalent'] else '✗'}")
    
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Adaptive Deep Networks Validation")
    print("=" * 60)
    print(f"Model size: {args.model_size}")
    print(f"Device: {args.device}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    
    # Setup
    run_dir = setup_environment(args)
    print(f"\nOutput directory: {run_dir}")
    
    # Print model config
    config = get_config(args.model_size)
    print_config(config)
    
    # Initialize model (or mock for FLOP-only tests)
    model = None
    tokenizer = None
    
    if not args.skip_model_tests:
        print("\nInitializing model...")
        try:
            model = create_adaptive_transformer(args.model_size)
            model = model.to(args.device)
            print(f"Model initialized: {model.count_parameters() / 1e6:.1f}M parameters")
            print(f"AttnRes parameters: {model.count_attnsres_parameters() / 1e6:.3f}M ({model.count_attnsres_parameters() / model.count_parameters() * 100:.3f}%)")
        except Exception as e:
            print(f"Warning: Could not initialize model: {e}")
            print("Continuing with FLOP analysis only...")
    
    # Run benchmarks
    all_results = {}
    benchmarks_to_run = args.benchmarks
    
    if 'all' in benchmarks_to_run:
        benchmarks_to_run = ['needle', 'math', 'flop', 'ablation']
    
    if 'needle' in benchmarks_to_run and model is not None:
        try:
            all_results['needle_haystack'] = run_needle_haystack(
                model, tokenizer, run_dir
            )
        except Exception as e:
            print(f"Needle-in-Haystack failed: {e}")
    
    if 'math' in benchmarks_to_run and model is not None:
        try:
            all_results['math'] = run_math_eval(model, tokenizer, run_dir)
        except Exception as e:
            print(f"MATH evaluation failed: {e}")
    
    if 'flop' in benchmarks_to_run:
        try:
            all_results['flop_analysis'] = run_flop_analysis_benchmark(run_dir)
        except Exception as e:
            print(f"FLOP analysis failed: {e}")
    
    if 'ablation' in benchmarks_to_run and model is not None:
        try:
            all_results['ablation'] = run_ablation_study(model, tokenizer, run_dir)
        except Exception as e:
            print(f"Ablation study failed: {e}")
    
    # Print summary
    print_summary(all_results, run_dir)
    
    print("\n✓ Validation complete!")


if __name__ == '__main__':
    main()
