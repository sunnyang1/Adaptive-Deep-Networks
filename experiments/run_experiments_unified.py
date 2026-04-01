#!/usr/bin/env python3
"""
Unified Experiment Runner

Consolidated experiment runner combining features from:
- run_all.py (quick/full modes, experiment definitions)
- run_refactored.py (refactored experiment classes)
- run_experiments.py (unified runner with discovery)
- run_all_experiments.py (complete experiment descriptions)

Usage:
    # Run all experiments
    python experiments/run_experiments_unified.py --all
    
    # Run specific category
    python experiments/run_experiments_unified.py --category core
    python experiments/run_experiments_unified.py --category validation
    
    # Run specific experiment
    python experiments/run_experiments_unified.py --exp exp1
    
    # Quick mode (reduced computation)
    python experiments/run_experiments_unified.py --all --quick
    
    # List available experiments
    python experiments/run_experiments_unified.py --list
    
    # Dry run (show what would be executed)
    python experiments/run_experiments_unified.py --all --dry-run
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ExperimentResult:
    """Experiment execution result."""
    id: str
    name: str
    status: str  # success, failed, skipped, timeout, error
    duration: float
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict] = None


@dataclass
class ExperimentDef:
    """Experiment definition."""
    id: str
    name: str
    description: str
    category: str
    script: str
    quick_args: List[str]
    full_args: List[str]
    timeout: int = 3600  # seconds


# Experiment Registry
EXPERIMENTS: Dict[str, ExperimentDef] = {
    # Core Experiments
    'exp1': ExperimentDef(
        id='exp1',
        name='Representation Burial Measurement',
        description='Measure gradient attenuation across layers (PreNorm vs AttnRes)',
        category='core',
        script='experiments/core/exp1_representation_burial/run_exp1.py',
        quick_args=['--num_samples', '10', '--num_layers', '32'],
        full_args=['--num_samples', '50', '--num_layers', '96'],
        timeout=1800
    ),
    'exp2': ExperimentDef(
        id='exp2',
        name='Logit Margin Analysis',
        description='Analyze logit margin requirements for long-context retrieval',
        category='core',
        script='experiments/core/exp2_margin_analysis/run_exp2.py',
        quick_args=['--context_lengths', '1024', '4096', '--num_samples', '5'],
        full_args=['--context_lengths', '1024', '4096', '16384', '32768', '--num_samples', '20'],
        timeout=3600
    ),
    'exp3': ExperimentDef(
        id='exp3',
        name='Gradient Flow Measurement',
        description='Measure gradient flow improvement with AttnRes',
        category='core',
        script='experiments/core/exp3_gradient_flow/run_exp3.py',
        quick_args=['--num_steps', '100', '--batch_size', '2'],
        full_args=['--num_steps', '1000', '--batch_size', '8'],
        timeout=2400
    ),
    'exp4': ExperimentDef(
        id='exp4',
        name='FLOP Equivalence Validation',
        description='Validate T_think ≈ 2 * N_qTTT * k formula',
        category='core',
        script='experiments/core/exp4_flop_equivalence/run_exp4.py',
        quick_args=['--total_flops', '1e13', '--model_size', 'small'],
        full_args=['--total_flops', '5e14', '--model_size', 'medium'],
        timeout=1800
    ),
    'exp5': ExperimentDef(
        id='exp5',
        name='Component Synergy Analysis',
        description='Quantify synergistic effects of AttnRes, qTTT, and Gating',
        category='core',
        script='experiments/core/exp5_synergy/run_exp5.py',
        quick_args=['--num_configs', '4'],
        full_args=['--num_configs', '8'],
        timeout=3600
    ),
    'exp6': ExperimentDef(
        id='exp6',
        name='Auxiliary Validation',
        description='Initialization effects, block size impact, hyperparameter sensitivity',
        category='core',
        script='experiments/core/exp6_auxiliary/run_exp6.py',
        quick_args=['--quick'],
        full_args=[],
        timeout=1200
    ),
    
    # Paper Table Experiments (new)
    'table3': ExperimentDef(
        id='table3',
        name='Inference Benchmark (Table 3)',
        description='End-to-end inference: throughput, memory, latency for width vs depth (Paper Table 3)',
        category='core',
        script='experiments/core/table3_inference_benchmark/run_table3.py',
        quick_args=['--quick'],
        full_args=[],
        timeout=1800
    ),
    'table5': ExperimentDef(
        id='table5',
        name='Logit Margin Distribution (Table 5)',
        description='Attention logit margins before/after qTTT adaptation (Paper Table 5)',
        category='core',
        script='experiments/core/table5_margin_distribution/run_table5.py',
        quick_args=['--quick'],
        full_args=[],
        timeout=1800
    ),
    'table8': ExperimentDef(
        id='table8',
        name='Accuracy-Compute Pareto (Table 8)',
        description='FLOPs vs accuracy trade-off, Pareto frontier (Paper Table 8)',
        category='core',
        script='experiments/core/table8_pareto/run_table8.py',
        quick_args=['--quick'],
        full_args=[],
        timeout=1800
    ),
    
    # Validation Experiments
    'val_small': ExperimentDef(
        id='val_small',
        name='Small Model Validation',
        description='Validate Small (2.2B) model architecture and metrics',
        category='validation',
        script='scripts/experiments/run_small_model_experiments_fast.py',
        quick_args=[],
        full_args=[],
        timeout=600
    ),
    'val_rabitq': ExperimentDef(
        id='val_rabitq',
        name='RaBitQ Validation',
        description='Test RaBitQ compression and accuracy',
        category='validation',
        script='scripts/validate_rabitq.py',
        quick_args=['--skip-model'],
        full_args=[],
        timeout=1200
    ),
    'val_rabitq_compression': ExperimentDef(
        id='val_rabitq_compression',
        name='RaBitQ Compression Verification',
        description='Verify RaBitQ compression ratio (~4.9x) and reconstruction quality',
        category='validation',
        script='experiments/rabitq/run_compression_verification.py',
        quick_args=['--quick'],
        full_args=[],
        timeout=600
    ),
    
    # Paper Metrics
    'paper_metrics': ExperimentDef(
        id='paper_metrics',
        name='Paper Metrics Summary',
        description='Generate paper metrics summary (Tables 4-8)',
        category='paper',
        script='scripts/experiments/paper_metrics_summary.py',
        quick_args=[],
        full_args=[],
        timeout=300
    ),
}


def get_device():
    """Get best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


def list_experiments(category: Optional[str] = None):
    """List all available experiments."""
    print("="*70)
    print("Available Experiments")
    print("="*70)
    
    categories = {}
    for exp_id, exp in EXPERIMENTS.items():
        cat = exp.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(exp)
    
    for cat, exps in sorted(categories.items()):
        if category and cat != category:
            continue
        print(f"\n{cat.upper()}:")
        for exp in exps:
            print(f"  {exp.id:12s} - {exp.name}")
            print(f"               {exp.description}")


def run_experiment(
    exp: ExperimentDef,
    quick: bool = False,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
    dry_run: bool = False
) -> ExperimentResult:
    """Run a single experiment."""
    
    print(f"\n{'='*70}")
    print(f"Experiment: {exp.name} ({exp.id})")
    print(f"{'='*70}")
    print(f"Description: {exp.description}")
    print(f"Category: {exp.category}")
    
    script_path = PROJECT_ROOT / exp.script
    if not script_path.exists():
        print(f"⚠️  Script not found: {script_path}")
        return ExperimentResult(
            id=exp.id,
            name=exp.name,
            status='skipped',
            duration=0,
            error_message='Script not found'
        )
    
    # Select args
    args = exp.quick_args if quick else exp.full_args
    
    # Build command
    cmd = [sys.executable, str(script_path)] + args
    
    # Add device
    if device:
        cmd.extend(['--device', device])
    
    # Add output dir
    if output_dir:
        exp_output = output_dir / exp.category / exp.id
        exp_output.mkdir(parents=True, exist_ok=True)
        cmd.extend(['--output-dir', str(exp_output)])
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if dry_run:
        print("🔍 DRY RUN - Not executing")
        return ExperimentResult(
            id=exp.id,
            name=exp.name,
            status='dry_run',
            duration=0
        )
    
    # Run experiment
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=exp.timeout,
            cwd=str(PROJECT_ROOT)
        )
        
        duration = time.time() - start_time
        
        # Save output
        if output_dir:
            log_file = exp_output / 'output.log'
            with open(log_file, 'w') as f:
                f.write(f"STDOUT:\n{result.stdout}\n\n")
                f.write(f"STDERR:\n{result.stderr}\n")
                f.write(f"Return code: {result.returncode}\n")
        else:
            log_file = None
        
        if result.returncode == 0:
            print(f"✅ Success ({duration:.1f}s)")
            status = 'success'
        else:
            print(f"❌ Failed (return code: {result.returncode})")
            status = 'failed'
        
        return ExperimentResult(
            id=exp.id,
            name=exp.name,
            status=status,
            duration=duration,
            output_file=str(log_file) if log_file else None
        )
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏱️  Timeout (>{exp.timeout}s)")
        return ExperimentResult(
            id=exp.id,
            name=exp.name,
            status='timeout',
            duration=duration,
            error_message=f'Timeout after {exp.timeout}s'
        )
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Error: {e}")
        return ExperimentResult(
            id=exp.id,
            name=exp.name,
            status='error',
            duration=duration,
            error_message=str(e)
        )


def run_experiments(
    experiment_ids: List[str],
    quick: bool = False,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
    dry_run: bool = False
) -> List[ExperimentResult]:
    """Run multiple experiments."""
    results = []
    
    for exp_id in experiment_ids:
        if exp_id not in EXPERIMENTS:
            print(f"⚠️  Unknown experiment: {exp_id}")
            continue
        
        result = run_experiment(
            EXPERIMENTS[exp_id],
            quick=quick,
            device=device,
            output_dir=output_dir,
            dry_run=dry_run
        )
        results.append(result)
    
    return results


def print_summary(results: List[ExperimentResult]):
    """Print execution summary."""
    print("\n" + "="*70)
    print("Execution Summary")
    print("="*70)
    
    total = len(results)
    success = sum(1 for r in results if r.status == 'success')
    failed = sum(1 for r in results if r.status == 'failed')
    skipped = sum(1 for r in results if r.status == 'skipped')
    timeout = sum(1 for r in results if r.status == 'timeout')
    
    total_time = sum(r.duration for r in results)
    
    print(f"\nTotal experiments: {total}")
    print(f"  ✅ Success:   {success}")
    print(f"  ❌ Failed:    {failed}")
    print(f"  ⏱️  Timeout:   {timeout}")
    print(f"  ⏭️  Skipped:   {skipped}")
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}m)")
    
    if results:
        print("\nDetails:")
        for r in results:
            status_icon = {
                'success': '✅',
                'failed': '❌',
                'skipped': '⏭️',
                'timeout': '⏱️',
                'error': '💥',
                'dry_run': '🔍'
            }.get(r.status, '❓')
            print(f"  {status_icon} {r.id:12s} {r.status:10s} ({r.duration:.1f}s)")


def save_summary(results: List[ExperimentResult], output_dir: Path):
    """Save execution summary to JSON."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total': len(results),
        'success': sum(1 for r in results if r.status == 'success'),
        'failed': sum(1 for r in results if r.status == 'failed'),
        'skipped': sum(1 for r in results if r.status == 'skipped'),
        'timeout': sum(1 for r in results if r.status == 'timeout'),
        'total_time': sum(r.duration for r in results),
        'experiments': [
            {
                'id': r.id,
                'name': r.name,
                'status': r.status,
                'duration': r.duration,
                'output_file': r.output_file,
                'error_message': r.error_message
            }
            for r in results
        ]
    }
    
    summary_file = output_dir / 'execution_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  %(prog)s --all
  
  # Run all core experiments
  %(prog)s --category core
  
  # Run specific experiments
  %(prog)s --exp exp1 exp2 exp3
  
  # Quick mode (reduced computation)
  %(prog)s --all --quick
  
  # List all experiments
  %(prog)s --list
  
  # Dry run (show what would be executed)
  %(prog)s --all --dry-run
        """
    )
    
    # Selection options
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--category', type=str,
                       choices=['core', 'validation', 'paper', 'all'],
                       help='Run experiments in category')
    parser.add_argument('--exp', '--experiments', nargs='+', dest='experiments',
                       help='Run specific experiments by ID')
    parser.add_argument('--list', action='store_true',
                       help='List available experiments')
    
    # Execution options
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced samples/computation')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (auto, cuda, cpu, mps)')
    parser.add_argument('--output-dir', type=str, default='results/experiments',
                       help='Output directory for results')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Override timeout (seconds)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_experiments(args.category)
        return 0
    
    # Determine what to run
    if args.all:
        experiment_ids = list(EXPERIMENTS.keys())
    elif args.category:
        if args.category == 'all':
            experiment_ids = list(EXPERIMENTS.keys())
        else:
            experiment_ids = [
                eid for eid, e in EXPERIMENTS.items()
                if e.category == args.category
            ]
    elif args.experiments:
        experiment_ids = args.experiments
    else:
        parser.print_help()
        print("\n⚠️  No experiments specified. Use --all, --category, or --exp")
        return 1
    
    # Validate experiment IDs
    invalid = [eid for eid in experiment_ids if eid not in EXPERIMENTS]
    if invalid:
        print(f"⚠️  Invalid experiment IDs: {', '.join(invalid)}")
        print(f"Use --list to see available experiments")
        return 1
    
    # Setup
    device = args.device or get_device()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Unified Experiment Runner")
    print("="*70)
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Experiments: {len(experiment_ids)}")
    if args.dry_run:
        print("Mode: DRY RUN (no actual execution)")
    print("="*70)
    
    # Run experiments
    results = run_experiments(
        experiment_ids,
        quick=args.quick,
        device=device,
        output_dir=output_dir,
        dry_run=args.dry_run
    )
    
    # Print summary
    print_summary(results)
    
    # Save summary
    if not args.dry_run:
        save_summary(results, output_dir)
    
    # Return code
    failed = sum(1 for r in results if r.status in ('failed', 'timeout', 'error'))
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
