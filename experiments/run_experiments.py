#!/usr/bin/env python3
"""
Unified Experiment Runner

Replaces the three separate runner scripts:
- run_all.py
- run_all_experiments.py  
- run_all_validations.py

Usage:
    # Run all experiments
    python experiments/run_experiments.py --all
    
    # Run specific category
    python experiments/run_experiments.py --category core
    python experiments/run_experiments.py --category validation
    
    # Run specific experiments
    python experiments/run_experiments.py --experiments exp1 exp2
    
    # List available experiments
    python experiments/run_experiments.py --list
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.common import ExperimentConfig, get_project_root
from experiments.runner import ExperimentRunner, discover_experiments, list_all_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments with unified runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                          # Run all experiments
  %(prog)s --category core                # Run core experiments only
  %(prog)s --category validation          # Run validation scripts
  %(prog)s --experiments exp1 exp2        # Run specific experiments
  %(prog)s --list                         # List available experiments
  %(prog)s --dry-run --all                # Show what would be run
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--category', type=str,
                       choices=['core', 'validation', 'real_model'],
                       help='Run experiments in category')
    parser.add_argument('--experiments', nargs='+',
                       help='Run specific experiments by name')
    parser.add_argument('--list', action='store_true',
                       help='List available experiments')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Timeout per experiment (seconds)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced samples')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run even if results exist')
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("="*60)
        print("Available Experiments")
        print("="*60)
        
        experiments = list_all_experiments()
        for category, names in experiments.items():
            print(f"\n{category.upper()}:")
            for name in names:
                print(f"  - {name}")
        
        return
    
    # Determine what to run
    if args.all:
        categories = ['core', 'validation', 'real_model']
        experiment_names = None
    elif args.category:
        categories = [args.category]
        experiment_names = None
    elif args.experiments:
        categories = None
        experiment_names = args.experiments
    else:
        parser.print_help()
        return
    
    # Discover experiments
    project_root = get_project_root()
    all_experiments = discover_experiments(project_root / "experiments")
    
    # Filter experiments
    to_run = []
    for exp in all_experiments:
        if categories and exp['category'] in categories:
            to_run.append(exp)
        elif experiment_names and exp['name'] in experiment_names:
            to_run.append(exp)
    
    if not to_run:
        print("No experiments to run!")
        return
    
    # Dry run mode
    if args.dry_run:
        print("="*60)
        print("Dry Run - Would execute:")
        print("="*60)
        for exp in to_run:
            print(f"  [{exp['category']}] {exp['name']}")
            print(f"    Script: {exp['script_path']}")
            if exp['description']:
                print(f"    Description: {exp['description']}")
        return
    
    # Run experiments
    print("="*60)
    print("Experiment Runner")
    print("="*60)
    print(f"Total experiments: {len(to_run)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print()
    
    runner = ExperimentRunner(
        output_dir=Path(args.output_dir),
        timeout=args.timeout,
        verbose=True
    )
    
    for exp in to_run:
        print(f"\n{'='*60}")
        print(f"Running: {exp['name']} ({exp['category']})")
        print('='*60)
        
        # Build command line args
        script_args = [
            f'--device={args.device}',
            f'--output-dir={args.output_dir}/{exp["category"]}/{exp["name"]}',
        ]
        
        if args.quick:
            script_args.append('--quick')
        
        # Run script
        result = runner.run_script(exp['script_path'], script_args)
        
        # Print result
        status = "✅ PASS" if result.success else "❌ FAIL"
        print(f"{status}: {result.duration_seconds:.2f}s")
        
        if not result.success and result.error:
            print(f"Error: {result.error[:200]}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    runner.print_summary()
    
    # Save summary
    summary_path = Path(args.output_dir) / "summary.md"
    runner.generate_summary(summary_path)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
