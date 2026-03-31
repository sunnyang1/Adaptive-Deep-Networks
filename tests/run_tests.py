#!/usr/bin/env python3
"""
Unified Test Runner for Adaptive Deep Networks.

Usage:
    # Run all tests
    python tests/run_tests.py
    
    # Run specific test module
    python tests/run_tests.py --module unit.test_attnres
    
    # Run with coverage
    python tests/run_tests.py --coverage
    
    # Run quick tests only
    python tests/run_tests.py --quick
    
    # List available tests
    python tests/run_tests.py --list
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Test module registry
TEST_MODULES = {
    'unit': {
        'test_attnres': 'Attention Residuals tests',
        'test_attnres_integration': 'AttnRes integration tests',
        'test_models': 'Model architecture tests',
        'test_gating': 'Dynamic gating tests',
        'test_qttt': 'qTTT tests',
        'test_turboquant_v3': 'TurboQuant V3 tests',
        'test_benchmarks': 'Benchmark tests',
    },
    'integration': {
        'test_small_model': 'Small model end-to-end tests',
        'test_medium_model': 'Medium model tests (if available)',
    },
    'legacy': {
        'test_models_simple': 'Simple model tests',
        'test_gating_simple': 'Simple gating tests',
    }
}


class TestRunner:
    """Unified test runner."""
    
    def __init__(self, verbose: bool = True, coverage: bool = False):
        self.verbose = verbose
        self.coverage = coverage
        self.results: Dict[str, Any] = {}
        
    def log(self, message: str):
        """Log a message."""
        if self.verbose:
            print(message)
    
    def list_tests(self):
        """List all available tests."""
        print("\nAvailable Test Modules:")
        print("=" * 60)
        
        for category, modules in TEST_MODULES.items():
            print(f"\n{category.upper()}:")
            for module, description in modules.items():
                print(f"  - {module}: {description}")
        
        print("\n" + "=" * 60)
    
    def run_module(self, module_path: str, quick: bool = False) -> Dict[str, Any]:
        """
        Run a single test module.
        
        Args:
            module_path: Path to test module (e.g., 'unit.test_attnres')
            quick: Whether to run in quick mode
            
        Returns:
            Test results dictionary
        """
        self.log(f"\nRunning: {module_path}")
        
        # Construct pytest command
        cmd = ['python', '-m', 'pytest', f'tests/{module_path.replace(".", "/")}.py', '-v']
        
        if quick:
            cmd.extend(['-x', '--tb=short'])  # Exit on first failure, short traceback
        
        if self.coverage:
            cmd = ['python', '-m', 'pytest', f'--cov=src', f'--cov-report=term-missing'] + cmd[3:]
        
        # Run tests
        start_time = datetime.now()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # Parse results
        success = result.returncode == 0
        
        test_result = {
            'module': module_path,
            'success': success,
            'duration': duration,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr if not success else None,
        }
        
        # Print output
        if self.verbose:
            print(result.stdout)
            if not success:
                print(result.stderr)
        
        return test_result
    
    def run_category(self, category: str, quick: bool = False) -> List[Dict[str, Any]]:
        """
        Run all tests in a category.
        
        Args:
            category: Category name (unit, integration, legacy)
            quick: Whether to run in quick mode
            
        Returns:
            List of test results
        """
        if category not in TEST_MODULES:
            print(f"Unknown category: {category}")
            print(f"Available: {list(TEST_MODULES.keys())}")
            return []
        
        self.log(f"\n{'='*60}")
        self.log(f"Running {category.upper()} tests")
        self.log('='*60)
        
        results = []
        for module in TEST_MODULES[category].keys():
            result = self.run_module(f"{category}.{module}", quick=quick)
            results.append(result)
            self.results[f"{category}.{module}"] = result
        
        return results
    
    def run_all(self, quick: bool = False) -> Dict[str, Any]:
        """
        Run all tests.
        
        Args:
            quick: Whether to run in quick mode
            
        Returns:
            Summary of all test results
        """
        print("\n" + "=" * 60)
        print("ADAPTIVE DEEP NETWORKS - TEST SUITE")
        print("=" * 60)
        
        start_time = datetime.now()
        
        all_results = []
        for category in ['unit', 'integration', 'legacy']:
            if category in TEST_MODULES:
                results = self.run_category(category, quick=quick)
                all_results.extend(results)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Generate summary
        passed = sum(1 for r in all_results if r['success'])
        failed = sum(1 for r in all_results if not r['success'])
        total = len(all_results)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_duration': total_duration,
            'total_modules': total,
            'passed': passed,
            'failed': failed,
            'results': self.results
        }
        
        # Print summary
        self.print_summary(summary)
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal modules: {summary['total_modules']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} {'✅' if summary['failed'] == 0 else '❌'}")
        print(f"Total duration: {summary['total_duration']:.2f}s")
        
        if summary['failed'] > 0:
            print("\nFailed tests:")
            for name, result in self.results.items():
                if not result['success']:
                    print(f"  ❌ {name}")
        
        print("=" * 60)
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save test results to file."""
        output_dir = Path('results/tests')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"test_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run ADN Test Suite')
    parser.add_argument('--module', type=str, help='Run specific test module (e.g., unit.test_attnres)')
    parser.add_argument('--category', type=str, choices=['unit', 'integration', 'legacy'],
                       help='Run all tests in category')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--list', action='store_true', help='List available tests')
    parser.add_argument('--quick', action='store_true', help='Quick mode (exit on first failure)')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    
    args = parser.parse_args()
    
    runner = TestRunner(
        verbose=args.verbose or not args.quiet,
        coverage=args.coverage
    )
    
    if args.list:
        runner.list_tests()
    elif args.module:
        result = runner.run_module(args.module, quick=args.quick)
        sys.exit(0 if result['success'] else 1)
    elif args.category:
        results = runner.run_category(args.category, quick=args.quick)
        failed = sum(1 for r in results if not r['success'])
        sys.exit(0 if failed == 0 else 1)
    else:
        # Default: run unit tests
        summary = runner.run_all(quick=args.quick)
        sys.exit(0 if summary['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
