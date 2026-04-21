"""
Experiment Runner Implementation

Orchestrates experiment execution with progress tracking and error handling.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime
import json
import time

from .base import ExperimentResult
from experiments.common.paths import OutputPaths


class ExperimentRunner:
    """
    Runner for executing experiments via subprocess or direct import.
    
    Supports both:
    - Script-based experiments (subprocess)
    - Class-based experiments (direct import)
    
    Example:
        >>> runner = ExperimentRunner(output_dir="results")
        >>> runner.run_script("experiments/core/exp1/run_exp1.py")
        >>> runner.run_all_in_directory("experiments/core/")
    """
    
    def __init__(
        self,
        output_dir: Path = Path("results"),
        timeout: int = 3600,
        verbose: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        self.verbose = verbose
        self.results: List[ExperimentResult] = []
        self.errors: List[tuple] = []
    
    def run_script(
        self,
        script_path: Path,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None
    ) -> ExperimentResult:
        """
        Run an experiment script via subprocess.
        
        Args:
            script_path: Path to Python script
            args: Command line arguments
            env: Environment variables
        
        Returns:
            ExperimentResult
        """
        script_path = Path(script_path)
        if not script_path.exists():
            return ExperimentResult(
                name=script_path.stem,
                success=False,
                error=f"Script not found: {script_path}"
            )
        
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        if self.verbose:
            print(f"Running: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**dict(os.environ), **(env or {})}
            )
            
            duration = time.time() - start_time
            
            # Try to parse JSON results if output to stdout
            metrics = {}
            try:
                # Look for JSON in stdout
                for line in result.stdout.split('\n'):
                    if line.strip().startswith('{'):
                        metrics = json.loads(line)
                        break
            except json.JSONDecodeError:
                pass
            
            success = result.returncode == 0
            
            exp_result = ExperimentResult(
                name=script_path.stem,
                success=success,
                duration_seconds=duration,
                metrics=metrics,
                error=result.stderr if not success else None,
            )
            
        except subprocess.TimeoutExpired:
            exp_result = ExperimentResult(
                name=script_path.stem,
                success=False,
                error=f"Timeout after {self.timeout} seconds"
            )
        except Exception as e:
            exp_result = ExperimentResult(
                name=script_path.stem,
                success=False,
                error=str(e)
            )
        
        self.results.append(exp_result)
        return exp_result
    
    def run_all_in_directory(
        self,
        directory: Path,
        pattern: str = "run_exp*.py",
        args: Optional[List[str]] = None
    ) -> Dict[str, ExperimentResult]:
        """
        Run all experiment scripts in a directory.
        
        Args:
            directory: Directory to search
            pattern: Glob pattern for scripts
            args: Arguments to pass to each script
        
        Returns:
            Dictionary mapping script names to results
        """
        directory = Path(directory)
        scripts = sorted(directory.glob(pattern))
        
        results = {}
        
        for script in scripts:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Running {script.name}")
                print('='*60)
            
            result = self.run_script(script, args)
            results[script.stem] = result
            
            if self.verbose:
                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"{status}: {result.duration_seconds:.2f}s")
        
        return results
    
    def generate_summary(self, output_path: Optional[Path] = None) -> str:
        """
        Generate summary report of all runs.
        
        Args:
            output_path: Path to save summary
        
        Returns:
            Markdown-formatted summary
        """
        lines = [
            "# Experiment Run Summary",
            "",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Total**: {len(self.results)}",
            f"**Passed**: {sum(1 for r in self.results if r.success)}",
            f"**Failed**: {sum(1 for r in self.results if not r.success)}",
            "",
            "## Results",
            "",
            "| Experiment | Status | Duration | Metrics |",
            "|------------|--------|----------|---------|",
        ]
        
        for result in self.results:
            status = "✅" if result.success else "❌"
            duration = f"{result.duration_seconds:.2f}s"
            metrics = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in list(result.metrics.items())[:3])
            lines.append(f"| {result.name} | {status} | {duration} | {metrics} |")
        
        summary = "\n".join(lines)
        
        if output_path:
            output_path.write_text(summary)
        
        return summary
    
    def print_summary(self):
        """Print summary to console."""
        print(self.generate_summary())


import os  # For environment in subprocess
