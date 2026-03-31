#!/usr/bin/env python3
"""
Common Experiment Runner for all scripts.

Provides unified interface for running experiments with:
- Consistent logging
- Result tracking
- Error handling
- Progress reporting
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class RunConfig:
    """Configuration for experiment runs."""
    name: str
    description: str = ""
    output_dir: Path = None
    verbose: bool = True
    save_results: bool = True
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = Path("results") / self.name
        self.output_dir = Path(self.output_dir)


@dataclass
class RunResult:
    """Result of an experiment run."""
    success: bool
    duration: float
    data: Dict[str, Any]
    error: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'duration': self.duration,
            'data': self.data,
            'error': self.error,
            'timestamp': self.timestamp
        }


class ExperimentLogger:
    """Unified logger for experiments."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logs: List[Dict[str, Any]] = []
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        self.logs.append(log_entry)
        
        if self.verbose:
            print(f"[{timestamp}] [{level}] {message}")
    
    def info(self, message: str):
        self.log(message, "INFO")
    
    def warning(self, message: str):
        self.log(message, "WARNING")
    
    def error(self, message: str):
        self.log(message, "ERROR")
    
    def success(self, message: str):
        self.log(message, "SUCCESS")
    
    def get_logs(self) -> List[Dict[str, Any]]:
        return self.logs


class ExperimentRunner:
    """
    Unified experiment runner.
    
    Usage:
        runner = ExperimentRunner(config)
        
        @runner.experiment(name="my_experiment")
        def my_experiment():
            # Experiment code
            return {"metric": value}
        
        result = runner.run("my_experiment")
    """
    
    def __init__(self, config: RunConfig):
        self.config = config
        self.logger = ExperimentLogger(config.verbose)
        self.experiments: Dict[str, Callable] = {}
        self.results: Dict[str, RunResult] = {}
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def experiment(self, name: Optional[str] = None):
        """Decorator to register an experiment function."""
        def decorator(func: Callable) -> Callable:
            exp_name = name or func.__name__
            self.experiments[exp_name] = func
            return func
        return decorator
    
    @contextmanager
    def _timer(self):
        """Context manager for timing experiments."""
        start = time.time()
        yield
        end = time.time()
        self._last_duration = end - start
    
    def run(self, name: str, **kwargs) -> RunResult:
        """
        Run a registered experiment.
        
        Args:
            name: Name of the experiment to run
            **kwargs: Arguments to pass to the experiment function
            
        Returns:
            RunResult with success status and data
        """
        if name not in self.experiments:
            raise ValueError(f"Unknown experiment: {name}")
        
        self.logger.info(f"Starting experiment: {name}")
        exp_func = self.experiments[name]
        
        start_time = time.time()
        
        try:
            with self._timer():
                data = exp_func(logger=self.logger, **kwargs)
            
            duration = time.time() - start_time
            result = RunResult(
                success=True,
                duration=duration,
                data=data or {}
            )
            
            self.logger.success(f"Experiment '{name}' completed in {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            traceback_str = traceback.format_exc()
            
            result = RunResult(
                success=False,
                duration=duration,
                data={},
                error=f"{error_msg}\n{traceback_str}"
            )
            
            self.logger.error(f"Experiment '{name}' failed: {error_msg}")
        
        self.results[name] = result
        
        # Save results
        if self.config.save_results:
            self._save_result(name, result)
        
        return result
    
    def run_all(self, names: Optional[List[str]] = None, **kwargs) -> Dict[str, RunResult]:
        """
        Run multiple experiments.
        
        Args:
            names: List of experiment names (None = all registered)
            **kwargs: Arguments to pass to each experiment
            
        Returns:
            Dictionary of experiment names to RunResults
        """
        names = names or list(self.experiments.keys())
        
        self.logger.info(f"Running {len(names)} experiments: {', '.join(names)}")
        
        for name in names:
            self.run(name, **kwargs)
        
        # Save summary
        self._save_summary()
        
        return self.results
    
    def _save_result(self, name: str, result: RunResult):
        """Save individual experiment result."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = self.config.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        self.logger.info(f"Results saved to: {filepath}")
    
    def _save_summary(self):
        """Save experiment summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'name': self.config.name,
                'description': self.config.description,
                'output_dir': str(self.config.output_dir)
            },
            'results': {
                name: result.to_dict()
                for name, result in self.results.items()
            },
            'summary': {
                'total': len(self.results),
                'successful': sum(1 for r in self.results.values() if r.success),
                'failed': sum(1 for r in self.results.values() if not r.success),
                'total_duration': sum(r.duration for r in self.results.values())
            },
            'logs': self.logger.get_logs()
        }
        
        summary_path = self.config.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary saved to: {summary_path}")
    
    def print_summary(self):
        """Print experiment summary."""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        
        total = len(self.results)
        successful = sum(1 for r in self.results.values() if r.success)
        failed = total - successful
        total_duration = sum(r.duration for r in self.results.values())
        
        print(f"\nTotal experiments: {total}")
        print(f"Successful: {successful} ✅")
        print(f"Failed: {failed} {'✅' if failed == 0 else '❌'}")
        print(f"Total duration: {total_duration:.2f}s")
        
        print("\nIndividual Results:")
        for name, result in self.results.items():
            status = "✅" if result.success else "❌"
            print(f"  {status} {name}: {result.duration:.2f}s")
        
        print("=" * 80)


def run_simple_experiment(
    name: str,
    experiment_func: Callable,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    **kwargs
) -> RunResult:
    """
    Simple wrapper to run a single experiment.
    
    Args:
        name: Experiment name
        experiment_func: Function to run
        output_dir: Output directory
        verbose: Whether to print logs
        **kwargs: Arguments to pass to experiment function
        
    Returns:
        RunResult
    """
    config = RunConfig(
        name=name,
        output_dir=output_dir or f"results/{name}",
        verbose=verbose
    )
    
    runner = ExperimentRunner(config)
    
    @runner.experiment(name="main")
    def main_experiment(logger, **kw):
        return experiment_func(logger=logger, **kw)
    
    return runner.run("main", **kwargs)
