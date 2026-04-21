"""
Base Classes for Experiments

Defines the interface that all experiments must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import json

from experiments.common.config import ExperimentConfig
from experiments.common.paths import OutputPaths


@dataclass
class ExperimentResult:
    """Result container for experiment execution."""
    
    name: str
    success: bool
    duration_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Path] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'success': self.success,
            'duration_seconds': self.duration_seconds,
            'metrics': self.metrics,
            'outputs': {k: str(v) for k, v in self.outputs.items()},
            'error': self.error,
            'timestamp': self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResult':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            success=data['success'],
            duration_seconds=data['duration_seconds'],
            metrics=data.get('metrics', {}),
            outputs={k: Path(v) for k, v in data.get('outputs', {}).items()},
            error=data.get('error'),
            timestamp=datetime.fromisoformat(data['timestamp']),
        )
    
    def save(self, path: Path) -> None:
        """Save result to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class BaseExperiment(ABC):
    """
    Abstract base class for all experiments.
    
    All experiments must implement:
    - run(): Execute the experiment
    - visualize(): Generate visualizations
    - generate_report(): Create human-readable report
    
    Example:
        >>> class MyExperiment(BaseExperiment):
        ...     def run(self, config):
        ...         # Run experiment
        ...         return ExperimentResult(name="my_exp", success=True)
        ...     
        ...     def visualize(self, result, output_dir):
        ...         # Create plots
        ...         pass
        ...     
        ...     def generate_report(self, result):
        ...         return "# Report\n..."
    """
    
    def __init__(self, name: str, category: str = "core"):
        self.name = name
        self.category = category
        self.config: Optional[ExperimentConfig] = None
        self.output_paths: Optional[OutputPaths] = None
    
    @abstractmethod
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Execute the experiment.
        
        Args:
            config: Experiment configuration
        
        Returns:
            ExperimentResult with metrics and status
        """
        pass
    
    def visualize(self, result: ExperimentResult, output_dir: Path) -> List[Path]:
        """
        Generate visualizations from results.
        
        Args:
            result: Experiment result
            output_dir: Directory to save figures
        
        Returns:
            List of generated figure paths
        """
        # Default: no visualizations
        return []
    
    def generate_report(self, result: ExperimentResult) -> str:
        """
        Generate human-readable report.
        
        Args:
            result: Experiment result
        
        Returns:
            Markdown-formatted report
        """
        lines = [
            f"# {self.name} Results",
            "",
            f"**Status**: {'✅ PASS' if result.success else '❌ FAIL'}",
            f"**Duration**: {result.duration_seconds:.2f} seconds",
            f"**Timestamp**: {result.timestamp.isoformat()}",
            "",
            "## Metrics",
            "",
        ]
        
        for key, value in result.metrics.items():
            lines.append(f"- **{key}**: {value}")
        
        if result.error:
            lines.extend([
                "",
                "## Error",
                "",
                f"```\n{result.error}\n```",
            ])
        
        return "\n".join(lines)
    
    def execute(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Full experiment execution pipeline.
        
        Runs: setup -> run -> visualize -> report -> save
        
        Args:
            config: Experiment configuration
        
        Returns:
            ExperimentResult
        """
        from experiments.common.logging_config import ExperimentLogger
        
        # Setup
        self.config = config
        self.output_paths = OutputPaths.from_config(config)
        
        logger = ExperimentLogger(config.name, self.output_paths.base_dir)
        logger.log_phase("setup", "started")
        
        # Save config
        config.to_yaml(self.output_paths.config)
        logger.log_phase("setup", "completed")
        
        # Run experiment
        logger.log_phase("run", "started")
        start_time = datetime.now()
        
        try:
            result = self.run(config)
            result.outputs['config'] = self.output_paths.config
        except Exception as e:
            result = ExperimentResult(
                name=self.name,
                success=False,
                error=str(e),
            )
            logger.log_error(e, "Experiment failed")
        
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        logger.log_phase("run", "completed" if result.success else "failed")
        
        # Visualize
        if result.success:
            logger.log_phase("visualize", "started")
            try:
                figures = self.visualize(result, self.output_paths.figures_dir)
                result.outputs['figures'] = figures
                logger.log_phase("visualize", "completed", num_figures=len(figures))
            except Exception as e:
                logger.log_error(e, "Visualization failed")
        
        # Generate report
        logger.log_phase("report", "started")
        try:
            report = self.generate_report(result)
            self.output_paths.report.write_text(report)
            result.outputs['report'] = self.output_paths.report
            logger.log_phase("report", "completed")
        except Exception as e:
            logger.log_error(e, "Report generation failed")
        
        # Save results
        result.save(self.output_paths.results)
        
        return result


class ExperimentRegistry:
    """
    Registry for discovering and running experiments.
    
    Example:
        >>> registry = ExperimentRegistry()
        >>> registry.register("exp1", MyExperiment())
        >>> result = registry.run("exp1", config)
    """
    
    def __init__(self):
        self._experiments: Dict[str, BaseExperiment] = {}
        self._results: Dict[str, ExperimentResult] = {}
    
    def register(self, name: str, experiment: BaseExperiment) -> 'ExperimentRegistry':
        """Register an experiment."""
        self._experiments[name] = experiment
        return self
    
    def get(self, name: str) -> Optional[BaseExperiment]:
        """Get experiment by name."""
        return self._experiments.get(name)
    
    def list_experiments(self, category: Optional[str] = None) -> List[str]:
        """List registered experiments."""
        names = []
        for name, exp in self._experiments.items():
            if category is None or exp.category == category:
                names.append(name)
        return sorted(names)
    
    def run(
        self,
        name: str,
        config: ExperimentConfig,
        force: bool = False
    ) -> ExperimentResult:
        """
        Run a registered experiment.
        
        Args:
            name: Experiment name
            config: Experiment configuration
            force: Re-run even if results exist
        
        Returns:
            ExperimentResult
        """
        experiment = self.get(name)
        if experiment is None:
            raise ValueError(f"Experiment '{name}' not found")
        
        # Check for existing results
        if not force:
            output_paths = OutputPaths.from_config(config)
            if output_paths.results.exists():
                print(f"Results already exist for {name}. Use force=True to re-run.")
                return ExperimentResult.from_dict(
                    json.loads(output_paths.results.read_text())
                )
        
        # Run experiment
        result = experiment.execute(config)
        self._results[name] = result
        
        return result
    
    def run_all(
        self,
        config_factory,
        categories: Optional[List[str]] = None,
        parallel: bool = False
    ) -> Dict[str, ExperimentResult]:
        """
        Run all registered experiments.
        
        Args:
            config_factory: Function that creates config for each experiment
            categories: Filter by categories
            parallel: Run in parallel (not implemented)
        
        Returns:
            Dictionary of results
        """
        results = {}
        
        for name in self.list_experiments(categories):
            config = config_factory(name)
            results[name] = self.run(name, config)
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        return {
            'total': len(self._experiments),
            'completed': len(self._results),
            'categories': list(set(exp.category for exp in self._experiments.values())),
            'experiments': {
                name: {
                    'category': exp.category,
                    'has_result': name in self._results,
                    'success': self._results.get(name, ExperimentResult(name, False)).success,
                }
                for name, exp in self._experiments.items()
            }
        }
