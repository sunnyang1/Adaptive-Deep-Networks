"""
Path Management for Experiments

Standardized output directory structure.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import datetime


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_results_dir() -> Path:
    """Get the results directory."""
    results_dir = get_project_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


@dataclass
class OutputPaths:
    """
    Standardized output path structure for experiments.
    
    Structure:
        results/
        └── {category}/
            └── {experiment_name}/
                ├── config.yaml          # Experiment configuration
                ├── results.json         # Numerical results
                ├── report.md            # Human-readable report
                ├── progress.log         # Execution log
                └── figures/             # Visualization outputs
                    ├── figure1.png
                    └── ...
    """
    base_dir: Path
    
    def __post_init__(self):
        # Ensure base_dir is Path
        if isinstance(self.base_dir, str):
            object.__setattr__(self, 'base_dir', Path(self.base_dir))
        
        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def config(self) -> Path:
        """Path to configuration file."""
        return self.base_dir / "config.yaml"
    
    @property
    def results(self) -> Path:
        """Path to JSON results file."""
        return self.base_dir / "results.json"
    
    @property
    def report(self) -> Path:
        """Path to Markdown report."""
        return self.base_dir / "report.md"
    
    @property
    def log(self) -> Path:
        """Path to progress log file."""
        return self.base_dir / "progress.log"
    
    @property
    def figures_dir(self) -> Path:
        """Directory for figures."""
        return self.base_dir / "figures"
    
    def figure(self, name: str, ext: str = "png") -> Path:
        """Get path for a specific figure."""
        return self.figures_dir / f"{name}.{ext}"
    
    def checkpoint(self, epoch: Optional[int] = None) -> Path:
        """Get path for checkpoint file."""
        if epoch is not None:
            return self.base_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt"
        return self.base_dir / "checkpoints" / "checkpoint_latest.pt"
    
    @classmethod
    def for_experiment(
        cls,
        name: str,
        category: str = "core",
        timestamp: bool = False
    ) -> "OutputPaths":
        """
        Create output paths for an experiment.
        
        Args:
            name: Experiment name
            category: Experiment category (core, validation, real_model)
            timestamp: Whether to include timestamp in directory name
        
        Returns:
            OutputPaths instance
        """
        project_root = get_project_root()
        
        if timestamp:
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = f"{name}_{time_str}"
        else:
            exp_dir = name
        
        base = project_root / "results" / category / exp_dir
        return cls(base)
    
    @classmethod
    def from_config(cls, config) -> "OutputPaths":
        """Create output paths from experiment config."""
        return cls.for_experiment(
            name=config.name,
            category=config.category,
            timestamp=False
        )
    
    def exists(self) -> bool:
        """Check if output directory exists."""
        return self.base_dir.exists()
    
    def list_figures(self) -> list:
        """List all generated figures."""
        if not self.figures_dir.exists():
            return []
        return sorted(self.figures_dir.glob("*.png"))
    
    def get_summary(self) -> dict:
        """Get summary of output directory contents."""
        summary = {
            'base_dir': str(self.base_dir),
            'exists': self.exists(),
            'has_config': self.config.exists(),
            'has_results': self.results.exists(),
            'has_report': self.report.exists(),
            'has_log': self.log.exists(),
            'num_figures': len(self.list_figures()),
        }
        return summary


def get_cache_dir() -> Path:
    """Get cache directory for temporary files."""
    cache_dir = get_project_root() / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def get_checkpoint_dir() -> Path:
    """Get default checkpoint directory."""
    checkpoint_dir = get_project_root() / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir
