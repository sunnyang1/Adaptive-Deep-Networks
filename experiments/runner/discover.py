"""
Experiment Discovery

Automatically discover and register experiments from directory structure.
"""

import importlib.util
import sys
from pathlib import Path
from typing import List, Dict, Type, Optional
import inspect

from .base import BaseExperiment, ExperimentRegistry


def discover_experiments(
    root_dir: Path,
    pattern: str = "run_*.py"
) -> List[Dict[str, any]]:
    """
    Discover experiments in directory structure.
    
    Args:
        root_dir: Root directory to search
        pattern: Pattern for experiment scripts
    
    Returns:
        List of experiment metadata dictionaries
    """
    root_dir = Path(root_dir)
    experiments = []
    
    for script_path in root_dir.rglob(pattern):
        # Extract category from parent directory name
        category = script_path.parent.parent.name
        if category == "experiments":
            category = "core"
        
        exp_name = script_path.parent.name
        
        experiments.append({
            'name': exp_name,
            'category': category,
            'script_path': script_path,
            'description': extract_docstring(script_path),
        })
    
    return sorted(experiments, key=lambda x: (x['category'], x['name']))


def extract_docstring(script_path: Path) -> str:
    """Extract first line of docstring from Python file."""
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Look for triple-quoted string at start
        if '"""' in content:
            start = content.find('"""') + 3
            end = content.find('"""', start)
            if end > start:
                return content[start:end].strip().split('\n')[0]
    except Exception:
        pass
    
    return ""


def load_experiment_class(
    script_path: Path,
    class_name: Optional[str] = None
) -> Optional[Type[BaseExperiment]]:
    """
    Load experiment class from script.
    
    Args:
        script_path: Path to Python file
        class_name: Specific class to load (None for auto-detect)
    
    Returns:
        Experiment class or None
    """
    script_path = Path(script_path)
    if not script_path.exists():
        return None
    
    # Load module
    spec = importlib.util.spec_from_file_location(
        script_path.stem,
        script_path
    )
    module = importlib.util.module_from_spec(spec)
    
    # Add project root to path for imports
    project_root = script_path.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    spec.loader.exec_module(module)
    
    # Find experiment class
    if class_name:
        return getattr(module, class_name, None)
    
    # Auto-detect: find first class inheriting from BaseExperiment
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseExperiment) and obj is not BaseExperiment:
            return obj
    
    return None


def create_registry_from_directory(
    root_dir: Path,
    pattern: str = "run_*.py"
) -> ExperimentRegistry:
    """
    Create experiment registry by discovering scripts.
    
    Args:
        root_dir: Root directory
        pattern: Pattern for scripts
    
    Returns:
        Populated ExperimentRegistry
    """
    registry = ExperimentRegistry()
    experiments = discover_experiments(root_dir, pattern)
    
    for exp_info in experiments:
        try:
            exp_class = load_experiment_class(exp_info['script_path'])
            if exp_class:
                experiment = exp_class(name=exp_info['name'])
                experiment.category = exp_info['category']
                registry.register(exp_info['name'], experiment)
        except Exception as e:
            print(f"Warning: Failed to load {exp_info['name']}: {e}")
    
    return registry


def list_all_experiments(project_root: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    List all available experiments by category.
    
    Args:
        project_root: Project root (default: auto-detect)
    
    Returns:
        Dictionary mapping category to list of experiment names
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    
    experiments_dir = project_root / "experiments"
    
    by_category = {}
    
    for category_dir in ["core", "validation", "real_model"]:
        category_path = experiments_dir / category_dir
        if not category_path.exists():
            continue
        
        experiments = []
        for exp_dir in category_path.iterdir():
            if exp_dir.is_dir():
                # Look for run_*.py files
                run_scripts = list(exp_dir.glob("run_*.py"))
                if run_scripts:
                    experiments.append(exp_dir.name)
        
        if experiments:
            by_category[category_dir] = sorted(experiments)
    
    return by_category
