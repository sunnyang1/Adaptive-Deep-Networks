# Migration Guide: Experiments & Scripts

This guide helps migrate existing experiments and scripts to the new architecture.

## Quick Reference

### Migrating an Experiment

**Before:**
```python
# run_exp1.py - 250 lines of standalone code
def run_experiment(config):
    # Hardcoded colors
    colors = {'prenorm': '#e74c3c', ...}
    
    # Inline model creation
    class SimpleModel(nn.Module): ...
    
    # Manual path setup
    sys.path.insert(0, os.path.join(...))
    
    # Inline validation logic
    if abs(actual - target) / target < 0.15: ...
    
    # String-based report generation
    report = "# Title\n..."
    with open(..., 'w') as f:
        f.write(report)

def main():
    parser = argparse.ArgumentParser()
    # ... 20 lines of arg parsing
    args = parser.parse_args()
    config = {...}
    results = run_experiment(config)
    # Save, visualize, report
```

**After:**
```python
# experiment.py - ~100 lines using base class
from experiments.core.base_core_experiment import CoreExperiment

class MyExperiment(CoreExperiment):
    def __init__(self):
        super().__init__(name="my_exp", config_path="config.yaml")
    
    def run(self, config):
        # Use self.create_model(arch)
        # Use self.get_architecture_color(arch)
        # Use validate_target() from mixin
        return ExperimentResult(...)
    
    def visualize(self, result, output_dir):
        # Use plot_architecture_comparison()
        return [figure_paths]
    
    def generate_report(self, result):
        # Override base report
        return markdown

def main():
    # Only need to handle CLI args
    experiment = MyExperiment()
    result = experiment.execute(config)
```

## Step-by-Step Migration

### 1. Create YAML Config

Create `experiments/core/my_experiment/config.yaml`:

```yaml
name: "my_experiment"
category: "core"
description: "Brief description"

model:
  architectures:
    - prenorm
    - postnorm
    - deepnorm
    - attnres
  num_layers: 96
  d_model: 4096

experiment:
  num_samples: 100
  seq_len: 512

visualization:
  colors:
    prenorm: "#e74c3c"
    postnorm: "#3498db"
    deepnorm: "#2ecc71"
    attnres: "#9b59b6"
```

### 2. Create Experiment Class

Create `experiments/core/my_experiment/experiment.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import yaml
from experiments.core.base_core_experiment import CoreExperiment, ValidationMixin
from experiments.runner import ExperimentResult
from experiments.common import ExperimentConfig

class MyExperiment(CoreExperiment, ValidationMixin):
    def __init__(self):
        super().__init__(
            name="my_experiment",
            config_path=Path(__file__).parent / "config.yaml"
        )
    
    def setup(self, config: ExperimentConfig) -> None:
        super().setup(config)
        
        # Load YAML config
        with open(self.config_path, 'r') as f:
            self.yaml_config = yaml.safe_load(f)
    
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Main experiment logic."""
        self.setup(config)
        
        # Access config values
        architectures = self.yaml_config['model']['architectures']
        
        results = {}
        for arch in architectures:
            # Use base class method to create model
            model = self.create_model(arch)
            
            # Run your experiment logic
            metrics = self._run_single_experiment(model, arch)
            results[arch] = metrics
        
        return ExperimentResult(
            name=self.name,
            success=True,
            metrics={'architectures': results}
        )
    
    def _run_single_experiment(self, model, arch):
        """Implement your measurement logic."""
        # Your code here
        return {'metric': value}
    
    def visualize(self, result, output_dir):
        """Generate plots."""
        from experiments.common.visualization import plot_architecture_comparison
        
        # Use standard plotting functions
        fig_path = plot_architecture_comparison(
            data={arch: {'metric': data['metric']}
                  for arch, data in result.metrics['architectures'].items()},
            metric='metric',
            output_path=output_dir / 'comparison.png',
            title='My Experiment Results'
        )
        
        return [fig_path]
    
    def generate_report(self, result):
        """Generate markdown report."""
        lines = [
            f"# {self.name} Results",
            "",
            "## Results",
            "",
        ]
        
        for arch, data in result.metrics['architectures'].items():
            label = self.get_architecture_label(arch)
            lines.append(f"- **{label}**: {data['metric']:.3f}")
        
        return "\n".join(lines)

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    config = ExperimentConfig(
        name='my_experiment',
        category='core',
        device=args.device,
        output_dir=args.output_dir or Path('results/core/my_experiment')
    )
    
    if args.quick:
        config.custom_settings['num_samples'] = 10
    
    experiment = MyExperiment()
    result = experiment.execute(config)
    
    return 0 if result.success else 1

if __name__ == '__main__':
    exit(main())
```

### 3. Register in Runner

Add to `experiments/run_refactored.py`:

```python
from experiments.core.my_experiment.experiment import MyExperiment

EXPERIMENT_REGISTRY = {
    # ... existing experiments
    'my_experiment': MyExperiment,
}
```

### 4. Test

```bash
# Test import
python -c "from experiments.core.my_experiment.experiment import MyExperiment; print('OK')"

# Run experiment
python experiments/run_experiments_unified.py --exp my_experiment

# Run with config
python experiments/run_experiments_unified.py --exp my_experiment --config configs/experiments/my_experiment.yaml
```

## Common Patterns

### Accessing Architecture Colors

```python
# Old
colors = {'prenorm': '#e74c3c', 'postnorm': '#3498db', ...}

# New
from experiments.common.visualization import ARCHITECTURE_COLORS
color = ARCHITECTURE_COLORS['prenorm']  # or self.get_architecture_color('prenorm')
```

### Creating Models

```python
# Old
model = SimpleTransformer(vocab_size=32000, d_model=4096, ...)

# New
model = self.create_model('prenorm', num_layers=96, d_model=4096)
```

### Validating Targets

```python
# Old
def validate(actual, target):
    return abs(actual - target) / target < 0.15

# New (with ValidationMixin)
validation = self.validate_target(
    actual=value,
    target=target_value,
    tolerance=0.15,
    name='metric_name'
)
if validation['passed']: ...
```

### Generating Reports

```python
# Old
with open('report.md', 'w') as f:
    f.write("# Title\n")
    f.write("## Section\n")
    f.write(f"Result: {value}\n")

# New (override method)
def generate_report(self, result):
    return f"""# Title
## Section
Result: {result.metrics['value']}
"""
```

## Migrating Training Scripts

### Before
```python
# Duplicated across train_model.py, train_h20.py, train_streaming.py

def save_checkpoint(...):
    torch.save(checkpoint, path)

def compute_loss(...):
    outputs = model(input_ids)
    return criterion(outputs, labels)

# Setup distributed (duplicate in 2 files)
if 'RANK' in os.environ:
    dist.init_process_group(backend='nccl')
```

### After
```python
# Use scripts/common/
from scripts.training.common import (
    CheckpointManager,      # Safe checkpoint save/load
    compute_loss,           # Shared loss computation
    train_step,             # Shared training step
    setup_distributed,      # Distributed setup
    cleanup_distributed,    # Cleanup
    is_main_process,        # Rank check
)

# In training loop
checkpoint_manager = CheckpointManager(checkpoint_dir)
checkpoint_manager.save(model, optimizer, epoch, loss)

# Distributed
rank, world_size, local_rank = setup_distributed()
if is_main_process(rank):
    print("Only print on main process")
```

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'experiments'`

**Solution:** Add project root to path at start of file:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
```

### Matplotlib Not Found

**Problem:** Visualization functions fail without matplotlib

**Solution:** Functions have built-in fallback:
```python
# Will print warning and skip if matplotlib not installed
plot_architecture_comparison(...)
```

### Config Not Found

**Problem:** `config.yaml` not found

**Solution:** Provide default config in `setup()`:
```python
def setup(self, config):
    super().setup(config)
    if self.config_path and self.config_path.exists():
        with open(self.config_path, 'r') as f:
            self.yaml_config = yaml.safe_load(f)
    else:
        # Default config
        self.yaml_config = {...}
```

## Checklist

When migrating an experiment:

- [ ] Create `config.yaml` with experiment settings
- [ ] Create `experiment.py` extending `CoreExperiment`
- [ ] Move hardcoded colors to use `ARCHITECTURE_COLORS`
- [ ] Replace inline model creation with `self.create_model()`
- [ ] Add `run()` method returning `ExperimentResult`
- [ ] Add `visualize()` method using standard plotting functions
- [ ] Add `generate_report()` method (optional, base class has default)
- [ ] Add `main()` function for CLI entry point
- [ ] Register in `experiments/run_experiments_unified.py`
- [ ] Test import: `python -c "from ... import MyExperiment"`
- [ ] Test run: `python experiments/run_experiments_unified.py --exp my_experiment`
- [ ] Update any documentation references

## Resources

- `experiments/core/exp1_representation_burial/experiment.py` - Full example
- `experiments/core/base_core_experiment.py` - Base class documentation
- `experiments/common/visualization.py` - Plotting utilities
- `configs/experiments/` - Example YAML configs
