# API Documentation

This directory contains API documentation for the Adaptive Deep Networks project.

## Module Structure

```
src/
├── attnres/          # Attention Residuals implementation
├── gating/           # Dynamic gating for adaptation
├── models/           # Model configurations and implementations
├── qttt/             # Query-Only Test-Time Training
└── turboquant/       # TurboQuant compression

experiments/
├── common/           # Shared experiment utilities
├── core/             # Core paper experiments
├── runner/           # Experiment execution framework
└── validation/       # Paper validation scripts
```

## Core Modules

### src.attnres

Block Attention Residuals implementation for preventing representation burial.

**Key Classes:**
- `RMSNorm` - Root Mean Square Layer Normalization
- `BlockAttnRes` - Block Attention Residuals layer
- `TwoPhaseBlockAttnRes` - Two-phase computation strategy

**Example:**
```python
from src.attnres.block_attnres import BlockAttnRes

layer = BlockAttnRes(dim=512, num_blocks=8)
h_attn, h_mlp = layer(blocks, hidden, use_attn=True, use_mlp=True)
```

### src.qttt

Query-Only Test-Time Training for dynamic adaptation.

**Key Classes:**
- `QueryOnlyTTT` - Main adaptation mechanism
- `KVCache` - Frozen KV cache for efficient inference
- `MarginMaximizationLoss` - Logit margin maximization

**Example:**
```python
from src.qttt.adaptation import QueryOnlyTTT

ttt = QueryOnlyTTT(dim=512, learning_rate=0.01)
adapted_query = ttt.adapt(query)
```

### src.gating

Dynamic gating controller for deciding when to adapt.

**Key Classes:**
- `DynamicThreshold` - Base threshold calibration
- `EMAThreshold` - EMA-based threshold
- `TargetRateThreshold` - Target-rate-based threshold

**Example:**
```python
from src.gating.threshold import EMAThreshold

gating = EMAThreshold(initial_threshold=2.0, beta=0.99)
should_adapt = gating.should_adapt(reconstruction_loss)
```

### src.turboquant

TurboQuant compression for 6x model compression.

**Key Classes:**
- `PolarQuant` - Polar coordinate quantization
- `QJLCompressor` - Quantized JL transform
- `TurboQuantPipeline` - Full compression pipeline

**Example:**
```python
from src.turboquant import TurboQuantPipeline, TurboQuantConfig

config = TurboQuantConfig(angle_bits=3, qjl_proj_dim=256)
pipeline = TurboQuantPipeline(dim=512, config=config)
r, theta, signs = pipeline.compress_vector(x)
```

## Experiment Framework

### experiments.common

Shared utilities for all experiments.

**Key Classes:**
- `ExperimentConfig` - Configuration management
- `OutputPaths` - Standardized output paths
- `DeviceManager` - Device and memory management
- `FigureManager` - Visualization utilities

**Example:**
```python
from experiments.common import ExperimentConfig, OutputPaths

config = ExperimentConfig(
    name='my_experiment',
    category='core',
    model_size='medium'
)
paths = OutputPaths.for_experiment('my_experiment', 'core')
```

### experiments.runner

Experiment execution framework.

**Key Classes:**
- `BaseExperiment` - Abstract base for all experiments
- `ExperimentRunner` - Orchestrates experiment execution
- `ExperimentRegistry` - Plugin-style experiment discovery

**Example:**
```python
from experiments.runner import BaseExperiment, ExperimentResult

class MyExperiment(BaseExperiment):
    def run(self, config) -> ExperimentResult:
        # Implement experiment logic
        return ExperimentResult(name=self.name, success=True)
```

## Usage Examples

See individual module documentation for detailed usage examples.

## Type Hints

All public APIs use Python type hints for better IDE support and documentation.

```python
def block_attn_res(
    blocks: List[torch.Tensor],
    partial_block: torch.Tensor,
    pseudo_query: torch.Tensor,
    norm: RMSNorm,
    eps: float = 1e-6
) -> torch.Tensor:
    ...
```

## Configuration

Most modules support YAML-based configuration:

```yaml
# config.yaml
model:
  hidden_dim: 4096
  num_layers: 32
  num_heads: 32

experiment:
  num_samples: 100
  device: cuda
```
