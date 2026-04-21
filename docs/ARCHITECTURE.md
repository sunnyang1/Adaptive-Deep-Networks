# Architecture Documentation

> **Note:** This document contains Mermaid diagrams that render automatically on GitHub.
> If viewing offline, use a Mermaid-compatible viewer or see the [text descriptions](#directory-structure) below.

## System Overview

Adaptive Deep Networks (ADN) is a modular transformer architecture designed for efficient long-context inference through three key innovations:

| Component | Purpose | Key Benefit |
|-----------|---------|-------------|
| **Attention Residuals (AttnRes)** | Prevents representation burial | O(Nd) memory vs O(Ld) |
| **Dynamic Gating with qTTT** | Adaptive computation allocation | 40% efficiency gain |
| **TurboQuant** | Model compression | 6x compression, 0% accuracy loss |

## Quick Navigation

- [High-Level Architecture](#high-level-architecture) - System diagram
- [Component Interactions](#component-interactions) - Data flow sequence
- [Module Dependencies](#module-dependencies) - Import relationships
- [AttnRes Flow](#attention-residuals-attnres-flow) - Block attention mechanism
- [qTTT Flow](#qttt-adaptation-flow) - Query adaptation process
- [TurboQuant Pipeline](#turboquant-compression-pipeline) - Compression stages
- [Directory Structure](#directory-structure) - File organization

## High-Level Architecture

```mermaid
flowchart TB
    Input([Input Tokens]) --> Embed[Embedding Layer]
    Embed --> Layers[Adaptive Transformer Layers]
    Layers --> Head[LM Head]
    Head --> Output([Output Logits])
    
    subgraph "Each Layer"
        direction TB
        AR[AttnRes] --> Attn[Attention]
        Attn --> Gate{Gating}
        Gate -->|High Loss| TTT[qTTT Adapt]
        Gate -->|Low Loss| Skip[Skip Adapt]
        TTT --> MLP
        Skip --> MLP[Feed Forward]
    end
    
    Layers -.-> TQ[TurboQuant]
    TQ --> PQ[PolarQuant]
    TQ --> QJL[QJL Transform]
```

## Component Interactions

```mermaid
sequenceDiagram
    autonumber
    participant I as Input
    participant A as AttnRes
    participant G as Gating
    participant Q as qTTT
    participant O as Output

    I->>A: Hidden States
    A->>A: Block Aggregation
    A->>G: Augmented Hidden
    
    G->>G: Compute Loss
    
    alt Loss > Threshold
        G->>Q: Trigger Adapt
        Q->>Q: N adaptation steps
        Q-->>O: Adapted Query
    else Loss ≤ Threshold
        G-->>O: Pass Through
    end
```

## Module Dependencies

```mermaid
flowchart TD
    subgraph Core["Core Modules (src/)"]
        direction TB
        M[models] --> Attn[attnres]
        M --> Q[qttt]
        M --> G[gating]
        M --> T[turboquant]
    end
    
    subgraph Exp["Experiments"]
        direction TB
        C[common] --> CR[core]
        C --> V[validation]
        C --> R[runner]
    end
    
    subgraph Scr["Scripts"]
        SC[common] --> TR[training]
    end
    
    Attn -.->|uses| Exp
    Q -.->|uses| Exp
    M -.->|uses| Scr
```

## Attention Residuals (AttnRes) Flow

```mermaid
flowchart LR
    subgraph Phase1["Phase 1: Inter-Block (Parallel)"]
        direction TB
        B1[Block 1] --> Agg[Aggregator]
        B2[Block 2] --> Agg
        BN[Block N] --> Agg
        BP[Partial] --> Agg
    end
    
    Agg --> PQ{Pseudo-Query}
    PQ --> WS[Weighted Sum]
    
    subgraph Phase2["Phase 2: Intra-Block (Sequential)"]
        direction TB
        WS --> LN[LayerNorm]
        LN --> AM[Attention/MLP]
        AM --> UP[Update Partial]
    end
```

## qTTT Adaptation Flow

```mermaid
sequenceDiagram
    autonumber
    participant Q as Query q
    participant A as Adapter
    participant C as Frozen KV Cache
    participant L as Margin Loss

    Q->>A: Initialize q_adapt
    
    loop N steps
        A->>C: attention(q_adapt, K, V)
        C->>L: compute distribution
        L->>L: L = -logit_margin
        L-->>A: backward()
        A->>A: q_adapt -= lr * grad
    end
    
    A-->>Q: adapted query
```

## TurboQuant Compression Pipeline

```mermaid
flowchart TD
    Input["Input x ∈ ℝᵈ"] --> HT["Hadamard Transform"]
    HT --> Polar["Cartesian → Polar"]
    
    Polar --> Mag["Magnitude r"]
    Polar --> Ang["Angles θ"]
    
    Ang --> LM["Lloyd-Max<br/>3-bit quant"]
    Mag --> QJL["QJL Residual<br/>1-bit sign"]
    
    LM --> Out["Compressed<br/>r: FP16 + θ: 3b + s: 1b"]
    QJL --> Out
    
    style Out fill:#e1f5fe
```

## Data Flow Through System

```mermaid
flowchart TB
    subgraph Train["Training"]
        direction LR
        T1[Tokens] --> T2[Embed] --> T3[Layers] --> T4[Head] --> T5[Loss]
    end
    
    subgraph Inf["Inference"]
        direction TB
        I1[Input] --> I2[KV Cache]
        I2 --> I3{Gating}
        I3 -->|High Loss| I4[qTTT]
        I3 -->|Low Loss| I5[Standard]
        I4 --> I6[Generate]
        I5 --> I6
    end
    
    subgraph Comp["Compression"]
        direction LR
        C1[Weights] --> C2[TurboQuant]
        C2 --> C3[4-bit Weights]
        C2 --> C4[Compressed KV]
    end
    
    style Train fill:#e8f5e9
    style Inf fill:#fff3e0
    style Comp fill:#fce4ec
```

## Directory Structure

```
Adaptive-Deep-Networks/
├── src/                          # Core implementation
│   ├── attnres/                  # Attention Residuals
│   │   ├── block_attnres.py     # Main implementation
│   │   └── pseudo_query.py      # Pseudo-query management
│   ├── qttt/                     # Query-Only TTT
│   │   ├── adaptation.py        # Core adaptation logic
│   │   ├── margin_loss.py       # Margin maximization
│   │   └── polar_adaptation.py  # Polar coordinate variant
│   ├── gating/                   # Dynamic gating
│   │   ├── threshold.py         # Threshold calibration
│   │   ├── reconstruction.py    # Loss computation
│   │   └── depth_priority.py    # Depth-priority policy
│   ├── models/                   # Model definitions
│   │   ├── adaptive_transformer.py
│   │   └── configs.py
│   └── turboquant/               # Compression
│       ├── polar_quant.py       # Polar quantization
│       ├── qjl.py               # QJL transform
│       └── turbo_quant.py       # Pipeline
│
├── experiments/                  # Experiment framework
│   ├── common/                   # Shared utilities
│   │   └── config.py            # YAML config loader
│   ├── core/                     # Core experiments (exp1-6)
│   │   ├── base_experiment.py   # Base class
│   │   └── exp*.py              # Individual experiments
│   ├── runner/                   # Experiment execution
│   └── validation/               # Paper validation
│
├── scripts/                      # Training scripts
│   └── train.py                 # Unified training
│
├── configs/                      # Configuration files
│   └── experiments/             # YAML configs
│
├── tests/                        # Test suite
│   └── unit/                    # Unit tests
│       ├── test_attnres.py
│       ├── test_qttt.py
│       └── test_gating.py
│
└── docs/                         # Documentation
    ├── api/                     # API reference
    │   └── README.md
    └── ARCHITECTURE.md          # This file
```

## Key Design Decisions

### 1. Block-Based Attention
- **Why**: Reduces memory from O(Ld) to O(Nd)
- **Trade-off**: Slight approximation for significant efficiency gain
- **Implementation**: `block_attn_res()` function

### 2. Query-Only Adaptation
- **Why**: Only 0.5% of parameters need updating
- **Benefit**: Fast adaptation without model modification
- **Implementation**: `QueryOnlyTTT` class

### 3. Polar Quantization
- **Why**: Natural separation of magnitude and direction
- **Benefit**: Better preserves relative rankings
- **Implementation**: `PolarQuant` class

### 4. YAML Configuration
- **Why**: Human-readable, version-controllable
- **Benefit**: Easy experiment reproduction
- **Implementation**: `ExperimentConfig` class

## Performance Considerations

| Component | Memory | Compute | Communication |
|-----------|--------|---------|---------------|
| AttnRes | O(Nd) | O(N²d) | O(Nd) |
| qTTT | O(d) | O(N_adapt × d) | O(1) |
| TurboQuant | O(d/6) | O(d) | O(d/6) |

## Extension Points

1. **New Architectures**: Extend `BaseExperiment`
2. **New Gating Policies**: Extend `DynamicThreshold`
3. **New Compression**: Extend `TurboQuantPipeline`
4. **New Adaptation**: Extend `QueryOnlyTTT`

## Troubleshooting

### Mermaid Diagrams Not Rendering

If diagrams don't render on GitHub:

1. **Check GitHub support**: Mermaid requires GitHub's native renderer
2. **Use GitHub Web**: The mobile app may not support Mermaid
3. **Alternative**: View the [API Documentation](./api/README.md) which includes ASCII diagrams

## References

- Chen et al. (2026): "Attention Residuals" Technical Report
- Bansal et al.: "Logit Margins" (for margin requirement)
- Adaptive Deep Networks Paper (Appendix A)
