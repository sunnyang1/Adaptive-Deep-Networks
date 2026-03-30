# Architecture Documentation

## System Overview

Adaptive Deep Networks (ADN) is a modular transformer architecture designed for efficient long-context inference through three key innovations:

1. **Attention Residuals (AttnRes)** - Prevents representation burial
2. **Dynamic Gating with qTTT** - Adaptive computation allocation
3. **TurboQuant** - 6x model compression

## High-Level Architecture

```mermaid
graph TB
    subgraph "Adaptive Deep Networks"
        A[Input] --> B[Embedding]
        B --> C[Adaptive Layers]
        C --> D[Output Head]
        D --> E[Logits]
        
        subgraph "Per Layer"
            C1[AttnRes Block] --> C2[Adaptive Attention]
            C2 --> C3[qTTT Adaptation]
            C3 --> C4[Adaptive MLP]
        end
        
        F[Gating Controller] --> C3
        G[KV Cache] --> C2
    end
    
    subgraph "Compression"
        H[TurboQuant] --> H1[PolarQuant]
        H --> H2[QJL]
    end
    
    C -.-> H
```

## Component Interactions

```mermaid
sequenceDiagram
    participant Input as Input
    participant AttnRes as AttnRes
    participant Attention as Adaptive Attention
    participant Gating as Gating Controller
    participant qTTT as qTTT
    participant MLP as Adaptive MLP
    participant Output as Output

    Input->>AttnRes: Hidden States
    AttnRes->>AttnRes: Aggregate Block Representations
    AttnRes->>Attention: AttnRes-Augmented Hidden
    
    Attention->>Gating: Compute Reconstruction Loss
    Gating->>Gating: Should Adapt?
    
    alt Loss > Threshold
        Gating->>qTTT: Enable Adaptation
        qTTT->>qTTT: Adapt Query (N steps)
        qTTT->>Attention: Adapted Query
    else Loss ≤ Threshold
        Gating->>Attention: Use Original Query
    end
    
    Attention->>MLP: Attention Output
    MLP->>Output: Final Hidden
```

## Module Dependencies

```mermaid
graph LR
    subgraph "Core Modules"
        A[attnres] --> B[models]
        C[qttt] --> B
        D[gating] --> C
        E[turboquant] --> B
    end
    
    subgraph "Experiments"
        F[common] --> G[core]
        F --> H[validation]
        F --> I[runner]
    end
    
    subgraph "Scripts"
        J[common] --> K[training]
    end
    
    B -.-> F
    B -.-> J
```

## Attention Residuals (AttnRes) Flow

```mermaid
graph LR
    subgraph "Inter-Block Phase"
        A[Block 1] --> Z[Aggregator]
        B[Block 2] --> Z
        C[Block N] --> Z
        P[Partial Block] --> Z
    end
    
    Z --> D{Pseudo-Query Attention}
    D --> E[Weighted Sum]
    
    subgraph "Intra-Block Phase"
        E --> F[Layer Norm]
        F --> G[Attention/MLP]
        G --> H[Update Partial Block]
    end
```

## qTTT Adaptation Flow

```mermaid
sequenceDiagram
    participant Query as Query q
    participant Cache as Frozen KV Cache
    participant Adapter as Query Adapter
    participant Optimizer as SGD Optimizer
    participant MarginLoss as Margin Loss

    Query->>Adapter: Initialize q_adapt
    
    loop N adaptation steps
        Adapter->>Cache: Attention(q_adapt, K, V)
        Cache->>MarginLoss: Compute attention distribution
        MarginLoss->>MarginLoss: L_margin = -logit_margin
        MarginLoss->>Optimizer: Backward(L_margin)
        Optimizer->>Adapter: q_adapt -= lr * grad
    end
    
    Adapter->>Query: Return adapted query
```

## TurboQuant Compression Pipeline

```mermaid
graph TB
    A[Input Vector x] --> B[Random Hadamard Transform]
    B --> C[Cartesian to Polar]
    C --> D[Magnitude r]
    C --> E[Angles θ]
    
    E --> F[Lloyd-Max Quantization]
    F --> G[Quantized θ indices]
    
    D --> H[QJL Residual]
    H --> I[Compute residual e]
    I --> J[Project Se]
    J --> K[Sign(Se)]
    
    L[Compressed] --> M[r: FP16]
    L --> N[θ: 3-bit]
    L --> O[sign: 1-bit]
```

## Data Flow Through System

```mermaid
graph TB
    subgraph "Training Phase"
        A[Input Tokens] --> B[Embed + Pos Encode]
        B --> C[Layer 1]
        C --> D[Layer 2]
        D --> E[...]
        E --> F[Layer L]
        F --> G[LM Head]
        G --> H[Loss]
    end
    
    subgraph "Inference Phase"
        I[Input] --> J[Cache KV]
        J --> K{Gating Check}
        
        K -->|High Loss| L[qTTT Adapt]
        K -->|Low Loss| M[Standard Forward]
        
        L --> N[Generate]
        M --> N
    end
    
    subgraph "Compression Phase"
        O[Model Weights] --> P[TurboQuant]
        P --> Q[4-bit Weights]
        P --> R[Compressed KV Cache]
    end
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
│   ├── core/                     # Core experiments (exp1-6)
│   ├── validation/               # Paper validation
│   └── real_model/              # Real model validation
│
├── scripts/                      # Training scripts
│   ├── common/                   # Shared training code
│   └── train_refactored.py      # Unified training
│
├── configs/                      # Configuration files
│   └── experiments/
│
├── tests/                        # Test suite
│   └── unit/
│
└── docs/                         # Documentation
    ├── api/                      # API docs
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

## References

- Chen et al. (2026): "Attention Residuals" Technical Report
- Bansal et al.: "Logit Margins" (for margin requirement)
- Adaptive Deep Networks Paper (Appendix A)
