# Architecture Optimization Based on Attention Residuals Paper

## Summary of Paper Findings (§5.3 & §5.4)

### §5.3 Ablation Study - Key Design Decisions

| Design Choice | Paper Finding | Our Implementation |
|--------------|---------------|-------------------|
| **Block Size (S)** | N=8 blocks recovers most FullAttnRes benefit; larger blocks (S=16,32) approach baseline | ✅ `num_blocks=8` (Small/Medium), `16` (Large) |
| **Pseudo-query** | Input-dependent (1.731) better but needs d×d projection; learned static preferred | ✅ Learned input-independent, **zero-init** |
| **RMSNorm on keys** | Critical! Without: Full +0.006, Block +0.004 loss degradation | ✅ RMSNorm applied to keys |
| **Multi-head depth** | H=16 hurts performance (1.752 vs 1.746) | ✅ Single-head depth attention |
| **Softmax vs Sigmoid** | Softmax (1.737) > Sigmoid (1.741) - competitive normalization helps | ✅ Softmax normalization |
| **Sliding Window** | SWA (W=8) = 1.764 vs Full 1.737 - distant access matters | ✅ Full cross-block attention |

### §5.4.1 Optimal Architecture - Critical Discovery

Paper conducted 5×5 grid sweep under fixed compute:

| Metric | Baseline Optimal | AttnRes Optimal | Implication |
|--------|-----------------|-----------------|-------------|
| `d_model/L_b` | ~60 | ~45 | AttnRes favors **deeper, narrower** networks |
| `H/L_b` | ~0.3 | ~0.3 | Head ratio constant across methods |

**Key Insight**: AttnRes shifts optimal depth-width trade-off toward ~25% deeper networks.

## Final Optimized Architectures

### Summary Table

| Size | Params | Layers | Hidden | Heads | Blocks | d_model/L_b | H/L_b |
|------|--------|--------|--------|-------|--------|-------------|-------|
| **Small** | 3.3B | 48 | 2048 | 16 | 8 | 42.7 ✅ | 0.33 ✅ |
| **Medium** | 6.6B | 56 | 2688 | 16 | 8 | 48.0 ✅ | 0.29 ✅ |
| **Large** | **27.5B** | **96** | **4224** | **32** | 16 | **44.0 ✅** | **0.33 ✅** |

### Detailed Configurations

#### Small Model (3.3B params)
- **Layers**: 48 (increased from 32 for AttnRes benefit)
- **Hidden dim**: 2048
- **Attention heads**: 16
- **AttnRes blocks**: 8 (S=6 layers per block)
- **Optimal ratios**: d_model/L_b = 42.7, H/L_b = 0.33

#### Medium Model (6.6B params)
- **Layers**: 56 (deeper for AttnRes benefit)
- **Hidden dim**: 2688
- **Attention heads**: 16
- **AttnRes blocks**: 8 (S=7 layers per block)
- **Optimal ratios**: d_model/L_b = 48.0, H/L_b = 0.29

#### Large Model (27.5B params) ⭐ Fully Optimized
- **Layers**: 96 (deeper for AttnRes benefit)
- **Hidden dim**: 4224
- **Attention heads**: 32
- **AttnRes blocks**: 16 (S=6 layers per block)
- **Optimal ratios**: d_model/L_b = **44.0** ✅, H/L_b = **0.33** ✅
- **Selection**: Grid search optimization for paper-optimal ratios at ~27B scale

## Grid Search Results for Large Model

Top configurations considered:

| Layers | Hidden | Heads | d/L_b | H/L_b | Params | Score |
|--------|--------|-------|-------|-------|--------|-------|
| 96 | 4352 | 32 | 45.3 | 0.333 | 29.2B | 1.33 |
| **96** | **4224** | **32** | **44.0** | **0.333** | **27.5B** | **2.00** ⭐ |
| 96 | 4480 | 28 | 46.7 | 0.292 | 31.0B | 1.92 |
| 88 | 3968 | 32 | 45.1 | 0.364 | 22.3B | 2.00 |

Selected **96/4224/32** for optimal balance of:
- Parameter count (~27.5B target)
- d_model/L_b ≈ 44 (paper optimal ~45)
- H/L_b ≈ 0.33 (paper optimal ~0.3)

## Why These Changes Matter

### 1. Depth-Width Trade-off (d_model/L_b ≈ 45)
- **Paper evidence**: AttnRes optimal at d_model/L_b ≈ 45 vs baseline ~60
- **Mechanism**: Selective aggregation across depth allows better utilization of deeper networks
- **Benefit**: Better compositional task performance (GPQA +7.5, Math +3.6, HumanEval +3.1)

### 2. Head Ratio (H/L_b ≈ 0.3)
- **Paper evidence**: Both methods optimal at H/L_b ≈ 0.3
- **Mechanism**: Balances attention capacity with computational efficiency
- **Benefit**: Optimal parallelization of attention computation

### 3. Block Count (N ≈ 8-16)
- **Paper evidence**: N=8 recovers most FullAttnRes benefit (Fig 6)
- **Mechanism**: Sufficient granularity for selective aggregation
- **Benefit**: O(Nd) memory vs O(Ld), minimal performance loss

## Training Dynamics Improvements

Per paper §5.2, AttnRes with optimized architecture shows:

1. **Bounded Output Magnitudes**: Prevents PreNorm dilution (O(L) growth → periodic bounded)
2. **Uniform Gradient Distribution**: Learned softmax weights regulate gradient flow
3. **Better Validation Loss**: Consistently lower throughout training, wider gap during decay

## Files Modified

1. `src/models/configs.py` - Updated all three model configurations
2. `experiments/common/config.py` - MODEL_SIZES dict
3. `scripts/model/calculate_params.py` - Updated calculations
4. `scripts/model/run_small_model_experiments.py` - **New experiment script**
5. `AGENTS.md` - Updated architecture documentation
6. `README.md` - Model configuration table
7. `ARCHITECTURE_OPTIMIZATION.md` - This document

## Running Experiments

### Quick Test (Experimental 150M Model)
```bash
python scripts/model/run_small_model_experiments.py --experimental
```

### Full Small Model (1.1B Params)
```bash
python scripts/model/run_small_model_experiments.py --full
```

## Experiment Results (Experimental Small)

| Metric | Value |
|--------|-------|
| Parameters | 149.6M |
| Build Time | ~2s |
| AttnRes Overhead | 0.03% |
| Throughput (256 tokens) | ~162 tok/sec |
| Architecture Ratios | d_model/L_b=44.0, H/L_b=0.25 |

All verifications passed:
- ✅ AttnRes zero-initialization
- ✅ Correct output shapes
- ✅ Proper block structure

## Reference

Chen et al. "Attention Residuals" Technical Report, 2026
- §5.3: Ablation Study (Table 4, Figure 6)
- §5.4.1: Optimal Architecture (Figure 7)
- §5.2: Training Dynamics (Figure 5)
