# Small Model Experiment Report

## 概述

本报告记录了 Adaptive Deep Networks Small Model (2.2B 参数) 的构建过程和实验结果。

**实验时间**: 2026-03-30  
**硬件环境**: Apple Silicon (MPS)  
**PyTorch版本**: 2.2.2

---

## 1. 模型规格

### 1.1 架构配置

| 参数 | 值 |
|------|-----|
| 总参数量 | 2.21B |
| 层数 | 32 |
| 隐藏维度 | 2048 |
| 注意力头数 | 32 |
| MLP比例 | 4 |
| 词汇表大小 | 32,000 |
| 最大序列长度 | 32,768 |
| AttnRes块数 | 8 |
| qTTT最大步数 | 16 |
| qTTT跨度长度 | 128 |

### 1.2 参数分布

```
Component              Parameters    Percentage
-----------------      ----------    ----------
Transformer Layers     2.15B         97.03%
Token Embedding        65.5M         2.96%
AttnRes Modules        0.3M          0.01%
RMS Norm               2K            0.00%
-----------------      ----------    ----------
Total                  2.21B         100%
```

**关键发现**: AttnRes 只增加了 0.26M 参数 (0.012%)，但实现了从 O(Ld) 到 O(Nd) 的内存复杂度降低，其中 N=8 是块数。

---

## 2. 构建指标

### 2.1 构建性能

| 指标 | 值 |
|------|-----|
| 构建时间 | 34.16 秒 |
| 模型内存占用 | 4.90 GB |
| 模型数据类型 | float32 |

### 2.2 内存使用

- **构建前内存**: 204.6 MB
- **构建后内存**: 5,223.8 MB
- **模型本身**: 4,900 MB (4.9 GB)

---

## 3. 推理性能实验

### 3.1 延迟 vs 序列长度

| 序列长度 | 平均延迟 | 标准差 | 吞吐量 |
|----------|----------|--------|--------|
| 64 tokens | 185.5 ms | 0.83 ms | 345.0 tok/s |
| 128 tokens | 334.5 ms | 1.48 ms | 382.7 tok/s |
| 256 tokens | 665.6 ms | 0.82 ms | 384.6 tok/s |
| 512 tokens | 1357.1 ms | 0.65 ms | 377.3 tok/s |

**观察**:
- 吞吐量在 128-512 tokens 范围内保持稳定 (~380 tok/s)
- 延迟与序列长度呈近似线性关系
- 标准差很小，表明性能稳定

### 3.2 延迟/Token 分析

| 序列长度 | 每Token延迟 |
|----------|-------------|
| 64 tokens | 2.90 ms/token |
| 128 tokens | 2.61 ms/token |
| 256 tokens | 2.60 ms/token |
| 512 tokens | 2.65 ms/token |

**结论**: 每token延迟在 2.6-2.9ms 之间，随序列长度增加略有下降，表明注意力计算的效率在长序列上略有提升。

---

## 4. FLOP 分析

### 4.1 每层 FLOPs

| 组件 | FLOPs | 占比 |
|------|-------|------|
| Attention QKV | 25.2 MFLOPs | 18.7% |
| Attention 计算 | 4.1 KFLOPs | ~0% |
| Attention Output | 8.4 MFLOPs | 6.2% |
| MLP (SwiGLU) | 100.7 MFLOPs | 75.0% |
| **总计每层** | **134.2 MFLOPs** | **100%** |

### 4.2 总 FLOPs

```
总 FLOPs = 32 layers × 134.2 MFLOPs/layer = 4.30 GFLOPs/token
```

**与论文对比**: 论文中 7B 模型的 FLOPs 约为 14 GFLOPs/token，Small 模型 (2.2B) 的比例约为 0.31，与参数量比例 (2.2/7 ≈ 0.31) 相符。

---

## 5. AttnRes 组件分析

### 5.1 内存复杂度对比

| 方法 | 内存复杂度 | 说明 |
|------|-----------|------|
| 标准 Attention | O(L × d) | L=32 layers, d=2048 |
| AttnRes | O(N × d) | N=8 blocks |
| **理论节省** | **75%** | 从 32×d 降到 8×d |

### 5.2 实现细节

- **块数**: 8
- **每层块数**: 32 / 8 = 4 layers/block
- **伪查询维度**: 与 hidden_dim (2048) 相同
- **两阶段计算**: Phase 1 (块间并行) + Phase 2 (块内顺序)

---

## 6. qTTT 配置分析

| 参数 | 值 | 说明 |
|------|-----|------|
| 最大步数 | 16 | 测试时训练的最大迭代次数 |
| 跨度长度 | 128 | 每次更新的token数量 |
| 学习率 | 0.005 | 查询参数的更新率 |

**特点**:
- 仅更新查询参数 (Query-only)
- KV Cache 在预填充后冻结
- 使用边际最大化损失

---

## 7. 实验结论

### 7.1 主要成果

1. **成功构建 Small Model**: 2.21B 参数，符合论文规格
2. **验证了 AttnRes 的低开销**: 仅增加 0.012% 参数
3. **量化了推理性能**: 380 tok/s 吞吐量 (MPS)
4. **确认了 FLOP 估算**: 4.30 GFLOPs/token

### 7.2 性能特征

- **吞吐量**: ~380 tokens/second (MPS设备)
- **内存占用**: ~5GB (float32)
- **延迟稳定性**: 标准差 < 2ms
- **扩展性**: 线性扩展至 512 tokens

### 7.3 与论文对比

| 指标 | 论文 (7B) | 本实验 (2.2B) | 比例 |
|------|-----------|---------------|------|
| 参数量 | 7B | 2.2B | 0.31x |
| FLOPs/token | ~14 GFLOPs | 4.3 GFLOPs | 0.31x |
| 层数 | 32 | 32 | 1x |
| 隐藏维度 | 4096 | 2048 | 0.5x |

比例关系符合理论预期。

---

## 8. 后续工作建议

1. **量化优化**: 使用 bfloat16/int8 减少内存占用
2. **批处理测试**: 测试更大的 batch size
3. **长序列测试**: 测试 1K-32K 序列长度
4. **qTTT 验证**: 实现并测试 qTTT 适应机制
5. **Gating 机制**: 测试动态计算门控

---

## 附录: 实验文件

- **结果 JSON**: `results/small_model_experiments.json`
- **构建脚本**: `scripts/model/build_and_benchmark_small.py`
- **实验脚本**: `scripts/experiments/run_small_experiments.py`

---

*Report generated on 2026-03-30*
