# TurboQuant 实验

本目录包含 TurboQuant 压缩和加速相关的实验。

## 实验列表

### TQ-1: PolarQuant 压缩验证
**目标**: 验证 6x+ 内存缩减，零精度损失

**指标**:
- 压缩比 (目标: ≥6x)
- 重构误差 (目标: 余弦相似度 >0.99)
- 量化后模型精度

**运行**:
```bash
python turboquant/test_polar_quant.py
```

### TQ-2: Polar qTTT 效率验证
**目标**: 验证 50% 参数减少，更快收敛

**指标**:
- 可训练参数减少 50%
- 收敛速度提升
- 最终准确率对比

**运行**:
```bash
python turboquant/test_polar_qttt.py
```

### TQ-3: 深度优先策略验证
**目标**: 验证 2.4x 吞吐提升

**指标**:
- 吞吐: 110 vs 45 tokens/s
- 延迟: 500ms vs 850ms p99
- KV cache: 2.8GB vs 16GB

**运行**:
```bash
python turboquant/test_depth_priority.py
```

### TQ-4: 端到端集成测试
**目标**: 综合性能验证

**运行**:
```bash
python turboquant/test_integration.py
```

## 详细步骤

参见: [../docs/turboquant_experiments.md](../docs/turboquant_experiments.md)
