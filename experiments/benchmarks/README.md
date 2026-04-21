# 基准测试

本目录包含模型基准测试代码。

## 测试列表

### Needle-in-Haystack (长上下文检索)
**目标**: 验证模型在极端长上下文中的信息检索能力

**配置**:
- 上下文长度: 1K, 4K, 16K, 32K, 64K, 128K, 256K
- 测试深度: 10个位置均匀分布
- 指标: 精确匹配准确率

**预期结果** (AttnRes + qTTT, 8.7B):
| 长度 | 准确率 |
|------|--------|
| 1K | 99.5% |
| 16K | 94.1% |
| 64K | 82.5% |
| 128K | 75.8% |
| 256K | 68.2% |
| **平均** | **86.9%** |

**运行**:
```bash
python benchmarks/run_needle.py --model-size medium --context-lengths 4096 16384 65536 131072
```

### MATH (数学推理)
**目标**: 验证模型数学推理能力

**配置**:
- 难度级别: 1-5
- 指标: 精确匹配准确率

**预期结果**:
- 8.7B 模型: 52.3% (匹配 50B 静态基线)
- 2.2B 模型: 56.1% (超过 8.7B 目标)

**运行**:
```bash
python benchmarks/run_math.py --model-size medium
```

### LongBench-v2 (综合评估)
**目标**: 综合长上下文理解能力评估

**类别**:
- Single-Doc QA
- Multi-Doc QA
- Summarization
- Few-shot Learning
- Synthetic Tasks
- Code Completion

**预期结果**: 平均 56.8%

**运行**:
```bash
python benchmarks/run_longbench.py --model-size medium
```

## 快速运行所有基准

```bash
python benchmarks/run_all.py --model-size small
```

## 结果位置

基准测试结果保存到:
- `results/benchmarks/needle/`
- `results/benchmarks/math/`
- `results/benchmarks/longbench/`
