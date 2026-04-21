# 核心实验详细说明

本文档详细说明核心验证实验（exp1-exp6）的设计、执行和预期结果。

## 实验列表

| 实验 | 名称 | 目标 | 预计时间 |
|------|------|------|----------|
| [实验1](#实验1) | Representation Burial测量 | 验证PreNorm信号衰减，对比AttnRes改善 | 1-2小时 |
| [实验2](#实验2) | Logit Margin分析 | 验证对数margin要求，展示qTTT实现 | 2-3小时 |
| [实验3](#实验3) | 梯度流测量 | 验证AttnRes改善梯度流均匀性 | 1-2小时 |
| [实验4](#实验4) | FLOP等价验证 | 验证 T_think ≈ 2*N_qttt*k | 2-3小时 |
| [实验5](#实验5) | 组件协同效应 | 验证AttnRes、qTTT、Gating协同 | 2-3小时 |
| [实验6](#实验6) | 辅助验证 | 初始化、块大小、超参数敏感性 | 2-3小时 |

---

## 实验1: Representation Burial定量测量

**目标**: 验证PreNorm配置下早期层信号随深度衰减的现象

### 关键指标
- 相对梯度幅度
- 信号衰减率: $(C_1 - C_L) / C_1$
- 有效深度: 贡献度降到50%时的层数

### 预期结果
| 架构 | 96层衰减率 | 有效深度 |
|------|-----------|---------|
| PreNorm | ~90% | ~24层 |
| PostNorm | ~70% | ~48层 |
| DeepNorm | ~60% | ~64层 |
| AttnRes | ~30% | >80层 |

### 运行命令
```bash
python core/exp1_representation_burial/run_exp1.py --num_samples 50
```

### 输出文件
- `results/core/exp1_results.json`: 原始数据
- `results/core/exp1_representation_burial.png`: 贡献度曲线
- `results/core/exp1_metrics_comparison.png`: 指标对比
- `results/core/exp1_report.md`: 详细报告

---

## 实验2: Logit Margin与上下文长度关系

**目标**: 验证Bansal et al. [4]的对数margin要求

### 关键发现
- Vanilla模型的margin随长度**下降**（违反理论要求）
- qTTT的margin随长度**增长**，满足对数要求

### 可视化
- Margin vs 上下文长度（对数坐标）
- 成功/失败样本的margin分布

### 运行命令
```bash
python core/exp2_margin_analysis/run_exp2.py --context_lengths 1024 4096 16384
```

---

## 实验3: 梯度流改善定量测量

**目标**: 验证AttnRes改善梯度流均匀性

### 关键指标
- 变异系数 (CV): $\sigma / \mu$
- 早期/晚期梯度比
- 梯度流评分

### 预期结果
| 架构 | CV (收敛后) | 早期/晚期梯度比 |
|------|------------|----------------|
| PreNorm | 2.34 | 0.004 |
| PostNorm | 1.89 | 0.048 |
| DeepNorm | 1.56 | 0.080 |
| AttnRes | 0.87 | 0.58 |

### 运行命令
```bash
python core/exp3_gradient_flow/run_exp3.py --num_steps 1000
```

---

## 实验4: FLOP等价公式实证验证

**目标**: 验证 $T_{\text{think}} \approx 2 N_{\text{qTTT}} k$

### 测试策略
1. Pure Width: 全部用于thinking tokens
2. Pure Depth: 全部用于qTTT steps
3. Balanced: 各50%
4. Depth-Priority: 80% depth, 20% width

### 预期结果
Depth-Priority策略在准确率和效率上表现最佳。

### 运行命令
```bash
python core/exp4_flop_equivalence/run_exp4.py --total_flops 5e14
```

---

## 实验5: 组件协同效应定量分析

**目标**: 验证AttnRes、qTTT、Gating的协同效应

### 设计
2³ 因子设计（8种配置）

### 协同效应计算
```
synergy_gain = actual_result - additive_prediction
synergy_coefficient = actual_result / additive_prediction
```

### 关键发现
- Gating带来显著的超加性效应
- 完整系统 > 各组件单独效果之和

### 运行命令
```bash
python core/exp5_synergy/run_exp5.py
```

---

## 实验6: 辅助验证实验

包含三个子实验:

### 6.1 伪查询初始化效果
- 比较零初始化 vs 随机初始化
- 测量训练稳定性和收敛速度

### 6.2 块大小(N)的影响
- 测试 N ∈ {4, 8, 16, 32}
- 寻找准确率和内存占用的最佳平衡点

### 6.3 qTTT超参数敏感性
- 扫描 N_qttt ∈ {4, 8, 16, 32, 64}
- 扫描 k ∈ {64, 128, 256, 512}
- 生成准确率热力图

### 运行命令
```bash
python core/exp6_auxiliary/run_exp6.py
```

---

## 工具函数使用

`utils/measurement.py` 提供以下测量功能:

```python
# Representation Burial测量
measure_representation_burial(model, input_ids)

# Attention Margin测量
measure_attention_margin(model, input_ids, query_pos, target_pos)
analyze_margin_distribution(model, test_samples)

# 梯度流测量
measure_gradient_statistics(model, batch)

# FLOP测量
measure_actual_flops(model, input_ids, config)
compute_flop_equivalent_config(total_flops, context_len, model_config, strategy)

# 协同效应计算
compute_synergy_score(full_result, component_results, baseline)
```

---

## 注意事项

1. **GPU内存**: 部分实验需要较大GPU内存，如内存不足请减小batch_size或模型规模
2. **运行时间**: 完整实验套件可能需要10-15小时，建议分实验运行
3. **复现性**: 设置随机种子以确保结果可复现
4. **数据保存**: 所有实验结果自动保存到 `results/core/` 目录

---

## 论文对应章节

| 实验 | 论文章节 | 新增图表 |
|------|---------|---------|
| 实验1 | 3.1.1 PreNorm Score Dilution | 图1, 表X |
| 实验2 | 3.1.1 & 4.3.6 | 图2, 表Y |
| 实验3 | 3.4.3 Gradient Flow | 图3, 表X |
| 实验4 | 4.3.3 FLOP Equivalence | 图4, 表Z |
| 实验5 | 5.5 Ablation Study | 图5 |
| 实验6 | Appendix C | - |
