# Adaptive Deep Networks: 实验套件

本目录包含验证论文理论声明的完整实验实现。

## 目录结构

```
experiments/
├── README.md                          # 本文件
├── run_experiments_unified.py         # 统一实验运行脚本（新版）
├── configs/                           # 实验配置文件
│   └── default.yaml
├── core/                              # 核心验证实验（原 exp1-exp6）
│   ├── exp1_representation_burial/    # Representation Burial 测量
│   ├── exp2_margin_analysis/          # Logit Margin 分析
│   ├── exp3_gradient_flow/            # 梯度流测量
│   ├── exp4_flop_equivalence/         # FLOP 等价验证
│   ├── exp5_synergy/                  # 组件协同效应
│   └── exp6_auxiliary/                # 辅助验证实验
├── turboquant/                        # TurboQuant 相关实验
│   └── (待添加)
├── benchmarks/                        # 基准测试
│   └── (待添加)
├── docs/                              # 实验文档
│   ├── core_experiments.md            # 核心实验详细说明
│   └── turboquant_experiments.md      # TurboQuant 实验步骤
├── utils/                             # 实验工具函数
│   ├── __init__.py
│   └── measurement.py
└── results/                           # 实验结果输出
    ├── core/                          # 核心实验结果
    ├── turboquant/                    # TurboQuant 结果
    └── benchmarks/                    # 基准测试结果
```

## 快速开始

### 运行所有实验（推荐）

```bash
# 快速模式（减少计算量，用于测试）
python run_experiments_unified.py --all --quick

# 完整模式
python run_experiments_unified.py --all

# 查看所有可用实验
python run_experiments_unified.py --list
```

### 使用 Makefile

```bash
# 列出所有实验
make list

# 快速运行所有实验
make quick

# 完整运行所有实验
make full

# 仅运行核心实验
make core

# 运行验证实验
make validate

# 生成论文指标
make paper-metrics

# 使用CPU
make quick-cpu
```

### 运行单个核心实验

```bash
# 实验1: Representation Burial
python core/exp1_representation_burial/run_exp1.py --num_samples 50

# 实验2: Margin分析
python core/exp2_margin_analysis/run_exp2.py --context_lengths 1024 4096 16384

# 实验3: 梯度流
python core/exp3_gradient_flow/run_exp3.py --num_steps 1000

# 实验4: FLOP等价
python core/exp4_flop_equivalence/run_exp4.py --total_flops 5e14

# 实验5: 协同效应
python core/exp5_synergy/run_exp5.py

# 实验6: 辅助验证
python core/exp6_auxiliary/run_exp6.py
```

### 使用统一脚本运行

```bash
# 运行所有实验
python run_experiments_unified.py --all

# 运行特定类别
python run_experiments_unified.py --category core
python run_experiments_unified.py --category validation

# 运行特定实验
python run_experiments_unified.py --exp exp1 exp2

# 使用 CPU
python run_experiments_unified.py --all --device cpu

# 快速模式
python run_experiments_unified.py --all --quick

# 干运行（查看会执行什么）
python run_experiments_unified.py --all --dry-run
```

## 实验分类

### 核心实验 (core/)

| 实验 | 名称 | 目标 | 预计时间 |
|------|------|------|----------|
| exp1 | Representation Burial | 验证PreNorm信号衰减 | 1-2小时 |
| exp2 | Logit Margin分析 | 验证对数margin要求 | 2-3小时 |
| exp3 | 梯度流测量 | 验证梯度流改善 | 1-2小时 |
| exp4 | FLOP等价验证 | 验证 T_think ≈ 2*N*k | 2-3小时 |
| exp5 | 组件协同效应 | 验证三组件协同 | 2-3小时 |
| exp6 | 辅助验证 | 超参数敏感性 | 2-3小时 |

详细说明：[docs/core_experiments.md](docs/core_experiments.md)

### TurboQuant 实验 (turboquant/)

| 实验 | 名称 | 目标 |
|------|------|------|
| TQ-1 | PolarQuant 压缩验证 | 验证 6x+ 压缩比 |
| TQ-2 | Polar qTTT 效率 | 验证 50% 参数减少 |
| TQ-3 | 深度优先策略 | 验证 2.4x 吞吐提升 |
| TQ-4 | 端到端集成 | 综合性能验证 |

详细说明：[docs/turboquant_experiments.md](docs/turboquant_experiments.md)

### 基准测试 (benchmarks/)

- Needle-in-Haystack 长上下文检索
- MATH 数学推理
- LongBench-v2 综合评估

## 工具函数

`utils/measurement.py` 提供以下测量功能：

```python
# Representation Burial测量
from experiments.utils import measure_representation_burial

# Attention Margin测量
from experiments.utils import measure_attention_margin

# 梯度流测量
from experiments.utils import measure_gradient_statistics

# FLOP测量
from experiments.utils import measure_actual_flops

# 协同效应计算
from experiments.utils import compute_synergy_score
```

## 依赖安装

```bash
pip install torch numpy matplotlib seaborn tqdm pyyaml
```

## 论文对应关系

| 实验 | 论文章节 | 新增图表 |
|------|---------|---------|
| exp1 | 3.1.1 PreNorm Score Dilution | 图1, 表X |
| exp2 | 3.1.1 & 4.3.6 | 图2, 表Y |
| exp3 | 3.4.3 Gradient Flow | 图3, 表X |
| exp4 | 4.3.3 FLOP Equivalence | 图4, 表Z |
| exp5 | 5.5 Ablation Study | 图5 |
| TQ-x | TurboQuant Section | 表TQ |

## 迁移说明

本次整理将原有实验从根目录移至 `core/` 子目录：
- `exp1_representation_burial/` → `core/exp1_representation_burial/`
- `exp2_margin_analysis/` → `core/exp2_margin_analysis/`
- ...以此类推

运行脚本已更新路径，使用方式保持不变。

## 引用

```bibtex
@article{adaptive_deep_networks_2026,
  title={Adaptive Deep Networks: Integrating Attention Residuals, Dynamic Computation, and Test-Time Adaptation},
  author={[Authors]},
  journal={arXiv preprint},
  year={2026}
}
```
