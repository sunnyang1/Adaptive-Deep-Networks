# 快速开始

## 运行所有实验

```bash
cd experiments

# 快速模式（推荐测试）
python run_experiments_unified.py --all --quick

# 完整模式
python run_experiments_unified.py --all

# 仅运行核心实验
python run_experiments_unified.py --category core

# 仅运行验证实验
python run_experiments_unified.py --category validation

# 使用 CPU
python run_experiments_unified.py --all --quick --device cpu
```

## 查看实验列表

```bash
# 列出所有实验
python run_experiments_unified.py --list

# 列出特定类别
python run_experiments_unified.py --list --category core
```

## 运行单个实验

### 使用统一脚本
```bash
# 运行特定实验
python run_experiments_unified.py --exp exp1

# 运行多个实验
python run_experiments_unified.py --exp exp1 exp2 exp3

# 干运行（查看会执行什么，不实际运行）
python run_experiments_unified.py --exp exp1 --dry-run
```

### 直接运行单个实验脚本
```bash
# 实验1: Representation Burial
python core/exp1_representation_burial/run_exp1.py --num_samples 10

# 实验2: Margin分析
python core/exp2_margin_analysis/run_exp2.py --context_lengths 1024 4096

# 实验3: 梯度流
python core/exp3_gradient_flow/run_exp3.py --num_steps 100

# 实验4: FLOP等价
python core/exp4_flop_equivalence/run_exp4.py --total_flops 1e13

# 实验5: 协同效应
python core/exp5_synergy/run_exp5.py

# 实验6: 辅助验证
python core/exp6_auxiliary/run_exp6.py
```

### TurboQuant验证
```bash
# 使用统一脚本
python run_experiments_unified.py --exp val_turboquant

# 或直接运行
python ../scripts/validate_turboquant_setup.py
```

## 输出位置

所有结果保存在 `results/experiments/` 目录：
- `{category}/{exp_id}/output.log` - 实验输出日志
- `execution_summary.json` - 执行汇总报告

## 使用 Makefile

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

# 使用CPU快速运行
make quick-cpu
```

## 快速检查结果

```bash
# 查看汇总
cat results/experiments/execution_summary.json | python -m json.tool

# 检查成功状态
python -c "import json; d=json.load(open('results/experiments/execution_summary.json')); print('成功:', d['success'], '/', d['total'])"
```

## 实验分类

| 类别 | 实验 | 说明 |
|------|------|------|
| core | exp1-exp6 | 核心验证实验 |
| validation | val_small, val_turboquant | 模型验证 |
| paper | paper_metrics | 论文指标生成 |

## 完整帮助

```bash
python run_experiments_unified.py --help
```
