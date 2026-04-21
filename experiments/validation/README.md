# Paper Validation Scripts

验证论文中所有关键表格和声明的实验脚本。

## 快速开始

```bash
# 运行所有验证
python run_all_validations.py

# 运行单个验证
python table1_representation_burial.py
python table4_needle_haystack.py

# 超长上下文测试 (up to 1M)
python extreme_context_scaling.py

# 渐进式上下文测试
python progressive_context_test.py --max-context 1048576
```

## 验证脚本列表

### 核心表格验证


| 脚本                                | 验证目标                           | 关键指标                      |
| --------------------------------- | ------------------------------ | ------------------------- |
| `table1_representation_burial.py` | Table 1: Representation Burial | 1.06× vs 13.5×衰减, 91L有效深度 |
| `table2_gradient_flow.py`         | Table 2: Gradient Flow         | CV改善 7.6× (0.11 vs 0.84)  |
| `turboquant_compression.py`       | TurboQuant压缩                   | 6×+压缩比, 5.7×KV Cache缩减    |
| `table4_needle_haystack.py`       | Table 4: Needle-in-Haystack    | 86.9%平均, 68.2%@256K       |
| `table6_math.py`                  | Table 6: MATH Dataset          | 52.3% (8.7B = 50B)        |
| `table7_synergy.py`               | Table 7: 组件协同                  | 协同系数 1.18                 |


### 扩展测试


| 脚本                            | 描述       | 范围                      |
| ----------------------------- | -------- | ----------------------- |
| `extreme_context_scaling.py`  | 极端上下文测试  | 128K → 256K → 512K → 1M |
| `progressive_context_test.py` | 渐进式上下文测试 | 可配置步长，1K to 1M          |


## 使用方法

### 基础验证

```bash
cd experiments/validation

# 所有基础验证
python run_all_validations.py

# 带输出目录
python run_all_validations.py --output-dir results/validation
```

### 极端上下文测试 (1M tokens)

```bash
# 标准极端测试 (128K, 256K, 512K, 1M)
python extreme_context_scaling.py

# 带可视化输出
python extreme_context_scaling.py --output-dir results/validation
```

预期输出：

- 128K: ~78.2% accuracy
- 256K: ~68.2% accuracy  
- 512K: ~58% accuracy
- 1M: ~48% accuracy (目标: >45%)

### 渐进式上下文测试

```bash
# 标准序列 (1K to 1M)
python progressive_context_test.py

# 自定义最大长度
python progressive_context_test.py --max-context 524288

# 指定特定长度
python progressive_context_test.py --lengths 4096 8192 16384 32768 65536 131072

# 自定义步长 (1.5x倍增)
python progressive_context_test.py --step-factor 1.5 --max-context 500000

# 快速模式
python progressive_context_test.py --quick

# 仅测试 ADB
python progressive_context_test.py --models adb
```

## 输出文件

每个脚本生成：

- **控制台**: 实时验证结果
- **JSON**: `results/validation/{test_name}_results.json`
- **PNG**: `results/validation/{test_name}.png` (可视化图表)

## 汇总报告

运行 `run_all_validations.py` 后生成：

- `results/validation/validation_summary.json` - 所有验证结果汇总

```bash
# 查看汇总
cat results/validation/validation_summary.json | python -m json.tool
```

## 预期结果速查

### Table 1: Representation Burial

- PreNorm: 13.5×衰减, 18L有效深度
- AttnRes: **1.06×衰减, 91L有效深度** ✅

### Table 4: Needle-in-Haystack


| Context     | Baseline  | ADB Target  |
| ----------- | --------- | ----------- |
| 4K          | 87.5%     | 98.5%       |
| 32K         | 22.1%     | 91.3%       |
| 128K        | 3.2%      | 78.2%       |
| 256K        | 1.5%      | 68.2%       |
| **Average** | **38.2%** | **86.9%** ✅ |


### Extreme Context (1M)

- Target: >45% accuracy at 1M context
- Expected: ~48% with graceful degradation

## 注意事项

1. 当前脚本使用**模拟数据**进行框架验证
2. 真实模型验证需要：
  - 加载预训练模型权重
  - 在真实数据集上运行
  - 替换 `simulate_`* 函数为实际推理

## 添加新的验证

在 `run_all_validations.py` 中的 `VALIDATIONS` 列表添加：

```python
{
    'id': 'my_validation',
    'name': 'My Validation Test',
    'script': 'my_validation.py',
    'description': '描述'
},
```

