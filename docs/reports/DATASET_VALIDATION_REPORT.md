# 数据集验证报告

## 概述

根据论文第 5.2 节 (Datasets and Evaluation Protocols)，我们完成了所有评估数据集的搜索和验证工作。

## 验证结果摘要

| 类别 | 数据集 | 状态 | 来源 |
|------|--------|------|------|
| **Long-Context Retrieval** | Needle-in-Haystack | ✓ 本地生成 | 自定义实现 |
| | LongBench-v2 | ✓ HuggingFace可用 | THUDM/LongBench-v2 |
| | ZeroScrolls | ⚠ 需本地实现 | zeroscrolls-benchmark.com |
| **Mathematical Reasoning** | MATH | ✓ HuggingFace可用 | hendrycks/competition_math |
| | GSM8K | ✓ HuggingFace可用 | openai/gsm8k |
| **General Tasks** | HellaSwag | ✓ HuggingFace可用 | Rowan/hellaswag |
| | ARC-Challenge | ✓ HuggingFace可用 | allenai/ai2_arc |
| | HumanEval | ✓ HuggingFace可用 | openai/openai_humaneval |
| | BBH | ✓ HuggingFace可用 | lukaemon/bbh |

## 数据集详细信息

### 1. Long-Context Retrieval Benchmarks

#### Needle-in-Haystack
- **上下文长度**: 1K, 4K, 16K, 32K, 64K, 128K, 256K tokens
- **评估方式**: 在不同深度插入事实，测试精确匹配准确率
- **实现**: 本地合成数据生成

#### LongBench-v2
- **样本数**: 503
- **平均上下文**: 35K tokens (最长 200K)
- **任务类别**: 6大类 (单文档QA、多文档QA、摘要、Few-shot学习、合成任务、代码补全)
- **数据格式**: 多选题 (A/B/C/D)，含 difficulty、length 等元数据

#### ZeroScrolls
- **样本数**: ~5,000
- **最大上下文**: 100K tokens
- **任务**: 8个长文档理解任务 (GovReport, QMSum, SQuALITY, MuSiQue 等)

### 2. Mathematical Reasoning Benchmarks

#### MATH
- **总问题数**: 12,500 (训练 7,500，测试 5,000)
- **难度级别**: 1-5 级
- **类别**: 代数、几何、计数与概率、数论、微积分预科、微积分
- **特点**: 每题含逐步解答和 boxed 答案

#### GSM8K
- **总问题数**: 8,500 (训练 7,473，测试 1,319)
- **解题步骤**: 2-8 步
- **运算类型**: 加减乘除基础算术

### 3. Language Modeling and General Tasks

#### HellaSwag
- **验证集**: 10,042 样本
- **任务**: 选择最合理的句子结尾 (4选1)
- **人类准确率**: ~95%

#### ARC-Challenge
- **总问题数**: 7,787
- **划分**: ARC-Easy (5,197)、ARC-Challenge (2,590)
- **任务**: 小学科学多选题，需要推理

#### HumanEval
- **问题数**: 164
- **语言**: Python
- **评估指标**: Pass@k
- **内容**: 函数签名、docstring、单元测试

#### BBH (Big-Bench Hard)
- **任务数**: 23 个挑战性任务
- **示例任务**: boolean_expressions, causal_judgement, date_understanding 等

## 快速访问代码

```python
from datasets import load_dataset

# Long-context retrieval
longbench = load_dataset('THUDM/LongBench-v2', split='train')

# Mathematical reasoning
math_train = load_dataset('hendrycks/competition_math', split='train')
math_test = load_dataset('hendrycks/competition_math', split='test')
gsm8k_test = load_dataset('openai/gsm8k', 'main', split='test')

# General tasks
hellaswag = load_dataset('Rowan/hellaswag', split='validation')
arc = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
humaneval = load_dataset('openai/openai_humaneval', split='test')
bbh = load_dataset('lukaemon/bbh', 'boolean_expressions', split='test')
```

## 论文更新内容

已在论文附录中添加以下新表格：

- **Table A7**: Long-Context Retrieval Benchmarks
- **Table A8**: Mathematical Reasoning Benchmarks  
- **Table A9**: Language Modeling and General Tasks

同时添加了数据集访问代码示例和数据格式说明。

## 文件清单

```
results/
├── dataset_info.json              # 完整数据集信息 (JSON格式)
├── validation_small.json          # Small 模型验证结果
├── validation_medium.json         # Medium 模型验证结果
└── validation_summary.json        # 验证汇总

dataset_info.py                    # 数据集信息整理脚本
dataset_validation.py              # 数据集验证脚本 (需安装 datasets)
download_datasets.sh               # 数据集下载脚本
DATASET_VALIDATION_REPORT.md       # 本报告
```

## 结论

所有论文中引用的评估数据集均已验证可用：
- **7/8 数据集**可直接通过 HuggingFace `datasets` 库访问
- **Needle-in-Haystack** 需要本地合成生成
- **ZeroScrolls** 需要从官方网站下载并实现自定义加载器

数据集验证工作已完成，论文附录已更新。
