# Real Model Validation Framework

真实模型验证框架，用于验证论文中的所有关键声明。

## 结构

```
experiments/real_model/
├── __init__.py              # 模块入口
├── README.md                # 本文件
├── datasets/                # 数据集
│   ├── __init__.py
│   └── needle_dataset.py    # Needle-in-Haystack 数据生成
├── model_loader.py          # 模型加载工具
├── needle_haystack_real.py  # NIH 测试
├── memory_profiler.py       # 内存分析
├── gradient_analyzer.py     # 梯度流分析
└── validator.py             # 统一验证入口
```

## 使用方式

### 1. 运行所有测试

```bash
# 使用检查点
python experiments/real_model/validator.py \
    --checkpoint checkpoints/adb_medium.pt \
    --all

# 使用预定义配置
python experiments/real_model/validator.py \
    --size medium \
    --all
```

### 2. 单独运行测试

```bash
# Needle-in-Haystack
python experiments/real_model/validator.py \
    --checkpoint checkpoints/adb_medium.pt \
    --test needle \
    --max-length 131072 \
    --num-samples 10

# 内存分析
python experiments/real_model/validator.py \
    --checkpoint checkpoints/adb_medium.pt \
    --test memory

# 梯度流分析
python experiments/real_model/validator.py \
    --checkpoint checkpoints/adb_medium.pt \
    --test gradient

# 吞吐量测试
python experiments/real_model/validator.py \
    --checkpoint checkpoints/adb_medium.pt \
    --test throughput
```

### 3. 直接运行测试模块

```bash
# Needle-in-Haystack
python experiments/real_model/needle_haystack_real.py \
    --checkpoint checkpoints/adb_medium.pt \
    --lengths 4096 16384 65536 \
    --num-samples 5

# 内存分析
python experiments/real_model/memory_profiler.py \
    --checkpoint checkpoints/adb_medium.pt \
    --context-lengths 4096 8192 16384 32768
```

## 论文声明验证清单


| 声明                      | 测试                | 目标值     | 文件                      |
| ----------------------- | ----------------- | ------- | ----------------------- |
| Table 4: NIH 4K         | needle_haystack   | 98.5%   | needle_haystack_real.py |
| Table 4: NIH 16K        | needle_haystack   | 91.3%   | needle_haystack_real.py |
| Table 4: NIH 64K        | needle_haystack   | 78.2%   | needle_haystack_real.py |
| Table 4: NIH 128K       | needle_haystack   | 68.2%   | needle_haystack_real.py |
| Table 4: NIH Avg        | needle_haystack   | 86.9%   | needle_haystack_real.py |
| TurboQuant: KV Cache    | memory_profiler   | 5.7x    | memory_profiler.py      |
| TurboQuant: Compression | memory_profiler   | 6x      | memory_profiler.py      |
| Table 2: CV             | gradient_analyzer | 0.11    | gradient_analyzer.py    |
| Throughput              | validator         | 110 t/s | validator.py            |


## 输出格式

所有测试结果保存为 JSON，包含：

- 原始测量数据
- 与目标的对比
- PASS/FAIL 状态
- 可视化图表（PNG）

输出目录: `results/real_model/`