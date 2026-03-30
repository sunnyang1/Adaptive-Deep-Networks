# 流式加载功能更新摘要

## 更新日期
2026-03-25

## 新增功能
支持 HuggingFace datasets 流式加载 (streaming=True)，适用于磁盘空间有限的环境（如 AutoDL 50GB 数据盘）。

## 新增文件

### 核心脚本
1. **`scripts/train_streaming.py`** - 流式加载训练主脚本
   - 支持 Dummy/FineWeb/SlimPajama/OpenWebText 数据集
   - 支持单卡/多卡分布式训练
   - 支持 DeepSpeed
   - 支持 checkpoint 断点续训
   - 支持混合精度 (BF16)

### 文档
2. **`STREAMING_TRAINING_GUIDE.md`** - 流式加载完整指南
   - 快速开始示例
   - 支持的预训练数据集列表
   - 参数说明
   - 故障排除

### 更新摘要
3. **`STREAMING_UPDATE_SUMMARY.md`** - 本文件

## 修改的文件

### H20 配置脚本
- `scripts/autodl_h20_setup.sh`
  - 添加流式加载命令示例
  - 强调流式加载推荐用法

- `scripts/quick_start_h20.sh`
  - 更新为流式加载优先的菜单
  - 添加 FineWeb 数据集示例

### 文档
- `README.md`
  - 添加 Streaming Training 章节
  - 强调零本地存储特性

- `MODEL_PARAMS_REPORT.md`
  - 添加流式加载快速参考
  - 更新数据盘空间建议

## 使用方法

### 基础用法

```bash
# 虚拟数据测试
python scripts/training/train_streaming.py --model-size small --max-steps 1000

# FineWeb 数据集
python scripts/training/train_streaming.py \
    --model-size medium \
    --dataset fineweb \
    --dataset-config sample-10BT \
    --max-steps 10000
```

### H20 4卡训练

```bash
# 4卡分布式
torchrun --nproc_per_node=4 scripts/train_streaming.py \
    --model-size medium \
    --dataset fineweb \
    --max-steps 100000

# Large 模型 + DeepSpeed
deepspeed --num_gpus=4 scripts/train_streaming.py \
    --model-size large \
    --use-deepspeed \
    --dataset fineweb
```

## 关键技术点

### 1. 流式数据集类
```python
class StreamingTextDataset(IterableDataset):
    def __init__(self, dataset_name, streaming=True):
        self.dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=True  # 关键参数
        )
```

### 2. 无限迭代器
```python
data_iter = iter(dataloader)
while step < max_steps:
    try:
        input_ids, targets = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)  # 重新启动
```

### 3. 检查点保存
- 每 N 步自动保存
- 支持断点续训 (--resume)
- 包含 optimizer 和 scaler 状态

## 数据集支持

| 数据集 | 大小 | 用途 |
|--------|------|------|
| Dummy | - | 测试/验证 |
| FineWeb-Edu | 1.3TB | 高质量教育文本 |
| SlimPajama | 627GB | 通用预训练 |
| OpenWebText | 40GB | 小规模测试 |

## 磁盘空间对比

| 方式 | Small 模型 | Medium 模型 | Large 模型 |
|------|-----------|-------------|-----------|
| 传统下载 | 50GB+ | 200GB+ | 500GB+ |
| **流式加载** | **<1GB** | **<1GB** | **<1GB** |

## 下一步计划

- [ ] 添加更多预训练数据集支持
- [ ] 实现数据预取优化
- [ ] 添加训练可视化 (wandb/tensorboard)
- [ ] 支持自定义数据混合
