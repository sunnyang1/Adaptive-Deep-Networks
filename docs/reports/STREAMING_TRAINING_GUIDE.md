# 流式加载训练指南

## 什么是流式加载？

流式加载 (Streaming) 允许在不下载完整数据集的情况下进行训练，数据实时从网络流式传输到 GPU。

### 优势
- **零本地存储**: 50GB 数据盘可训练 TB 级数据集
- **即时开始**: 无需等待数据下载
- **自动更新**: 始终使用最新版本的数据

### 适用场景
- AutoDL 等小磁盘云服务器
- 大数据集 (FineWeb, SlimPajama 等)
- 多机分布式训练

---

## 快速开始

### 1. 环境准备

```bash
# 运行 H20 专用安装脚本
bash scripts/autodl_h20_setup.sh

# 激活环境
conda activate adn
```

### 2. 单卡流式训练

```bash
# Small 模型 - 虚拟数据（测试用）
python scripts/training/train_streaming.py \
    --model-size small \
    --max-steps 10000

# Medium 模型 - FineWeb 数据集
python scripts/training/train_streaming.py \
    --model-size medium \
    --dataset fineweb \
    --dataset-config sample-10BT \
    --max-steps 100000
```

### 3. 4卡分布式流式训练

```bash
# Medium 模型 - 4卡 H20
torchrun --nproc_per_node=4 scripts/train_streaming.py \
    --model-size medium \
    --dataset fineweb \
    --max-steps 100000 \
    --batch-size 2

# Large 模型 - DeepSpeed + 4卡
deepspeed --num_gpus=4 scripts/train_streaming.py \
    --model-size large \
    --use-deepspeed \
    --dataset fineweb \
    --max-steps 50000
```

---

## 支持的预训练数据集

| 数据集 | 大小 | 命令 | 说明 |
|--------|------|------|------|
| **Dummy** | - | `--dataset dummy` | 虚拟数据，用于测试 |
| **FineWeb-Edu** | 1.3TB | `--dataset fineweb` | 高质量教育文本，推荐 |
| **SlimPajama** | 627GB | `--dataset slimpajama` | 去重后的 Pile |
| **OpenWebText** | 40GB | `--dataset openwebtext` | Reddit 链接文本 |

### FineWeb 配置选项

```bash
# 10B tokens 样本 (约 20GB 网络流量)
--dataset fineweb --dataset-config sample-10BT

# 100B tokens 样本
--dataset fineweb --dataset-config sample-100BT

# 完整数据集 (1.3TB)
--dataset fineweb --dataset-config default
```

---

## 关键参数说明

### 训练参数

```bash
--model-size {small,medium,large}  # 模型大小
--max-steps 100000                 # 训练步数 (流式用 steps 不用 epochs)
--batch-size 2                     # 每卡 batch size
--seq-len 2048                     # 序列长度
--learning-rate 3e-4              # 学习率
```

### 数据参数

```bash
--dataset {dummy,fineweb,slimpajama,openwebtext}  # 数据集
--dataset-config CONFIG                            # 数据集配置
```

### 保存与恢复

```bash
--save-every 5000                  # 每 5000 步保存 checkpoint
--output-dir /path/to/checkpoints  # checkpoint 保存路径
--resume /path/to/checkpoint.pt    # 从 checkpoint 恢复
```

---

## 磁盘空间管理

### 当前配置 (50GB 数据盘)

```bash
# 设置 HuggingFace 缓存到系统盘 (30GB)
export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface

# 限制缓存大小
export HF_DATASETS_IN_MEMORY_MAX_SIZE=10000000000  # 10GB
```

### Checkpoint 管理

```bash
# 只保留最新 checkpoint
python scripts/training/train_streaming.py \
    --model-size medium \
    --save-every 10000 \
    --max-steps 100000

# 定期清理旧 checkpoint
# 在训练脚本中添加自动清理逻辑
```

---

## 训练时长估算

基于 H20 4卡配置：

| 模型 | 数据集 | 步数 | 预估时间 | 费用 (¥20/小时) |
|------|--------|------|---------|----------------|
| Small | FineWeb 10B | 50K | ~3 天 | ~¥1440 |
| Medium | FineWeb 10B | 50K | ~7 天 | ~¥3360 |
| Medium | FineWeb 100B | 200K | ~28 天 | ~¥13440 |

---

## 故障排除

### 网络连接问题

```bash
# 如果 HuggingFace 连接不稳定，设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### 内存不足

```bash
# 减小 batch size
--batch-size 1

# 减小序列长度
--seq-len 1024

# 启用梯度检查点
--use-deepspeed
```

### 数据加载慢

```bash
# 增加预读取缓冲区 (在代码中修改)
# 修改 buffer_size 参数

# 使用更快的网络连接
# AutoDL 通常有较好带宽
```

---

## 进阶配置

### 自定义数据集

```python
# 修改 train_streaming.py 中的 dataset_map
 dataset_map = {
    'dummy': ('dummy', ''),
    'fineweb': ('HuggingFaceFW/fineweb-edu', 'sample-10BT'),
    'slimpajama': ('cerebras/SlimPajama-627B', ''),
    'openwebtext': ('openwebtext', ''),
    'your_dataset': ('your-org/your-dataset', ''),  # 添加自定义
}
```

### 混合精度训练

```bash
# 启用 BF16 (H20 支持)
python scripts/training/train_streaming.py \
    --model-size medium \
    --mixed-precision
```

### 学习率调度

```bash
# 自定义 warmup 步数
python scripts/training/train_streaming.py \
    --model-size medium \
    --warmup-steps 5000
```

---

## 完整示例

### 场景 1: 快速验证 (1天)

```bash
python scripts/training/train_streaming.py \
    --model-size small \
    --dataset fineweb \
    --dataset-config sample-10BT \
    --max-steps 10000 \
    --batch-size 8 \
    --save-every 5000
```

### 场景 2: 中等规模训练 (1周)

```bash
torchrun --nproc_per_node=4 scripts/train_streaming.py \
    --model-size medium \
    --dataset fineweb \
    --dataset-config sample-10BT \
    --max-steps 50000 \
    --batch-size 2 \
    --save-every 10000 \
    --output-dir /root/autodl-tmp/checkpoints
```

### 场景 3: 大规模训练 (1个月)

```bash
deepspeed --num_gpus=4 scripts/train_streaming.py \
    --model-size large \
    --use-deepspeed \
    --dataset fineweb \
    --dataset-config sample-100BT \
    --max-steps 200000 \
    --save-every 20000
```

---

## 相关脚本

| 脚本 | 用途 |
|------|------|
| `train_streaming.py` | 流式加载训练主脚本 |
| `autodl_h20_setup.sh` | H20 环境安装 |
| `quick_start_h20.sh` | 快速启动向导 |
| `ds_config_h20.json` | DeepSpeed 配置 |
