# H20-NVLink 4卡配置摘要

## 更新日期
2026-03-25

## 硬件配置
- **GPU**: H20-NVLink 96GB × 4
- **CPU**: AMD EPYC 9K84 16核
- **内存**: 150 GB
- **数据盘**: 50 GB
- **CUDA**: ≤ 13.0
- **驱动**: 580.65.06
- **总显存**: 384 GB

## 训练配置速查表

| 模型 | 单卡 Batch | 4卡 Batch | 显存/卡 | 时间/epoch |
|------|-----------|----------|---------|-----------|
| Small (2.2B) | 8 | 32 | ~20GB | ~30min |
| Medium (8.7B) | 2 | 8 | ~90GB | ~1.5h |
| Large (27B) | - | 4* | ~80GB | ~5h |

*Large 模型必须使用 DeepSpeed ZeRO-3

## 常用命令

```bash
# 环境设置
bash scripts/autodl_h20_setup.sh
conda activate adn

# Small 模型 - 4卡训练
torchrun --nproc_per_node=4 scripts/train_model.py \
    --model-size small \
    --batch-size 8 \
    --epochs 3

# Medium 模型 - 4卡训练
torchrun --nproc_per_node=4 scripts/train_model.py \
    --model-size medium \
    --batch-size 2 \
    --epochs 3

# Large 模型 - 4卡 DeepSpeed
deepspeed --num_gpus=4 scripts/train_model.py \
    --model-size large \
    --deepspeed scripts/ds_config_h20.json
```

## 注意事项

1. **Large 模型最低要求 4 卡**，你的配置刚好满足
2. **数据盘 50GB 可能不足**，建议使用网盘或精简数据集
3. **NVLink 4卡互联** 提供高速通信，分布式训练效率高
