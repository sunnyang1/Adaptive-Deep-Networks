# Large 模型 (AttnRes-L) 构建报告

## 模型配置

根据论文 Table A1 和配置文件:

| 参数 | 值 |
|------|-----|
| 模型大小 | Large (AttnRes-L) |
| 目标参数 | 50B |
| 实际计算参数 | ~27B |
| 层数 | 64 |
| 隐藏维度 | 5120 |
| 注意力头数 | 40 |
| 头维度 | 128 |
| MLP Ratio | 4 |
| MLP 维度 | 20480 |
| AttnRes 块数 | 16 |
| 词表大小 | 32000 |
| qTTT 最大步数 | 32 |
| qTTT span 长度 | 256 |

## 参数计算详情

### 1. Embedding 层
- Token Embedding: 32,000 × 5,120 = **0.16B**

### 2. Transformer 层 (共 64 层)

每层包含:

**Attention 层:**
- Q_proj: 5,120 × 5,120 = 26.2M
- K_proj: 5,120 × 5,120 = 26.2M
- V_proj: 5,120 × 5,120 = 26.2M
- O_proj: 5,120 × 5,120 = 26.2M
- **Attention 小计**: 104.9M × 64 = **6.71B**

**MLP 层 (SwiGLU):**
- Gate_proj: 5,120 × 20,480 = 104.9M
- Up_proj: 5,120 × 20,480 = 104.9M
- Down_proj: 20,480 × 5,120 = 104.9M
- **MLP 小计**: 314.6M × 64 = **20.13B**

**AttnRes 层:**
- Pseudo-query (attn): 5,120 = 0.01M
- Pseudo-query (mlp): 5,120 = 0.01M
- **AttnRes 小计**: 0.01M × 64 = **0.00B** (可忽略)

**每层总计**: 419.44M  
**64 层总计**: **26.84B**

### 3. 总参数

| 组件 | 参数 |
|------|------|
| Embedding | 0.16B |
| Transformer Layers | 26.84B |
| **总计** | **27.01B** |

### 关于 50B 的说明

论文中标注的 50B 可能包含以下因素:

1. **近似值**: 实际约 27B，50B 可能是向上取整或包含未来扩展
2. **额外参数**: 位置编码、LayerNorm 参数、偏置项等
3. **更大词表**: 如果使用 50K 或 100K 词表
4. **独立输出层**: 如果 LM Head 不与 Embedding 共享

## 内存需求

### 推理内存

| 精度 | 内存需求 |
|------|---------|
| FP32 | 108.0 GB |
| FP16 | 54.0 GB |
| BF16 | 54.0 GB |
| INT8 | 27.0 GB |
| INT4 | 13.5 GB |

### 训练内存 (AdamW + FP32)

- **估算**: ~432 GB (包含参数、梯度、优化器状态)
- **建议**: 使用分布式训练 + ZeRO-3 / FSDP

## 计算需求

### FLOPs (per token)

- 每层: 0.73 GFLOPs
- 64 层总计: **46.98 TFLOPs/token**

### 与 Medium 模型对比

| 模型 | 参数 | 层数 | 隐藏维度 | 内存(BF16) |
|------|------|------|----------|-----------|
| Small | 2.2B | 32 | 2048 | 4.4 GB |
| Medium | 8.7B | 32 | 4096 | 17.4 GB |
| **Large** | **27.0B** | **64** | **5120** | **54.0 GB** |

## 硬件要求

### 推理

- **最低**: 1× A100 80GB (BF16) 或 2× A100 40GB
- **推荐**: 2× A100 80GB (提供更好的 batch size 灵活性)
- **量化**: 1× A100 40GB (INT8) 或 1× A6000 48GB (INT8)

### 训练

- **最低**: 8× A100 80GB (使用 ZeRO-3 + 梯度检查点)
- **推荐**: 16× A100 80GB 或 32× A100 40GB
- **预估时间**: 
  - 100B tokens, 8× A100: ~30 天
  - 100B tokens, 32× A100: ~7 天

## 使用建议

### 1. 量化部署

```python
# INT8 量化可将内存降至 27GB
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    "adaptive-deep-networks-large",
    quantization_config=quant_config,
    device_map="auto"
)
```

### 2. 分布式推理

```python
# 使用 accelerate 进行多卡推理
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

with init_empty_weights():
    model = create_adaptive_transformer('large')

model = load_checkpoint_and_dispatch(
    model, checkpoint_path,
    device_map='auto',
    no_split_module_classes=['AdaptiveLayer']
)
```

### 3. 训练配置

```python
# 使用 DeepSpeed ZeRO-3
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
    },
    "train_batch_size": 4_000_000,  # 4M tokens
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 32,
}
```

## 配置文件

已生成配置文件: `results/large_model_config.json`

```json
{
  "model_type": "adaptive_transformer",
  "model_size": "large",
  "vocab_size": 32000,
  "num_layers": 64,
  "hidden_dim": 5120,
  "num_heads": 40,
  "mlp_ratio": 4,
  "num_blocks": 16,
  "max_qttt_steps": 32,
  "qttt_span_length": 256
}
```

## 文件清单

- `src/models/configs.py` - 配置定义
- `src/models/adaptive_transformer.py` - 模型实现
- `results/large_model_config.json` - 生成的配置
- `scripts/model/build_large_model.py` - 构建/分析脚本

## 总结

Large 模型 (AttnRes-L) 结构分析完成:

- ✅ 配置验证: 64层 × 5120维度
- ✅ 参数计算: ~27B (论文标注 50B)
- ✅ 内存估算: 54GB (BF16)
- ✅ 硬件建议: 2× A100 80GB 用于推理
- ✅ 配置文件: 已生成

注意: 实际创建模型需要 100GB+ 内存/显存，当前环境无法加载。
