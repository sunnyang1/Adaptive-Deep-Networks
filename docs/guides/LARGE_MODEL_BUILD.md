# Large 模型 (AttnRes-L) 构建报告

## 模型配置

根据论文 §5.4.1 优化后的配置 (d_model/L_b ≈ 45, H/L_b ≈ 0.3):

| 参数 | 值 |
|------|-----|
| 模型大小 | Large (AttnRes-L) |
| 目标参数 | 27.5B |
| 层数 | 96 |
| 隐藏维度 | 4224 |
| 注意力头数 | 32 |
| 头维度 | 132 |
| MLP Ratio | 4 |
| MLP 维度 | 16896 |
| AttnRes 块数 | 16 |
| 词表大小 | 32000 |
| qTTT 最大步数 | 32 |
| qTTT span 长度 | 256 |
| d_model/L_b | 44.0 (论文最优 ~45) |
| H/L_b | 0.33 (论文最优 ~0.3) |

## 参数计算详情

### 1. Embedding 层
- Token Embedding: 32,000 × 4,224 = **0.14B**

### 2. Transformer 层 (共 96 层)

每层包含:

**Attention 层:**
- Q_proj: 4,224 × 4,224 = 17.8M
- K_proj: 4,224 × 4,224 = 17.8M
- V_proj: 4,224 × 4,224 = 17.8M
- O_proj: 4,224 × 4,224 = 17.8M
- **Attention 小计**: 71.4M × 96 = **6.85B**

**MLP 层 (SwiGLU):**
- Gate_proj: 4,224 × 16,896 = 71.4M
- Up_proj: 4,224 × 16,896 = 71.4M
- Down_proj: 16,896 × 4,224 = 71.4M
- **MLP 小计**: 214.1M × 96 = **20.55B**

**AttnRes 层:**
- Pseudo-query (attn): 4,224 = 0.004M
- Pseudo-query (mlp): 4,224 = 0.004M
- **AttnRes 小计**: 0.008M × 96 = **0.00B** (可忽略)

**每层总计**: 285.5M  
**96 层总计**: **27.41B**

### 3. 总参数

| 组件 | 参数 |
|------|------|
| Embedding | 0.14B |
| Transformer Layers | 27.41B |
| **总计** | **27.54B** |

## 内存需求

### 推理内存

| 精度 | 内存需求 |
|------|---------|
| FP32 | 110.2 GB |
| FP16 | 55.1 GB |
| BF16 | 55.1 GB |
| INT8 | 27.5 GB |
| INT4 | 13.8 GB |

### 训练内存 (AdamW + FP32)

- **估算**: ~440 GB (包含参数、梯度、优化器状态)
- **建议**: 使用分布式训练 + ZeRO-3 / FSDP

## 计算需求

### FLOPs (per token)

- 每层: 0.50 GFLOPs
- 96 层总计: **48.0 TFLOPs/token**

### 与 Medium/Small 模型对比

| 模型 | 参数 | 层数 | 隐藏维度 | 内存(BF16) |
|------|------|------|----------|-----------|
| Small | 3.3B | 48 | 2048 | 6.6 GB |
| Medium | 6.6B | 56 | 2688 | 13.1 GB |
| **Large** | **27.5B** | **96** | **4224** | **55.1 GB** |

## 硬件要求

### 推理

- **最低**: 1× A100 80GB (BF16) 或 2× A100 40GB
- **推荐**: 2× A100 80GB (提供更好的 batch size 灵活性)
- **量化**: 1× A100 40GB (INT8) 或 1× A6000 48GB (INT8)

### 训练

- **最低**: 8× A100 80GB (使用 ZeRO-3 + 梯度检查点)
- **推荐**: 16× A100 80GB 或 32× A100 40GB
- **预估时间**: 
  - 100B tokens, 8× A100: ~35 天
  - 100B tokens, 32× A100: ~9 天

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
  "num_layers": 96,
  "hidden_dim": 4224,
  "num_heads": 32,
  "mlp_ratio": 4,
  "num_blocks": 16,
  "max_qttt_steps": 32,
  "qttt_span_length": 256,
  "d_model_per_layer": 44.0,
  "heads_per_layer": 0.33
}
```

## 文件清单

- `src/models/configs.py` - 配置定义
- `src/models/adaptive_transformer.py` - 模型实现
- `results/large_model_config.json` - 生成的配置
- `scripts/model/build_large_model.py` - 构建/分析脚本

## 总结

Large 模型 (AttnRes-L) 结构分析完成:

- ✅ 配置验证: 96层 × 4224维度 (优化后)
- ✅ 参数计算: ~27.5B 
- ✅ 内存估算: 55GB (BF16)
- ✅ 硬件建议: 2× A100 80GB 用于推理
- ✅ 架构优化: d_model/L_b = 44.0, H/L_b = 0.33 (符合论文 §5.4.1)

注意: 实际创建模型需要 100GB+ 内存/显存，当前环境无法加载。
