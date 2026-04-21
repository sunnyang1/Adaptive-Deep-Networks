# PRD: Batch Processing Integration

## 目标
将批量处理功能完整集成到 AdaptiveTransformer，实现并行处理多个样本。

## 需求

### 功能需求
1. `generate_batch()` - 批量生成方法
2. `forward_batch()` - 批量前向传播
3. 支持可变长度序列的 padding 和 masking
4. 与现有 qTTT、AttnRes、RaBitQ 兼容

### 性能需求
- 批量处理 4 个样本比单样本顺序处理快 40%+
- 内存使用可控（不 OOM）

## 验收标准

```python
# 测试代码
batch_size = 4
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

# 顺序处理
t0 = time.time()
for i in range(batch_size):
    model.generate(input_ids[i:i+1], ...)
t_sequential = time.time() - t0

# 批量处理
t0 = time.time()
model.generate_batch(input_ids, ...)
t_batch = time.time() - t0

assert t_batch < t_sequential * 0.6  # 40%+ speedup
```

## 技术方案
1. 在 AdaptiveTransformer 添加 batch 方法
2. 使用 torch.nn.utils.rnn.pad_sequence 处理变长
3. 创建 batch 版本的 qTTT adaptation
4. 添加 batch 版本的 AttnRes
