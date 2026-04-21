# PRD: 增量 KV Cache 新模型结构

## 1. 引言

### 1.1 目标
实现 O(L) 复杂度的增量 KV Cache 更新，替代现有的 O(T×L) 重建方式。

### 1.2 背景
当前 `get_kv_cache()` 每次重建整个缓存，导致长序列生成成本高昂。

### 1.3 范围
- ✅ IncrementalGenerator 包装器类
- ✅ 显式状态管理 (IncrementalState)
- ✅ 单 token 前向传播
- ✅ 集成到现有 generate()
- ❌ 底层 C++ 优化 (未来)

## 2. 用户故事

### US1: IncrementalState 数据结构
**作为** 开发者，**我希望** 有结构化的状态管理，**从而** 清晰跟踪中间结果。

**验收标准:**
- [ ] IncrementalState dataclass 定义
- [ ] 包含 kv_caches, block_reps, partial_block
- [ ] 状态验证方法
- [ ] 序列化/反序列化支持

**预估:** 10分钟

### US2: AdaptiveLayer 单 token 支持
**作为** 开发者，**我希望** 每层支持单 token 前向，**从而** 复用历史状态。

**验收标准:**
- [ ] 添加 forward_single_token() 方法
- [ ] 正确处理 KV cache 追加
- [ ] 正确处理 block_representations 更新
- [ ] 数值与批量前向一致

**预估:** 20分钟

### US3: IncrementalGenerator 实现
**作为** 用户，**我希望** 有高级包装器，**从而** 简化增量生成。

**验收标准:**
- [ ] prefill() 方法初始化状态
- [ ] step() 方法单步生成
- [ ] generate() 方法循环生成
- [ ] 性能监控（时间/内存）

**预估:** 25分钟

### US4: 集成与测试
**作为** 开发者，**我希望** 新功能通过完整测试，**从而** 确保正确性。

**验收标准:**
- [ ] 数值一致性测试（vs 原始实现）
- [ ] 性能基准测试
- [ ] 长序列稳定性测试
- [ ] 内存使用测试

**预估:** 20分钟

## 3. 技术需求

### 3.1 数据结构
```python
@dataclass
class IncrementalState:
    kv_caches: List[KVCache]           # Per layer
    block_representations: List[Tensor] # Per completed block
    partial_block: Tensor               # Current block accumulation
    seq_len: int                        # Current sequence length
```

### 3.2 核心算法
```python
def step(self, new_token_id):
    # 1. 获取新 token 的 embedding
    new_hidden = embedding(new_token_id)
    
    # 2. 逐层处理
    for layer_idx, layer in enumerate(self.layers):
        # 2.1 AttnRes: 使用现有的 block_representations
        h_attn = attnres(block_reps, partial_block)
        
        # 2.2 拼接历史和新 token
        layer_input = concat([cached_hidden, new_hidden])
        
        # 2.3 只计算新 token 的 KV
        new_k, new_v = compute_kv(layer_input[:, -1:])
        
        # 2.4 追加到 cache
        kv_caches[layer_idx] = append(kv_caches[layer_idx], new_k, new_v)
        
        # 2.5 更新 block state
        partial_block = partial_block + attn_out
        
        # 2.6 检查 block 边界
        if is_block_boundary(layer_idx):
            block_representations.append(partial_block)
            partial_block = zeros()
    
    # 3. 输出 logits
    return logits[:, -1, :]
```

### 3.3 API 设计
```python
# 使用方式 1: 包装器模式
generator = IncrementalGenerator(model)
generator.prefill(input_ids)
for _ in range(max_new_tokens):
    next_token = generator.step()
    
# 使用方式 2: 便捷方法
output = model.generate_fast(input_ids, max_new_tokens=100)
```

## 4. 设计考虑

### 4.1 数值精度
- 使用相同的浮点精度 (FP32/FP16/BF16)
- 累积误差监控
- 与原始实现对比测试

### 4.2 内存管理
- 显式管理 KV cache 追加
- 避免不必要的张量复制
- 支持梯度检查点（训练时）

### 4.3 错误处理
- 状态一致性检查
- 序列长度不匹配检测
- 优雅的降级到原始实现

## 5. 成功指标
- 速度提升: 5-10x for >4K sequences
- 内存稳定: 不随长度线性增长
- 数值一致: < 1e-6 差异
- API 兼容: 100% 向后兼容
