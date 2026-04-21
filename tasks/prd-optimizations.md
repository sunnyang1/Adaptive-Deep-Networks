# PRD: ADN 性能优化

## 1. RaBitQ 缓存解压优化

### 现状
- 框架已添加 `_rabitq_kv_cache` 和 `invalidate_rabitq_cache`
- 需要完成 `_get_cached_rabitq_kv` 实现
- 当前每次 forward 都重新解压，导致 185×  slowdown

### 需求
完成缓存机制实现，确保：
- 首次解压后缓存结果
- 输入变化时使缓存失效
- 内存使用可控

### 验收标准
```python
# 测试伪代码
model.init_rabitq_caches(total_bits=1)

# 第一次 forward - 解压并缓存
t0 = time.time()
output1 = model(input_ids, use_rabitq=True, use_attnres=True)
t1 = time.time() - t0

# 第二次 forward - 使用缓存（相同输入）
output2 = model(input_ids, use_rabitq=True, use_attnres=True)
t2 = time.time() - t0

# 验证：第二次应该明显更快
assert t2 < t1 * 0.5  # 至少 2× 加速
```

---

## 2. qTTT JIT 编译

### 现状
- Polar adaptation 使用纯 Python 循环
- 每次迭代都有 autograd 开销
- SphericalSGD 步进可以编译优化

### 需求
对关键路径使用 `torch.jit.script` 或 `torch.compile`：
- `spherical_step` 函数
- `compute_adaptation_loss` 函数
- 可选：整个 `adapt_query_projection` 方法

### 验收标准
```python
# 基准测试
qttt_config = PolarQTTTConfig(num_steps=4)
qttt = PolarQTTT(qttt_config, hidden_dim=512, num_heads=8)

# 编译前
with timer():
    adapted_query = qttt.adapt_query_projection(...)
t_before = timer.elapsed

# 编译后 (使用 JIT)
with timer():
    adapted_query = qttt.adapt_query_projection_jit(...)
t_after = timer.elapsed

assert t_after < t_before * 0.7  # 至少 30% 加速
```

---

## 3. 批量处理优化

### 现状
- qTTT 逐 token 处理
- AttnRes 逐层处理
- 没有利用 batch 并行

### 需求
实现批量版本的关键函数：
- `adapt_query_projection_batch` - 同时处理多个 query
- `block_attn_res_batch` - 并行处理多个 block

### 验收标准
```python
# 单样本
for query in queries:
    result = adapt(query)  # 逐个处理

# 批量
results = adapt_batch(queries)  # 一起处理

# 批量应该更快
assert t_batch < t_single * 0.6  # 至少 40% 加速
```

---

## 任务拆分

### Story 1: RaBitQ Cache Implementation
- [ ] 完成 `_get_cached_rabitq_kv` 方法
- [ ] 在 `forward` 中集成缓存逻辑
- [ ] 添加缓存失效检测
- [ ] 性能测试验证

### Story 2: qTTT JIT Compilation
- [ ] 将 `spherical_step` 改为 JIT 版本
- [ ] 将 `compute_adaptation_loss` 改为 JIT 版本
- [ ] 创建 `PolarQTTTJIT` 类
- [ ] 性能对比测试

### Story 3: Batch Processing
- [ ] 实现 `adapt_query_projection_batch`
- [ ] 实现 `block_attn_res_batch`
- [ ] 更新 `AdaptiveTransformer` 支持 batch
- [ ] 端到端性能测试

---

## 技术约束
- PyTorch >= 2.0
- 保持 backward compatibility
- 支持 CPU 和 CUDA
