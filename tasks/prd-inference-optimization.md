# PRD: 推理流程优化

## 1. 引言

### 1.1 目标
实现论文 §3.4 描述的完整推理 pipeline，包括 Ponder Gate 条件触发、增量 KV Cache 和动态 qTTT 配置。

### 1.2 背景
当前实现中 qTTT 无条件执行，导致计算资源浪费。根据论文，Ponder Gate 应在查询不确定性高时才触发 qTTT。

### 1.3 范围
- ✅ Ponder Gate 实现
- ✅ 增量 KV Cache 优化
- ✅ 动态 qTTT 配置
- ❌ 异步 qTTT（未来优化）

## 2. 用户故事

### US1: Ponder Gate 条件触发
**作为** 部署工程师，**我希望** qTTT 只在必要时执行，**从而** 节省计算资源。

**验收标准:**
- [ ] 实现 PonderGate 类，支持熵和最大概率判断
- [ ] 集成到 generate() 流程
- [ ] 可通过阈值参数调整敏感度
- [ ] 当 use_qttt='adaptive' 时启用

**预估时间:** 15分钟

### US2: 增量 KV Cache
**作为** 研究人员，**我希望** KV cache 增量更新而非重建，**从而** 加速长序列推理。

**验收标准:**
- [ ] 实现增量更新机制
- [ ] 仅处理新 token 的 KV
- [ ] 保持与现有 get_kv_cache() 兼容
- [ ] 长序列 (>1K) 速度提升 50%+

**预估时间:** 20分钟

### US3: 动态 qTTT 配置
**作为** 研究人员，**我希望** qTTT 步数根据上下文动态调整，**从而** 平衡质量和速度。

**验收标准:**
- [ ] 基于序列长度调整 num_steps
- [ ] 基于梯度大小调整 learning_rate
- [ ] 向后兼容固定配置

**预估时间:** 10分钟

### US4: 集成测试
**作为** 开发者，**我希望** 所有优化通过测试，**从而** 确保代码质量。

**验收标准:**
- [ ] 所有现有测试通过
- [ ] 新增 Ponder Gate 单元测试
- [ ] 新增增量 KV 性能测试
- [ ] 端到端推理测试

**预估时间:** 15分钟

## 3. 技术需求

### 3.1 Ponder Gate
```python
class PonderGate:
    def __init__(
        self,
        entropy_threshold: float = 2.0,
        min_prob_threshold: float = 0.3
    )
    
    def should_adapt(self, logits: torch.Tensor) -> bool:
        # 基于熵和最大概率判断
```

### 3.2 增量 KV Cache
```python
def update_kv_cache_incremental(
    self,
    new_token_id: torch.Tensor,
    layer_idx: int,
    existing_cache: KVCache
) -> KVCache:
    # 仅计算新 token 的 KV
```

### 3.3 动态配置
```python
@dataclass
class AdaptiveQTTTConfig:
    base_steps: int = 4
    max_steps: int = 16
    seq_len_thresholds: List[int] = field(
        default_factory=lambda: [128, 512, 1024]
    )
```

## 4. 非目标
- 异步 qTTT 执行
- 多线程优化
- 底层 C++ 实现

## 5. 设计考虑

### 5.1 API 兼容性
- 保持 `generate()` 签名不变
- `use_qttt` 参数扩展为 `Union[bool, str]`
- 新增配置通过 Optional 参数传递

### 5.2 性能考虑
- Ponder Gate 判断本身要轻量
- 增量更新避免重复计算
- 动态配置减少不必要的迭代

### 5.3 错误处理
- 增量更新失败时回退到全量重建
- Ponder Gate 异常时默认执行 qTTT

## 6. 成功指标
- Ponder Gate 减少 30%+ qTTT 调用
- 长序列推理速度提升 50%+
- 测试覆盖率保持 >80%
