# PRD: Adaptive Deep Networks 验证框架

## 1. 引言

### 1.1 目标
构建完整的验证框架，用于在 Lambda AI 上验证 Adaptive Deep Networks 论文的三个核心创新组件。

### 1.2 范围
- **包含**: AttnRes 实现、Gating 机制、qTTT 训练、基准测试、Lambda AI 部署
- **不包含**: 完整预训练（仅验证架构）、多模态扩展

### 1.3 参考文献
- 论文: Adaptive Deep Networks Final Draft
- 参考代码: Attention Residuals Technical Report (Chen et al., 2026)

---

## 2. 用户故事

### US-001: Block AttnRes 核心实现
**作为** 研究人员  
**我希望** 实现 Block Attention Residuals 机制  
**以便** 验证其在深度网络中的表示保留能力

**验收标准:**
- [ ] 实现 `block_attn_res` 函数，支持两阶段计算
- [ ] 伪查询向量零初始化
- [ ] 内存复杂度 O(Nd) 验证
- [ ] 单元测试覆盖率 >90%

**时间估算:** 5分钟

---

### US-002: 伪查询向量学习验证
**作为** 研究人员  
**我希望** 验证伪查询向量的学习动态  
**以便** 确认其从均匀分布到专业化的转变

**验收标准:**
- [ ] 监控注意力权重分布变化
- [ ] 验证零初始化行为
- [ ] 记录专业化模式
- [ ] 可视化注意力热图

**时间估算:** 3分钟

---

### US-003: Dynamic Gating 实现
**作为** 研究人员  
**我希望** 实现基于重建损失的动态门控  
**以便** 自适应分配计算资源

**验收标准:**
- [ ] 实现 `L_rec` 计算（TTT 重建损失）
- [ ] 实现 EMA 阈值校准
- [ ] 目标更新率维持（ρ_target）
- [ ] 门控决策准确率 >80%

**时间估算:** 4分钟

---

### US-004: qTTT 实现
**作为** 研究人员  
**我希望** 实现 Query-only Test-Time Training  
**以便** 在推理时进行适应性优化

**验收标准:**
- [ ] 实现 qTTT_adapt 函数
- [ ] 支持伪查询和查询投影两种目标
- [ ] 保持 KV Cache 冻结
- [ ] 边距最大化损失实现

**时间估算:** 5分钟

---

### US-005: Needle-in-Haystack 基准
**作为** 研究人员  
**我希望** 运行针在大海捞针测试  
**以便** 验证长上下文检索能力

**验收标准:**
- [ ] 支持 1K-256K 上下文长度
- [ ] 多深度位置测试（10 depths/length）
- [ ] 准确率报告（目标: 86.9% avg）
- [ ] 与基线对比

**时间估算:** 4分钟

---

### US-006: MATH 数据集验证
**作为** 研究人员  
**我希望** 在 MATH 数据集上评估  
**以便** 验证数学推理能力

**验收标准:**
- [ ] 支持 5 个难度级别
- [ ] 准确率报告（目标: 52.3% @ 7B）
- [ ] 与 CoT、Self-Consistency 对比
- [ ] 计算效率分析

**时间估算:** 4分钟

---

### US-007: FLOP 等效性验证
**作为** 研究人员  
**我希望** 验证宽度-深度 FLOP 等效性  
**以便** 确认 T_think ≈ 2 * N_qTTT * k

**验收标准:**
- [ ] 测量思考 token 生成成本
- [ ] 测量 qTTT 步骤成本
- [ ] 验证等效公式
- [ ] 生成对比报告

**时间估算:** 3分钟

---

### US-008: Lambda AI 部署脚本
**作为** 工程师  
**我希望** 获得 Lambda AI 部署脚本  
**以便** 快速启动验证实验

**验收标准:**
- [ ] 环境配置脚本
- [ ] 多 GPU 启动脚本
- [ ] 资源监控
- [ ] 实验调度脚本

**时间估算:** 4分钟

---

### US-009: 端到端集成测试
**作为** 研究人员  
**我希望** 运行端到端集成测试  
**以便** 验证所有组件协同工作

**验收标准:**
- [ ] 完整前向传播测试
- [ ] 梯度流动验证
- [ ] 内存使用监控
- [ ] 性能基准对比

**时间估算:** 5分钟

---

### US-010: 消融研究脚本
**作为** 研究人员  
**我希望** 运行消融研究  
**以便** 量化各组件贡献

**验收标准:**
- [ ] w/o qTTT 配置
- [ ] w/o Gating 配置
- [ ] w/o AttnRes 配置
- [ ] 组件贡献分析报告

**时间估算:** 4分钟

---

## 3. 技术设计

### 3.1 架构概览

```
src/
├── attnres/              # Block AttnRes 实现
│   ├── __init__.py
│   ├── block_attnres.py  # 核心实现
│   └── pseudo_query.py   # 伪查询管理
├── gating/               # 动态门控
│   ├── __init__.py
│   ├── reconstruction.py # 重建损失
│   └── threshold.py      # 阈值校准
├── qttt/                 # qTTT 实现
│   ├── __init__.py
│   ├── adaptation.py     # 适配循环
│   └── margin_loss.py    # 边距损失
├── models/               # 模型定义
│   ├── __init__.py
│   ├── adaptive_transformer.py
│   └── configs.py        # 模型配置
├── benchmarks/           # 基准测试
│   ├── needle_haystack.py
│   ├── math_eval.py
│   └── flop_analysis.py
└── utils/                # 工具函数
    ├── __init__.py
    ├── memory.py         # 内存监控
    └── distributed.py    # 分布式支持

tests/
├── unit/                 # 单元测试
├── integration/          # 集成测试
└── benchmarks/           # 基准测试

scripts/
├── lambda_setup.sh       # Lambda AI 设置
├── run_benchmarks.py     # 基准测试运行器
└── ablation_study.py     # 消融研究
```

### 3.2 核心接口

```python
# Block AttnRes
class BlockAttnRes(nn.Module):
    def forward(
        self, 
        blocks: List[Tensor],      # [N, B, T, D]
        partial_block: Tensor,     # [B, T, D]
        pseudo_query: Tensor       # [D]
    ) -> Tensor:                  # [B, T, D]

# Dynamic Gating
class DynamicGating(nn.Module):
    def forward(
        self,
        reconstruction_loss: float
    ) -> Tuple[bool, float]:      # (should_adapt, threshold)

# qTTT
class QueryOnlyTTT:
    def adapt(
        self,
        queries: Tensor,           # [B, k, D]
        kv_cache: KVCache,         # Frozen
        num_steps: int,
        learning_rate: float
    ) -> Tensor:                  # Adapted queries
```

### 3.3 关键算法

**Algorithm: Block AttnRes Forward**
```
Input: blocks [b₀, ..., bₙ₋₁], partial_block bₙⁱ⁻¹, layer i, block n, pseudo-query w

// Phase 1: Inter-block attention
V ← stack(blocks + [partial_block])  // [N+1, B, T, D]
K ← RMSNorm(V)
logits ← w · K                        // [N+1, B, T]
α ← softmax(logits, dim=0)
h ← Σᵢ αᵢ · Vᵢ                        // [B, T, D]

// Phase 2: Standard transformer + residual
attn_out ← Attention(LayerNorm(h))
partial_block ← partial_block + attn_out

return blocks, partial_block
```

**Algorithm: qTTT Adaptation**
```
Input: queries, kv_cache, w_l, num_steps, learning_rate

w_adapted ← w_l.clone().detach().requires_grad_(True)

for step in range(num_steps):
    attn_out ← compute_attention(queries, kv_cache, w_adapted)
    logits ← project_to_vocab(attn_out)
    
    // Margin maximization
    target_logits ← logits[:, :, target_positions]
    max_distractor ← logits[:, :, distractor_positions].max(dim=-1)
    margin_loss ← -logsigmoid(target_logits - max_distractor).mean()
    
    // Gradient step on w_adapted only
    grad ← autograd.grad(margin_loss, w_adapted)[0]
    w_adapted ← w_adapted - learning_rate * grad
    w_adapted ← w_adapted.detach().requires_grad_(True)

return w_adapted.detach()
```

### 3.4 模型配置

| 参数 | Small (1.5B) | Medium (7B) | Large (50B) |
|-----|-------------|-------------|-------------|
| Layers | 32 | 32 | 64 |
| Hidden Dim | 2048 | 4096 | 5120 |
| MLP Ratio | 4 | 4 | 4 |
| Heads | 32 | 32 | 40 |
| Blocks (N) | 8 | 8 | 16 |
| qTTT Steps (max) | 16 | 32 | 32 |
| qTTT Span (k) | 128 | 128 | 256 |

---

## 4. 非目标

- 完整模型预训练
- 生产级推理优化
- 多模态扩展
- 跨平台支持（仅 Lambda AI）

---

## 5. 成功指标

| 指标 | 目标 | 测量方式 |
|-----|------|---------|
| 代码覆盖率 | >80% | pytest-cov |
| NIH 准确率 | 86.9% | 自动化测试 |
| MATH 准确率 | 52.3% | 自动化测试 |
| 门控决策准确率 | >80% | 与 oracle 对比 |
| 文档完整性 | 100% | 检查清单 |

---

## 6. 依赖关系

```
US-001 (AttnRes) 
    ├── US-002 (伪查询学习)
    ├── US-005 (NIH 基准)
    └── US-009 (集成测试)

US-003 (Gating)
    └── US-006 (MATH 验证)

US-004 (qTTT)
    ├── US-005 (NIH 基准)
    ├── US-006 (MATH 验证)
    └── US-007 (FLOP 验证)

US-008 (部署) → 所有其他 US
US-010 (消融) → 所有核心组件
```

---

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|-----|------|------|---------|
| 内存不足 | 中 | 高 | 使用 gradient checkpointing, 减小 batch size |
| 训练不稳定 | 低 | 高 | 零初始化， careful learning rate |
| Lambda GPU 不可用 | 中 | 高 | 支持本地测试模式 |
| 性能不达预期 | 中 | 中 | 详细日志，可调参数 |

---

## 8. 附录

### 8.1 参考实现

基于 Attention Residuals Technical Report (Page 5):

```python
def block_attn_res(
    blocks: list[Tensor], 
    partial_block: Tensor, 
    proj: Linear, 
    norm: RMSNorm
) -> Tensor:
    """Inter-block attention: attend over block reps + partial sum."""
    V = torch.stack(blocks + [partial_block])  # [N+1, B, T, D]
    K = norm(V)
    logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
    h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
    return h
```

### 8.2 环境要求

```
Python 3.10+
PyTorch 2.1+
CUDA 12.1+
FlashAttention 2
Triton (for custom kernels)
```
