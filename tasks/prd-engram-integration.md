# PRD: Engram Integration for Adaptive Deep Networks

## 1. 项目概述

### 1.1 背景
Engram 是 DeepSeek 提出的一种显式 N-gram 记忆机制，通过在特定 Transformer 层引入基于哈希的 n-gram 嵌入来增强模型的长距离依赖建模能力。本 PRD 定义将 Engram 集成到 ADN 框架的实现计划。

### 1.2 目标
- 实现可复用的 Engram 模块
- 集成到 AdaptiveTransformer 架构
- 验证在 Needle-in-Haystack 等长文本任务上的性能提升
- 保持与现有 ADN 组件 (AttnRes, qTTT, Gating) 的兼容性

### 1.3 非目标
- 不实现完整的 DeepSeek-V3 架构
- 不修改现有的 TurboQuant 和 RaBitQ 组件
- 不针对特定下游任务微调

---

## 2. 用户故事

### US1: 作为研究员，我希望配置 Engram 参数
**描述**: 用户可以通过 ModelConfig 启用和配置 Engram 模块

**验收标准**:
- [ ] 可以在 ModelConfig 中启用/禁用 Engram
- [ ] 可以配置 n-gram 大小 (max_ngram_size)
- [ ] 可以配置嵌入维度 (n_embed_per_ngram)
- [ ] 可以配置头数 (n_head_per_ngram)
- [ ] 可以指定应用 Engram 的层 (layer_ids)

**估算**: 2 小时

---

### US2: 作为开发者，我需要一个可用的 Engram 模块
**描述**: 实现 Engram 核心组件，包括 N-gram 哈希映射、多头嵌入和短卷积

**验收标准**:
- [ ] 实现 CompressedTokenizer 压缩词汇表
- [ ] 实现 NgramHashMapping 计算层特定的 n-gram 哈希
- [ ] 实现 MultiHeadEmbedding 支持多头的嵌入查找
- [ ] 实现 ShortConv 短卷积处理局部依赖
- [ ] 实现 Engram 主模块，整合上述组件

**估算**: 4 小时

---

### US3: 作为开发者，我需要 Engram 集成到 AdaptiveTransformer
**描述**: 将 Engram 模块集成到现有的 Transformer 层中

**验收标准**:
- [ ] 修改 AdaptiveTransformer 支持 Engram 层
- [ ] 保持与现有 hyper-connection (hc_mult) 机制的兼容
- [ ] 确保梯度正确传播
- [ ] 支持 checkpoint 保存/加载

**估算**: 3 小时

---

### US4: 作为 QA，我需要单元测试验证 Engram 功能
**描述**: 为 Engram 组件编写全面的单元测试

**验收标准**:
- [ ] CompressedTokenizer 测试 (压缩率、逆映射)
- [ ] NgramHashMapping 测试 (哈希冲突率、层特定性)
- [ ] MultiHeadEmbedding 测试 (嵌入查找正确性)
- [ ] ShortConv 测试 (卷积计算正确性)
- [ ] Engram 模块集成测试 (前向传播、梯度)

**估算**: 3 小时

---

### US5: 作为研究员，我需要验证 Engram 的性能提升
**描述**: 运行基准测试验证 Engram 在 ADN 中的效果

**验收标准**:
- [ ] 运行 Needle-in-Haystack 基准 (256K 上下文)
- [ ] 对比基线 ADN 和 ADN+Engram 的性能
- [ ] 测量内存和计算开销
- [ ] 生成性能报告

**估算**: 4 小时

---

### US6: 作为开发者，我需要消融实验分析
**描述**: 分析不同 Engram 配置对性能的影响

**验收标准**:
- [ ] 测试不同 n-gram 大小 (2, 3, 4)
- [ ] 测试不同嵌入维度 (256, 512, 1024)
- [ ] 测试不同层位置 (early, middle, late)
- [ ] 分析性能与计算开销的 trade-off

**估算**: 3 小时

---

## 3. 技术规格

### 3.1 核心组件

#### CompressedTokenizer
```python
class CompressedTokenizer:
    """压缩词汇表，合并语义相同的 token"""
    def __init__(self, tokenizer_name_or_path: str)
    def __call__(self, input_ids: np.ndarray) -> np.ndarray
```

**关键逻辑**:
- 使用 NFKC + NFD + 去除重音符号 + 小写化归一化
- 将 Unicode 变体合并为统一表示
- 保持可逆映射用于调试

#### NgramHashMapping
```python
class NgramHashMapping:
    """计算层特定的 n-gram 哈希"""
    def __init__(
        self,
        engram_vocab_size: List[int],
        max_ngram_size: int,
        n_embed_per_ngram: int,
        n_head_per_ngram: int,
        layer_ids: List[int],
        tokenizer,
        pad_id: int,
        seed: int,
    )
    def hash(self, input_ids: np.ndarray) -> Dict[int, np.ndarray]
```

**关键逻辑**:
- 每层使用不同的随机种子生成乘数
- 使用 XOR 混合多个 token 的哈希
- 对不同 head 使用不同的质数取模

#### MultiHeadEmbedding
```python
class MultiHeadEmbedding(nn.Module):
    """多头的嵌入层"""
    def __init__(self, list_of_N: List[int], D: int)
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor
```

#### ShortConv
```python
class ShortConv(nn.Module):
    """短卷积处理局部依赖"""
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        hc_mult: int = 4,
    )
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

#### Engram Module
```python
class Engram(nn.Module):
    """Engram 主模块"""
    def __init__(self, layer_id: int, config: EngramConfig)
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, L, hc_mult, D]
        input_ids: torch.Tensor,       # [B, L]
    ) -> torch.Tensor  # [B, L, hc_mult, D]
```

### 3.2 集成点

```python
# 在 AdaptiveTransformer 层中
class TransformerBlockWithEngram(nn.Module):
    def __init__(self, layer_id, config):
        # ... existing components
        self.engram = None
        if layer_id in config.engram_layer_ids:
            self.engram = Engram(layer_id, config.engram_config)
    
    def forward(self, hidden_states, input_ids, ...):
        # Apply Engram before attention
        if self.engram is not None:
            hidden_states = self.engram(hidden_states, input_ids) + hidden_states
        
        # ... rest of the layer
```

### 3.3 配置参数

```python
@dataclass
class EngramConfig:
    enabled: bool = False
    engram_vocab_size: List[int] = field(default_factory=lambda: [100000, 100000])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
```

---

## 4. 依赖关系

### 4.1 外部依赖
- `torch` >= 2.1.0
- `numpy` >= 1.24.0
- `sympy` (for prime number generation)
- `transformers` (for tokenizer)

### 4.2 内部依赖
- `src.models.configs.ModelConfig`
- `src.models.adaptive_transformer.AdaptiveTransformer`
- 现有的 tokenizer 基础设施

---

## 5. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 哈希冲突导致性能下降 | 中 | 高 | 使用多个 head 和质数取模分散风险 |
| 内存开销过大 | 中 | 中 | 提供可调参数，支持梯度检查点 |
| 与 AttnRes 的梯度冲突 | 低 | 高 | careful 初始化，分离实验验证 |
| Tokenizer 不兼容 | 低 | 高 | 使用压缩 tokenizer 适配现有 tokenizer |

---

## 6. 验收标准总结

### 功能验收
- [ ] 所有单元测试通过
- [ ] 集成测试通过
- [ ] 训练不崩溃
- [ ] Checkpoint 可正常保存/加载

### 性能验收
- [ ] Needle-in-Haystack 256K: > 88% (基线 86.9%)
- [ ] 推理速度下降 < 15%
- [ ] 内存开销 < 20%

### 代码质量
- [ ] 代码覆盖率 > 80%
- [ ] 通过 black, flake8, mypy 检查
- [ ] AGENTS.md 更新

---

*PRD Version: 1.0*
*Created: 2026-04-05*
*Framework: Superpowers Skill*
