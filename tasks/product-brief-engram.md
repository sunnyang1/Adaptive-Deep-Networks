# Product Brief: Engram Integration

## 战略愿景

将 DeepSeek 的 Engram 架构集成到 Adaptive Deep Networks (ADN) 中，通过显式的 N-gram 记忆机制增强 Transformer 的长距离依赖建模能力，验证其在表示埋葬缓解和推理效率方面的性能提升。

## 目标用户

- ADN 框架的研究人员和开发者
- 需要改进长文本建模能力的 LLM 从业者
- 对显式记忆机制感兴趣的研究者

## 成功指标

### 技术指标
1. **功能完整性**: Engram 组件成功集成到 ADN 的 AdaptiveTransformer
2. **性能提升**: 在 Needle-in-Haystack 任务上相比基线 ADN 有 measurable 提升
3. **训练稳定性**: 集成后不出现训练崩溃或梯度异常
4. **计算效率**: 推理开销 < 15%

### 验证指标
- Needle-in-Haystack 256K: 基线 86.9% → 目标 > 88%
- 训练收敛性: 与基线相当或更快
- 内存开销: < 20%

## 约束条件

### 技术约束
- 必须兼容现有的 ADN 架构 (AttnRes, qTTT, Gating)
- 保持 PyTorch 2.1+ 兼容性
- 支持单卡 A100 80GB 训练

### 实现约束
- 使用 TDD 方法开发
- 所有代码必须有单元测试覆盖
- 遵循现有代码风格 (black, flake8)

## 关键设计决策

### 架构决策
1. **集成层**: 在特定 Transformer 层 (如第1层和第15层) 插入 Engram 模块
2. **Hyper-connection 兼容**: 适配现有的 hc_mult (hyper-connection multiplier) 机制
3. **Tokenizer 复用**: 使用 ADN 现有的 tokenizer，实现 CompressedTokenizer 适配

### 实现决策
1. **模块化设计**: Engram 作为独立模块，可插拔
2. **配置驱动**: 通过 ModelConfig 控制 Engram 参数
3. **渐进式集成**: 先在小模型上验证，再扩展到大模型

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Engram 与 AttnRes 冲突 | 高 | 分离实验，逐步集成 |
| 内存开销过大 | 中 | 可调参数控制，梯度检查点 |
| 训练不稳定 | 高 | 小学习率预热， careful 初始化 |
| 性能未提升 | 中 | 消融实验，hyper-parameter 搜索 |

## 时间线

- **Phase 1** (Day 1): 核心组件实现 (NgramHashMapping, MultiHeadEmbedding, ShortConv)
- **Phase 2** (Day 2): 集成到 AdaptiveTransformer，单元测试
- **Phase 3** (Day 3): 性能验证实验，消融研究
- **Phase 4** (Day 4): 文档完善，对抗性审查

---

*Created: 2026-04-05*
*Framework: Superpowers Skill*
