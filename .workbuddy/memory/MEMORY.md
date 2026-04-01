# MEMORY.md - Adaptive Deep Networks 长期记忆

## Code Review Fixes (2026-03-26)

Fixed 6 issues from comprehensive code review. All 53 tests pass after fixes.

1. **HIGH**: `AdaptiveInference.forward()` 调用 `adapt_query_projection()` 时缺失 `target_positions` 和 `distractor_positions` 参数 → 添加了 Optional 默认值参数
2. **MEDIUM**: `target_positions` 语义混淆：`qt Adapt()` 中是序列位置索引，`MarginMaximizationLoss` 中是 vocab token ID → adaptation.py 中重命名为 `seq_positions`，margin_loss.py 保持 `target_positions`
3. **LOW**: `hard_negative_weight` 在 MarginMaximizationLoss 中是死代码 → 当提供显式 distractors 时作为 margin 缩放因子使用
4. **LOW**: tokenizer.py 和 math_eval.py 中裸 `except:` → 替换为具体异常类型
5. **LOW**: `mlp_flops()` 中 SwiGLU 系数用 2（应为 3）→ 添加 `use_swiglu` 参数区分
6. **LOW**: `src/models/` 和 `src/benchmarks/` 缺少 `__init__.py` → 已创建

## User Preferences

- 搜索引擎偏好：使用 Bing 而非百度

## Project Conventions

- 测试命令: `pytest tests/ -v`
- Python 3.8.19 环境
- 使用 SwiGLU MLP（3 个投影）而非标准 MLP（2 个投影）
- qTTT 中 `seq_positions` = 序列位置索引，`target_token_ids` = vocab token ID
- KV Cache 压缩模块: **RaBitQ** (非 TurboQuant)，位于 `src/rabitq/`
  - 类名: `RaBitQ`, `RaBitQConfig`, `RaBitQCache`, `MSECompressor`
  - `MSECompressor.fit(sample, head_dim)` 和 `compress(x, head_dim)` 需要 `head_dim` 参数
  - 旧 TurboQuant 代码保留在 `src/turboquant/`、`scripts/legacy/`、`experiments/turboquant/` (不活跃)

## QTTT 论文引用审查 (2026-03-27)

arXiv:2512.13898 论文实际标题为 "Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs"（Bansal et al.），非我们论文写的 "Liu, J." / "QTTT: Query-Only Test-Time Training for Long-Context Retrieval"。

论文存在的概念：score dilution、logarithmic margin requirement、frozen KV cache、query-only update、FLOP equivalence T≈2Nk、12.6%/14.1% 提升。

论文**不存在**的概念（不应归因于 [4]）：gating mechanism、EMA threshold calibration、oracle recovery 82-89%、correlation r=0.42-0.84、reconstruction loss 作为 gating signal。详见 QTTT_CITATION_AUDIT.md。

## 全文引用审查修正完成 (2026-03-27)

共修正 17 处引用错误：
- P0 严重错误：[4] 作者/标题（Bansal非Liu）、[9]/[59] MoD 引用混淆、[14] 作者 Xiao,C→G、[35] 作者完全错误（Narayanan非Xiao）、[34] 作者错误（Liu,Z非Zhang,Z）、Positioning表格[9]→[59]
- P1 需修正：[6] NeurIPS 2022、[16] NeurIPS 2021、[18] NeurIPS 2024、[19] EMNLP 2023、[23] ICML 2023、[27] ACL 2020、[40] ICML 2016、[42] Makes→Making、[55] ICLR 2024
- 正文归因：删除4处虚构引用数据（correlation r=0.42-0.84、oracle recovery 82-89%），reconstruction loss 标注为 "inspired by [4]"，pseudo-queries 明确归属 AttnRes

## Auto-Research Skill 开发完成 (2026-03-27)

位置: `~/.workbuddy/skills/自动写论文/`（内部名: auto-research）

8 阶段研究流水线：选题初始化 → 文献检索 → 知识合成 → 实验设计 → 实验执行 → 分析决策 → 论文撰写 → 最终化导出。

新增文件：
- `llm_client.py` - LLM 调用封装（支持 OpenAI API 和 requests 回退）
- `prompt_loader.py` - Prompt 模板加载器
- 6 个 Prompt 模板: stage5_repair.md, stage7_outline.md, stage7_draft.md, stage7_review.md, stage7_revision.md, quality_gate.md
- `templates/neurips2025.tex` - NeurIPS 2025 LaTeX 论文模板
- `.env.example` - 环境变量配置示例

代码质量修复：
- experiment_run.py: 3 处裸 except → 替换为具体异常类型
- paper_writing.py: 8 处 `\\n` 硬编码转义 → 修正为 `\n`
- experiment_design.py: 2 处嵌入代码中的 `\\n` → 修正为 `\n`

LLM 集成：topic_init/synthesis/paper_writing 三个阶段支持 LLM 增强（use_llm=True），不可用时自动 fallback 到模板生成。
