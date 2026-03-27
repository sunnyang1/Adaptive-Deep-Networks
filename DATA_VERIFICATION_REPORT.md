# Adaptive Deep Networks 数据验证报告

**生成日期**: 2026-03-27  
**目的**: 标注论文中所有估算数据的验证状态，对比已知的公开数据

---

## 一、数据验证总览

### 验证标准
- ✅ **已验证**: 与公开文献/排行榜数据一致
- ⚠️ **需标注**: 论文数据为估算值，与已知数据存在偏差，需在论文中标注 `*`
- ❌ **不可验证**: 引用的论文尚未发表或数据不可查

### 数据分布统计
| 状态 | 数量 | 说明 |
|------|------|------|
| ✅ 已验证 | 8 | 与公开数据一致 |
| ⚠️ 需标注 | 35+ | 本项目自身实验数据，未经独立验证 |
| ❌ 不可验证 | 5 | 引用论文数据（QTTT, Attention Residuals）不可查 |

---

## 二、关键数据逐项验证

### 2.1 摘要中的核心声明

| 论文声明 | 验证状态 | 已知公开数据 | 说明 |
|----------|----------|-------------|------|
| GPT-4 NIH 平均 82.3% | ⚠️* | 无统一的 NIH 标准基准；Kamradt 测试为非正式评估 | GPT-4 在 128K 上下文中确实会出现检索失败，但具体百分比因测试方法而异 |
| 86.9% NIH (8.7B) | ⚠️* | 无同条件可比数据 | 本项目自身实验数据 |
| 89.4% NIH (2.2B) | ⚠️* | 无同条件可比数据 | 本项目自身实验数据 |
| 52.3% MATH (8.7B) | ⚠️* | GPT-4 base: **42.5%** (已验证) | 论文声明远超 GPT-4 MATH 基线，需实验验证 |
| 56.1% MATH (2.2B) | ⚠️* | 同上 | 小模型声称超越大模型，需特别验证 |
| 40% compute reduction | ⚠️* | MoD 论文报告 30-50% FLOP 缩减 | 需独立 FLOP 测量验证 |

### 2.2 Table 1: NIH 准确率（各上下文长度）

**论文中所有 NIH 数据均为估算值，标记为 ⚠️***

| 数据点 | 论文值 | 验证状态 | 说明 |
|--------|--------|----------|------|
| Transformer (8.7B) 全行 | 38.2% avg | ⚠️* | 基线数据为估算 |
| DeepNorm (8.7B) 全行 | 41.2% avg | ⚠️* | 基线数据为估算 |
| MoD (8.7B) 全行 | 41.6% avg | ⚠️* | 基线数据为估算 |
| TTT-Linear (8.7B) 全行 | 62.3% avg | ⚠️* | 基线数据为估算 |
| AttnRes (8.7B) 全行 | 69.9% avg | ⚠️* | 本项目数据 |
| **AttnRes + qTTT (8.7B)** | **86.9% avg** | ⚠️* | **核心声明，需重点验证** |
| GPT-4 (API) 全行 | 82.3% avg | ⚠️* | 非标准测试，数值为估算 |
| Claude-3 (API) 全行 | 79.8% avg | ⚠️* | 非标准测试，数值为估算 |

### 2.3 Table 2: LongBench-v2 性能

| 数据点 | 论文值 | 验证状态 | 已知公开数据 |
|--------|--------|----------|-------------|
| Transformer avg | 39.7% | ⚠️* | LLaMA-2 7B 级模型在 LongBench-v2 上约 35-40% |
| TTT-Linear avg | 45.9% | ⚠️* | 不可独立验证 |
| AttnRes avg | 50.1% | ⚠️* | Qwen3.5-4B (2026) 在 LongBench-v2 上约 50.0% |
| **AttnRes + qTTT avg** | **56.8%** | ⚠️* | **核心声明，需重点验证** |

**注**: LongBench-v2 排行榜（2026-03-27）显示：
- Qwen3.5-4B: 50.0%
- Qwen3.5-9B: 55.2%
- o1-preview: 57.7%
- 人类基线: 53.7%

论文中 8.7B 模型的 56.8% 与 Qwen3.5-9B (55.2%) 和 o1-preview (57.7%) 相当，这是一个很强的声明，**必须用独立实验验证**。

### 2.4 Table 3: MATH 数据集性能

| 数据点 | 论文值 | 验证状态 | 已知公开数据 | 偏差分析 |
|--------|--------|----------|-------------|----------|
| Transformer | 35.2% | ⚠️* | LLaMA-2 7B: ~14% (with CoT); Qwen-7B: ~24% | ⚠️ 论文基线偏高 |
| CoT (5 samples) | 41.5% | ⚠️* | 合理范围 | — |
| Self-Consistency (10) | 44.8% | ⚠️* | 合理范围 | — |
| TTT-Linear | 48.9% | ⚠️* | 不可独立验证 | — |
| **AttnRes + qTTT (gated)** | **52.3%** | ⚠️* | **GPT-4 base: 42.5%** (已验证) | ⚠️ 声称超越 GPT-4 约 10 个点 |
| **AttnRes + qTTT (max)** | **58.9%** | ⚠️* | GPT-4 + CoT: ~52%; GPT-4 + Code: 84.3% | ⚠️ 超越 GPT-4+CoT |
| **2.2B Small: 56.1%** | **56.1%** | ⚠️* | GPT-4 base: 42.5% | ⚠️ 2.2B 模型声称超越 GPT-4，需特别验证 |

**GPT-4 MATH 分级数据（已验证，来自 OpenAI 技术报告）**:
| Level | GPT-4 (Base) | 论文 AttnRes+qTTT (8.7B) | 论文 AttnRes+qTTT (2.2B) |
|-------|-------------|--------------------------|--------------------------|
| Level 1 | 69.1% | 76.2% ⚠️* | 76.3% ⚠️* |
| Level 2 | 45.8% | 66.8% ⚠️* | 66.5% ⚠️* |
| Level 3 | 35.2% | 56.4% ⚠️* | 56.6% ⚠️* |
| Level 4 | 29.6% | 46.2% ⚠️* | 46.1% ⚠️* |
| Level 5 | 24.3% | 34.5% ⚠️* | 34.9% ⚠️* |
| Overall | **42.5%** ✅ | **52.3%** ⚠️* | **56.1%** ⚠️* |

### 2.5 Table 3b & 4: GSM8K 结果

| 数据点 | 论文值 | 验证状态 | 已知公开数据 |
|--------|--------|----------|-------------|
| AttnRes + qTTT (Small) | 81.5% | ⚠️* | LLaMA-2 7B: ~46.7%; LLaMA-3 8B: ~68.4% (with CoT) |
| AttnRes + qTTT (Medium) | 81.4% | ⚠️* | 8.7B 模型达到 81.4% 超过 LLaMA-3 8B，需验证 |
| Transformer baseline | 74.2% | ⚠️* | 对标准 8.7B 模型偏高 |
| CoT baseline | 79.5% | ⚠️* | 合理范围 |

**GPT-4 GSM8K（已验证）**: GPT-4 在 GSM8K 上约 **92.0%**（来自 GPT-4 技术报告及多个独立验证）

### 2.6 Table 5: 计算效率分析

| 数据点 | 论文值 | 验证状态 | 说明 |
|--------|--------|----------|------|
| 40% compute reduction | — | ⚠️* | 声称需独立 FLOP 测量 |
| Oracle recovery 82% | 82% | ⚠️* | 合理范围 |
| Gating 1.28× FLOP | — | ⚠️* | 需实际 FLOP 测量 |

### 2.7 Table 6: 消融实验

| 配置 | 论文值 | 验证状态 | 说明 |
|------|--------|----------|------|
| Full System | 56.8 | ⚠️* | 全部为本项目实验数据 |
| w/o qTTT | 50.1 | ⚠️* | — |
| w/o Gating | 53.2 | ⚠️* | — |
| w/o AttnRes | 48.9 | ⚠️* | — |
| Standard Transformer | 39.7 | ⚠️* | — |

### 2.8 引用的外部论文数据（不可验证）

| 引用 | 数据 | 状态 | 说明 |
|------|------|------|------|
| [4] QTTT arXiv:2512.13898 | 多处引用 | ⚠️ 已找到 (03-27) | 论文存在，标题为 "Let's (not) just put things in Context: TTT for Long-Context LLMs" (Bansal et al.)。方法与论文描述不同——通过梯度更新做 TTT 而非纯 query-only。Qwen3-4B 在 LongBench-v2/ZeroScrolls 上分别提升 12.6/14.1 个百分点 |
| [58] Attention Residuals arXiv:2603.15031 | 多处引用 | ❌ | 该论文标注为 2026 年 3 月，为未来日期，可能尚未正式发表 |
| [4] Oracle recovery 82-89% | 82-89% | ❌ | 引用自不可验证的 QTTT 论文 |
| [4] Correlation r = 0.42 to 0.84 | — | ❌ | 同上 |

---

## 三、已验证的公开数据参考

### GPT-4 基准数据（来源: OpenAI GPT-4 Technical Report, 2023）

| Benchmark | GPT-4 (Base) | GPT-4 (Code Interpreter) |
|-----------|-------------|-------------------------|
| MATH (Overall) | **42.5%** ✅ | ~84.3% ✅ |
| MATH Level 1 | 69.1% ✅ | — |
| MATH Level 2 | 45.8% ✅ | — |
| MATH Level 3 | 35.2% ✅ | — |
| MATH Level 4 | 29.6% ✅ | — |
| MATH Level 5 | 24.3% ✅ | — |
| GSM8K | **~92.0%** ✅ | — |
| MMLU | **86.4%** ✅ | — |

### Claude 系列模型 GSM8K 数据（来源: llm-stats.com, 2026-03-27）

| Model | GSM8K Score |
|-------|------------|
| Claude 3 Opus | 95.0% |
| Claude 3.5 Sonnet | 96.4% |
| Claude 3 Sonnet | 92.3% |
| Claude 3 Haiku | 88.9% |

### LongBench-v2 排行榜数据（来源: llm-stats.com, 2026-03-27）

| Model | Score | Params |
|-------|-------|--------|
| Qwen3.5-397B-A17B | 63.2% | 397B |
| Qwen3.5-27B | 60.6% | 27B |
| Qwen3.5-9B | 55.2% | 9B |
| Qwen3.5-4B | 50.0% | 4B |
| o1-preview | 57.7% | — |
| Human baseline | 53.7% | — |

### Mixture of Depths（来源: Raposo et al., arXiv:2404.02258）

- FLOP 缩减: 30-50% per forward pass
- 性能: Match baseline performance at equivalent FLOPs
- 未提供在 MATH 或 LongBench 上的具体数字

---

## 四、关键发现与风险提示

### 🔴 高风险声明（需最优先验证）

1. **2.2B 模型 MATH 56.1%**：声称 2.2B 参数模型在 MATH 上超越 GPT-4 (42.5%) 达 13.6 个百分点。目前公开记录中，即使 70B 级别开源模型也难以在 MATH 上达到 56%。**这是论文中最强的声明，必须用独立实验验证。**

2. **2.2B 模型 NIH 89.4% > GPT-4 82.3%**：声称 2.2B 模型在长上下文检索上超越 GPT-4。虽然 AttnRes 专为长上下文设计，但 450× 参数差距的超越需要验证。

3. **8.7B 模型 LongBench-v2 56.8%**：接近 o1-preview (57.7%)，远超 Qwen3.5-9B (55.2%)。对于非推理模型来说，这个数字偏高。

### 🟡 中等风险声明

4. **AttnRes <2% 推理开销**：需要实际推理延迟测量
5. **82% oracle recovery**：gating 机制的有效性需要验证
6. **FLOP equivalence ratio = 1.000**：理论推导已通过本地验证（Table A4），但实际运行时可能有偏差

### 🟢 低风险 / 合理声明

7. **GPT-4 MATH 42.5%**：与公开数据一致 ✅
8. **论文的理论公式**：FLOP 等价关系 $T_{think} \approx 2 N_{qTTT} k$ 的数学推导正确
9. **数据集可用性**：所有 8 个评估数据集已验证可用

---

## 五、论文修改建议

### 5.1 需要标注 `*` 的数据

论文中以下类别的数据应标注 `*`（表示待验证）：

1. **所有 Table 1-11 中非公开来源的数字**
2. **摘要和引言中的性能声明数字**
3. **消融实验中的所有数值**
4. **效率分析中的 FLOP 数字**

### 5.2 建议添加脚注

在论文中添加统一脚注说明：

> * denotes results estimated from local validation experiments. Independent replication on standardized benchmarks is recommended. See Data Verification Report for comparison with known public results.

### 5.3 建议修改的声明

| 原声明 | 建议修改 | 原因 |
|--------|---------|------|
| "matching 50B static baselines" (摘要) | "matching or surpassing GPT-4 base (42.5%)" | 更准确地反映对比基线 |
| GPT-4 MATH 标注为 "—" | 添加 GPT-4 MATH = 42.5% (已知) | 提供有意义的对比 |
| GPT-4 GSM8K 标注为 "—" | 添加 GPT-4 GSM8K ≈ 92.0% (已知) | 提供有意义的对比 |

---

## 六、参考文献可查性

| 引用 | 可查性 | 说明 |
|------|--------|------|
| [1] Vaswani et al. 2017 | ✅ | 经典论文 |
| [3] Sun et al. TTT ICML 2020 | ✅ | 已发表 |
| [4] Bansal et al. QTTT arXiv:2512.13898 | ⚠️ 已找到 (03-27) | 论文存在，标题为 "Let's (not) just put things in Context: TTT for Long-Context LLMs"。注意：方法与论文描述的 "query-only TTT" 不同，实际是通过梯度更新做 TTT。作者列表也不包含 "Liu" |
| [9] Riquelme et al. MoD NeurIPS 2021 | ✅ | 实际是 Vision MoE，非 Language MoD |
| [12] LayerSkip arXiv:2024 | ✅ | 可查 |
| [23] DeepNet/DeepNorm arXiv:2022 | ✅ | 可查 |
| [30] Kamradt NIAH GitHub | ✅ | 可查，但非正式论文 |
| [58] Chen et al. AttnRes arXiv:2603.15031 | ❌ | **日期为 2026 年 3 月**，可能尚未发表 |

---

*本报告由 AI 助手基于公开网络数据生成，建议人工复核所有引用。*
