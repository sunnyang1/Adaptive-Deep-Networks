# QTTT 引用对照审查报告

**日期**: 2026-03-27  
**对照原文**: Bansal et al., "Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs", arXiv:2512.13898, 2025  
**我们论文引用**: `[4] Liu, J., et al. "QTTT: Query-Only Test-Time Training for Long-Context Retrieval." arXiv:2512.13898, 2025.`

---

## 一、引用元数据错误（必须修正）

| 字段 | 我们论文写的 | QTTT 原文实际 | 状态 |
|------|-------------|--------------|------|
| **作者** | Liu, J., et al. | Rachit Bansal, Aston Zhang, Rishabh Tiwari 等 11 人 | ❌ **完全错误** |
| **标题** | "QTTT: Query-Only Test-Time Training for Long-Context Retrieval" | "Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs" | ❌ **完全错误** |
| **arXiv ID** | 2512.13898 | 2512.13898 | ✅ 正确 |
| **作者归属** | "Liu et al. [4]" (正文第63行) | 第一作者是 Bansal | ❌ **错误** |

**修正建议**: 
```
[4] Bansal, R., Zhang, A., Tiwari, R., et al. "Let's (not) just put things in Context: 
    Test-Time Training for Long-Context LLMs." arXiv:2512.13898, 2025.
```
正文中 `Liu et al. [4]` → `Bansal et al. [4]`

---

## 二、逐条引用准确性审查

### ✅ 正确引用（与原文一致）

#### 2.1 Score Dilution 概念（第13行）
> 我们论文: "attention score dilution problem...making precise retrieval impossible regardless of model capacity [4]"

**原文确认**: ✅ 论文确实提出了 "score dilution" 概念："we attribute these failures to score dilution, a phenomenon inherent to static self-attention"

#### 2.2 Score Dilution 数学公式（第120行）
> 我们论文: "when at least $m$ distractor keys satisfy $z_{i,j} \geq z_{i,j^*} - \Delta$ for margin $\Delta \geq 0$, the attention mass on the true target is bounded above by $1/(1 + me^{-\Delta})$ [4]"

**原文确认**: ✅ Lemma 2.2 完全一致："Lemma 2.2 (Score dilution). If at least m distractor keys satisfy $z \geq z_{j^*} - \Delta$ for some $\Delta \geq 0$, then... $\alpha \leq 1/(1+me^{-\Delta})$"

#### 2.3 Logarithmic Margin Requirement（第63行）
> 我们论文: "achieving reliable retrieval requires logarithmic margin growth with sequence length"

**原文确认**: ✅ Lemma 2.3 (Logarithmic margin requirement) 完全一致："guaranteeing a fixed target mass against worst-case distractors requires a gap that scales as $\Omega(\log T)$"

#### 2.4 qTTT 方法描述 — Frozen KV Cache（第25行）
> 我们论文: "gradient-based adaptation of attention query parameters with frozen key-value caches"

**原文确认**: ✅ Algorithm 1 Step 2 和 5 确认：先缓存 {K,V}，后续计算使用 "frozen {K(ℓ), V(ℓ)}"

#### 2.5 qTTT 方法描述 — Query Projection 更新（第25行）
> 我们论文: "adaptation of attention query parameters (pseudo-queries or query projections)"

**原文确认**: ⚠️ 部分正确。原文更新的是 query projection matrices {W_Q^(ℓ)}，**不是** pseudo-queries。我们论文中提到的 "pseudo-queries" 是我们自己 AttnRes 组件的概念，不应归因于 [4]。

#### 2.6 FLOP 等价公式（第363-374行）
> 我们论文: "$T_{\text{think}} \approx 2 N_{\text{qTTT}} k$" 

**原文确认**: ✅ 原文 Eq.(3.2) 完全一致："$T_{\text{think}} \approx 2N_{qTTT}k$ (long $T$, span $k \ll T$)"

#### 2.7 FLOP 等价验证（第374行）
> 我们论文: "generating $T_{\text{think}}=8192$ tokens equates to $N_{\text{qTTT}}=16$ steps with $k=256$, or $N_{\text{qTTT}}=32$ steps with $k=128$ [4]"

**原文确认**: ✅ 原文使用 $(k, N)=(128, 32)$ 作为默认设置，与计算一致。8192 ≈ 2×32×128 = 8192 ✅

#### 2.8 Thinking Tokens 不能修复 Dilution（第401行）
> 我们论文: "thinking tokens cannot repair missing evidence access—their attention mass on needles is bounded by the same dilution affecting original queries. qTTT explicitly reshapes queries to maximize margins, directly counteracting dilution [4]"

**原文确认**: ✅ Proposition 2.4 和 Corollary 2.5 完全支持此说法："Any generated token can carry is at most its own attention mass on the needle"

#### 2.9 12.6% 和 14.1% 提升（第412行）
> 我们论文: "qTTT outperforms FLOP-matched thinking-token baselines by 12.6% and 14.1% for 4B-parameter models"

**原文确认**: ✅ 摘要和正文均确认："qTTT leads to massive 12.6% and 14.1% points improvements for Qwen3-4B on average across subsets of LongBench-v2 and ZeroScrolls benchmarks"

---

### ❌ 错误引用（原文不存在或描述不准确）

#### 3.1 TTT Reconstruction Loss 公式归因（第300-302行）
> 我们论文: "The TTT reconstruction loss $\mathcal{L}_{\text{rec}}$ serves as the core gating signal. Computed from a dedicated test-time training layer using frozen key-value caches from initial prefill, this loss provides immediate difficulty assessment without additional forward passes [4]"
> 
> 公式: $\mathcal{L}_{\text{TTT}}(\theta; x_s) = -\sum_{i=t}^{t+k-1} \log p_\theta(x_{i+1} | x_{1:i}; \{K^{(\ell)}, V^{(\ell)}\})$

**原文实际情况**: 
- ✅ 公式本身与原文 Eq.(3.1) 一致：$L_{TTT}(\theta;x_s) = -\sum \log p_\theta(x_{i+1}|x_{1:i};\{K^{(\ell)},V^{(\ell)}\})$
- ❌ **"reconstruction loss" 的命名** — 原文称之为 "next-token prediction loss" 或 "$L_{TTT}$"，**从未**称之为 "reconstruction loss"
- ❌ **"gating signal" 的归因** — 原文 QTTT 论文中 $L_{TTT}$ 用于**直接优化 query**，不是作为 gating signal。原文中 **没有 gating mechanism**
- ❌ **"dedicated test-time training layer"** — 原文是在所有层同时更新 W_Q，不是用专门的 TTT 层

**严重性**: 🔴 **高** — 这是我们论文 Dynamic Gating 组件的核心设计，但被错误地归因于 QTTT 原文

#### 3.2 Correlation r = 0.42 to 0.84（第304行）
> 我们论文: "Correlation with oracle advantage ranges from **$r = 0.42$ to $0.84$** depending on setting [4]"

**原文实际情况**: ❌ **原文中完全不存在此数据**。搜索全文未找到 "correlation"、"r = 0.42"、"r = 0.84" 等内容。QTTT 论文中**没有 oracle correlation 实验**。

**严重性**: 🔴 **高** — 虚构引用数据

#### 3.3 EMA Threshold Calibration（第329-331行）
> 我们论文: "$\tau_{t+1} = \beta \tau_t + (1-\beta) \cdot \text{percentile}(\mathcal{L}_{\text{rec}}^{(t)}, p_{\text{target}})$ with $\beta \in [0.9, 0.999]$ controlling tracking speed [4]"

**原文实际情况**: ❌ **原文中没有 EMA threshold calibration**。QTTT 原文不包含任何 threshold、gating 或 adaptive computation mechanism。这是我们自己提出的 Dynamic Gating 设计。

**严重性**: 🔴 **高** — 将自己的创新归因于他人论文

#### 3.4 Target Update Rate Maintenance（第337行）
> 我们论文: "proportional control ensures predictable computational budgeting [4]"

**原文实际情况**: ❌ 原文中不存在此机制

**严重性**: 🔴 **高**

#### 3.5 Oracle Recovery 82–89%（第339行）
> 我们论文: "Empirical validation achieves **82–89% oracle recovery**—gating decisions matching perfect foresight in 82–89% of cases [4]"

**原文实际情况**: ❌ **原文中完全不存在**。QTTT 论文中没有 gating decision 实验，也没有 oracle recovery 概念。数值 "82" 和 "89" 在原文中只出现在参考文献页码中。

**严重性**: 🔴 **高** — 虚构引用数据

#### 3.6 TTT Reconstruction Loss 作为 Gating Signal 的概念（第300行）
> 我们论文: "Computed from a dedicated test-time training layer using frozen key-value caches from initial prefill, this loss provides immediate difficulty assessment without additional forward passes [4]"

**原文实际情况**: ⚠️ 部分相关。原文确实使用 frozen KV cache 和 $L_{TTT}$，但目的完全不同：
- **原文用途**: 用 $L_{TTT}$ 做 gradient descent 来更新 query parameters
- **我们论文用途**: 用 $L_{rec}$ 作为 gating signal 来决定是否触发 adaptation

**严重性**: 🟡 **中** — 概念借用但用途不同，不应标注 [4] 作为 gating signal 的来源

#### 3.7 "Liu et al." 作者引用（第63行）
> 我们论文: "Liu et al. [4] establish that achieving reliable retrieval requires logarithmic margin growth"

**原文实际情况**: ❌ 第一作者是 Rachit Bansal，不是 Liu。论文作者列表中确实有 "Bingbin Liu" 出现在致谢部分（Acknowledgments），但不是论文作者。

**严重性**: 🟡 **中** — 作者归属错误

---

## 三、引用分类汇总

### 按严重性分类

| 严重性 | 数量 | 说明 |
|--------|------|------|
| 🔴 高（虚构引用数据） | **4** | correlation r=0.42-0.84、oracle recovery 82-89%、EMA threshold、gating mechanism |
| 🟡 中（不准确归因） | **2** | reconstruction loss 命名/用途、Liu et al. 作者 |
| 🟢 正确 | **7** | score dilution 概念、数学公式、log margin、frozen KV、FLOP 等价、thinking token 限制、12.6%/14.1% 数据 |

### 按 [4] 引用位置汇总

| 行号 | 引用内容 | 判定 |
|------|---------|------|
| L13 | score dilution problem | ✅ 正确 |
| L63 | "Liu et al. [4]" logarithmic margin | ⚠️ 作者错，内容正确 |
| L120 | $1/(1+me^{-\Delta})$ score dilution bound | ✅ 正确 |
| L25 | qTTT: frozen KV + query adaptation | ✅ 正确（但 pseudo-query 不是 [4] 的概念） |
| L300-302 | TTT reconstruction loss as gating signal | ❌ 概念错误 |
| L304 | correlation r=0.42 to 0.84 | ❌ **虚构数据** |
| L321 | score dilution → qTTT connection | ✅ 正确 |
| L329-331 | EMA threshold calibration [4] | ❌ **原文不存在** |
| L337 | proportional control budgeting [4] | ❌ **原文不存在** |
| L339 | oracle recovery 82-89% [4] | ❌ **虚构数据** |
| L361-374 | FLOP equivalence derivation | ✅ 正确 |
| L401 | thinking tokens can't repair dilution | ✅ 正确 |
| L412 | 12.6% and 14.1% improvement | ✅ 正确 |

---

## 四、修正建议

### 立即修正（P0）

1. **参考文献 [4] 条目**:
   ```
   [4] Bansal, R., Zhang, A., Tiwari, R., Madaan, L., Duvvuri, S.S., Khatri, D., 
       Brandfonbrener, D., Alvarez-Melis, D., Bhargava, P., Kale, M.S., Jelassi, S.
       "Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs."
       arXiv:2512.13898, 2025.
   ```

2. **第63行**: `Liu et al. [4]` → `Bansal et al. [4]`

3. **删除所有错误归因于 [4] 的内容**:
   - 第304行: 删除 `[4]`，标注为我们的实验结果或移除此数据
   - 第329-331行: 删除 `[4]`，EMA threshold 是我们的 Dynamic Gating 设计
   - 第337行: 删除 `[4]`
   - 第339行: 删除 `[4]`，oracle recovery 数据要么提供来源要么删除

4. **重新审视 Dynamic Gating 部分**:
   - 第300-302行的 $L_{rec}$ 公式与 QTTT 的 $L_{TTT}$ 公式结构相似，但用途不同
   - 建议改为："$L_{rec}$ 是受 [4] 中 TTT loss 启发的自适应设计，但用途从 query optimization 转变为 difficulty gating"

### 建议修正（P1）

5. **第25行**: "pseudo-queries or query projections" → 删除 "pseudo-queries"。pseudo-queries 是我们 AttnRes 的概念，不是 QTTT 的。

6. **第300行**: "reconstruction loss" → 建议改为 "next-token prediction loss (inspired by [4])" 或 "adaptation loss"

### 术语对齐（P2）

7. QTTT 原文中将损失函数称为 **$L_{TTT}$**（next-token prediction loss），不是 "reconstruction loss"。我们的论文中称为 $\mathcal{L}_{rec}$，建议在文中说明这是我们借鉴 [4] 的重新命名。

---

## 五、QTTT 原文关键信息摘要

### 方法概述
- **输入**: 长上下文 $x_{1:T}$，步骤数 $N_{TTT}$，span 长度 $k$，学习率 $\eta$
- **算法**:
  1. 一次前向传播缓存所有层的 K, V（$O(T^2)$）
  2. 重复 $N_{TTT}$ 次：随机采样 span $x_s=x_{t:t+k}$，用 frozen KV 计算 $L_{TTT}$，**只更新** query projection matrices $\{W_Q^{(\ell)}\}$
  3. 返回适配后的模型生成最终答案

### 核心理论贡献
1. **Lemma 2.2 (Score Dilution)**: $m$ 个 distractor 在 $\Delta$ 范围内时，target attention mass $\leq 1/(1+me^{-\Delta})$
2. **Lemma 2.3 (Logarithmic Margin Requirement)**: 需要 $\Omega(\log T)$ 的 margin 才能保持固定 target mass
3. **Proposition 2.4 (Needle-signal bound)**: thinking token 携带的 needle signal 不超过其自身对 needle 的 attention mass
4. **Proposition 3.1 (Directional query update)**: gradient step 严格增大 target-distractor margin
5. **Eq.(3.2) (FLOP Equivalence)**: $T_{think} \approx 2N_{qTTT}k$

### 原文 **不包含** 的内容
- ❌ Gating mechanism（adaptive computation）
- ❌ Threshold calibration（EMA 或 target-rate）
- ❌ Oracle recovery 实验
- ❌ Reconstruction loss 作为 difficulty signal
- ❌ Correlation with oracle advantage
- ❌ Pseudo-queries 或 AttnRes 相关概念

### 原文实验数据
- **模型**: Qwen3-{1.7B, 4B, 8B}
- **LongBench-v2**: Thinking 33.5% → qTTT 39.7% (+6.2)
- **ZeroScrolls**: Thinking 23.9% → qTTT 32.5% (+8.6)
- **默认参数**: $T_{think}=8192, k=128, N_{qTTT}=32$

---

*报告生成时间: 2026-03-27 10:30 CST*
