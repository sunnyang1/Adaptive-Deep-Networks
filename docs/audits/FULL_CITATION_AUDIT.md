# 全文引用对照审查报告

**论文**: Adaptive Deep Networks: Integrating Attention Residuals, Dynamic Computation, and Test-Time Adaptation  
**审查日期**: 2026-03-27  
**引用总数**: 58 个  

---

## 一、审查总览

| 严重性 | 数量 | 引用编号 |
|--------|------|----------|
| 🔴 严重错误 | **5** | [4], [9], [14], [35], [42] |
| ⚠️ 需修正 | **12** | [2], [5], [6], [16], [18], [19], [23], [27], [34], [40], [55], [58] |
| 🟢 准确/可接受 | **41** | 其余所有引用 |

---

## 二、🔴 严重错误（P0 — 必须立即修正）

### [4] 作者和标题完全错误 — ✅ 已修正
- **我们写的**: `Liu, J., et al. "QTTT: Query-Only Test-Time Training for Long-Context Retrieval." arXiv:2512.13898, 2025.`
- **实际应为**: `Bansal, R., Zhang, A., Tiwari, R., et al. "Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs." arXiv:2512.13898, 2025.`
- **问题**: 作者完全错误（Bansal 非 Liu），标题完全错误
- **正文影响**: 第63行 `Liu et al. [4]` → `Bansal et al. [4]`
- **正文归因错误**: 第300-339行多处将 gating mechanism、EMA threshold、oracle recovery 82-89%、correlation r=0.42-0.84 错误归因于 [4]，这些概念在 QTTT 原文中**完全不存在**（详见 `QTTT_CITATION_AUDIT.md`）
- **修正**:
  ```
  [4] Bansal, R., Zhang, A., Tiwari, R., Madaan, L., Duvvuri, S.S., Khatri, D.,
      Brandfonbrener, D., Alvarez-Melis, D., Bhargava, P., Kale, M.S., Jelassi, S.
      "Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs."
      arXiv:2512.13898, 2025.
  ```

### [9] 引用编号指错论文
- **我们写的**: `Riquelme, C., et al. "Scaling vision with sparse mixture of experts." NeurIPS, 2021.`
- **问题**: 参考文献条目本身正确（Vision MoE），但**正文**中多次将 [9] 引用为 **"Mixture of Depths (MoD)"**（第55行、第69行）
- **实际 Mixture of Depths 论文**: Raposo et al., "Mixture of Depths: Dynamically Allocating Compute in Transformer-Based Language Models," arXiv:2404.02258, 2024
- **修正**: 需新增 MoD 论文为引用 [59]，并将正文中 [9] 对应 MoD 的引用改为 [59]

### [14] 第一作者姓名有误
- **我们写的**: `Xiao, C., et al.` (StreamingLLM)
- **实际应为**: `Xiao, G., et al.` (Guangxuan Xiao)
- **修正**: `[14] Xiao, G., et al. "Efficient streaming language models with attention sinks." ICLR, 2024.`

### [35] 作者完全错误
- **我们写的**: `Xiao, G., et al. "Efficient large-scale language model training on GPU clusters using megatron-LM." arXiv, 2023.`
- **实际应为**: `Narayanan, D., Shoeybi, M., Casper, J., et al. "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM." SC, 2021.`
- **问题**: 第一作者应为 **Narayanan, D.**（Deepak Narayanan），不是 Xiao, G.。该论文发表于 **SC 2021**（非 arXiv 2023）
- **修正**:
  ```
  [35] Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M.,
      Korthikanti, V.A., Vainbrand, D., Kashinkunti, P., Bernauer, J.,
      Catanzaro, B., Phanishayee, A., Zaharia, M.
      "Efficient large-scale language model training on GPU clusters using megatron-LM." SC, 2021.
  ```

### [42] 标题语法错误
- **我们写的**: `"Makes large language models better reasoners with step-aware verifier."`
- **实际应为**: `"Making large language models better reasoners with step-aware verifier."`
- **问题**: 标题中 "Makes" 应为动名词 "Making"
- **修正**: `[42] Liao, B., et al. "Making large language models better reasoners with step-aware verifier." arXiv, 2024.`

---

## 三、⚠️ 需修正（P1）

### [2] 年份可优化
- **我们写的**: `TPAMI, 2020`
- **建议**: 论文在线预出版为2020年12月，但正式卷号 Vol.44, No.8 为 2022年。可标注 `TPAMI, 2021` 或保持 `TPAMI, 2020`（均可接受）

### [5] 建议补充 arXiv 编号
- **建议**: 添加 `arXiv:2112.11446`

### [6] 发表场所不完整
- **我们写的**: `arXiv, 2022`
- **实际**: 已被 **NeurIPS 2022** 接收（Chinchilla 论文）
- **修正**: `[6] Hoffmann, J., et al. "Training compute-optimal large language models." NeurIPS, 2022.`

### [16] 发表场所不完整
- **我们写的**: `arXiv, 2021`
- **实际**: 已被 **NeurIPS 2021** 接收
- **修正**: `[16] Cobbe, K., et al. "Training verifiers to solve math word problems." NeurIPS, 2021.`

### [18] 发表场所不完整
- **我们写的**: `arXiv, 2023`
- **实际**: 已被 **NeurIPS 2024** 接收
- **修正**: `[18] Zhang, T., et al. "LongBench: A bilingual, multitask benchmark for long context understanding." NeurIPS, 2024.`

### [19] 标题不完整 + 发表场所不完整
- **我们写的**: `"ZeroSCROLLS: Zero-shot evaluation on long document understanding." arXiv, 2023.`
- **实际**: `"ZeroSCROLLS: Zero-shot evaluation of long context extraction and summarization." EMNLP (Findings), 2023.`
- **修正**:
  ```
  [19] Shaham, U., et al. "ZeroSCROLLS: Zero-shot evaluation of long context
      extraction and summarization." EMNLP (Findings), 2023.
  ```

### [23] 发表场所不完整
- **我们写的**: `arXiv, 2022`
- **实际**: 已被 **ICML 2023** 接收
- **修正**: `[23] Wang, H., et al. "DeepNet: Scaling transformers to 1,000 layers." ICML, 2023.`

### [27] 发表场所不完整
- **我们写的**: `arXiv, 2020`
- **实际**: 已被 **ACL 2020** 接收
- **修正**: `[27] Beltagy, I., et al. "Longformer: The long-document transformer." ACL, 2020.`

### [34] 第一作者姓名有误
- **我们写的**: `Zhang, Z., et al.` (H2O)
- **实际应为**: `Xiao, G., et al.` 或进一步核实（该论文 arXiv ID 需确认）
- **说明**: H2O 论文的第一作者首字母存疑，建议通过 arXiv 或 Google Scholar 核实

### [40] 发表场所不完整
- **我们写的**: `arXiv, 2016`
- **实际**: 已被 **ICML 2016** 接收
- **修正**: `[40] Graves, A. "Adaptive computation time for recurrent neural networks." ICML, 2016.`

### [55] 发表场所/年份矛盾
- **我们写的**: `ICLR, 2023`
- **实际**: arXiv 2023年3月预印本，被 **ICLR 2024** 接收
- **修正**: `[55] Sheng, Y., et al. "FlexGen: High-throughput generative inference of large language models with a single GPU." ICLR, 2024.`

### [58] 元数据基本正确（但存在风险）
- **我们写的**: `Chen, G., Zhang, Y., Su, J., Xu, W., Pan, S., Wang, Y., et al. "Attention Residuals." Technical Report, Kimi Team. arXiv:2603.15031, 2026.`
- **实际**: 论文已于 2026-03-17 发表于 arXiv，标题 "Attention Residuals" ✅，arXiv ID 2603.15031 ✅
- **实际作者**: Guangyu Chen, Yu Zhang, Jianlin Su, Weixin Xu, Siyuan Pan, Yaoyu Wang, Yucheng Wang, Guanduo Chen, Bohong Yin, Yutian Chen, Junjie Yan, Ming Wei, Y. Zhang, Fanqing Meng, Chao Hong, Xiaotong Xie, Shaowei Liu, Enzhe Lu, Yunpeng Tai, Yanru Chen, Xin Men, Haiqing Guo, Y. Charles, Haoyu Lu, Lin Sui, Jinguo Zhu, Zaida Zhou, Weiran He, Weixiao Huang, Xinran Xu, Yuzhi Wang, Guokun Lai, Yulun Du, Yuxin Wu, Zhilin Yang, Xinyu Zhou
- **问题**: 引用的前6位作者正确（Chen, G., Zhang, Y., Su, J., Xu, W., Pan, S. ✅），但第7位写的 `Wang, Y.` 应对应 `Yaoyu Wang` ✅，第8位写的 `et al.` 跳过了后续35位作者
- **正文归因**: [58] 在正文中出现5次（第21行、第53行、第136行、第142行、第169行），引用内容均与原文一致 ✅
- **风险**: 论文为 Kimi Team 内部技术报告，非同行评审论文。作为项目核心依赖，需注意
- **建议**: 补充完整作者列表或至少标注 `Chen, G., et al.`

---

## 四、引用 [58] 正文归因审查

[58] "Attention Residuals" (Kimi Team, arXiv:2603.15031) 在正文中被引用 **5 处**，逐一审查：

| 位置 | 引用内容 | 判定 |
|------|---------|------|
| L21 | "building upon the Attention Residuals framework of Chen et al. [58]" | ✅ 正确 |
| L53 | "Our approach builds upon the Attention Residuals framework proposed by Chen et al. [58], who formalized the duality between depth-wise accumulation and sequential recurrence in RNNs." | ✅ 正确（原文确实提出了 depth-sequence duality） |
| L136 | "The complete Attention Residuals framework [58] replaces the fixed accumulation..." | ✅ 公式与原文一致 |
| L142 | "Block AttnRes [58], which partitions the L layers into N blocks" | ✅ Block AttnRes 概念与原文一致 |
| L169 | "Block AttnRes enables efficient implementation through a two-phase computation strategy [58]" | ✅ 两阶段计算策略与原文一致 |

**结论**: [58] 的正文归因全部正确，无虚构或不准确归因。

---

## 五、完整审查结果表

| # | 判定 | 问题 |
|---|------|------|
| [1] | ✅ | — |
| [2] | ⚠️ | 年份可优化（TPAMI 2020/2021/2022 均可） |
| [3] | ✅ | — |
| [4] | 🔴 | 作者/标题完全错误 + 正文多处虚构归因 |
| [5] | ⚠️ | 建议补充 arXiv ID |
| [6] | ⚠️ | 应标注 NeurIPS 2022 |
| [7] | ✅ | — |
| [8] | ✅ | — |
| [9] | 🔴 | 正文引用内容（MoD）与参考文献条目（Vision MoE）不匹配 |
| [10] | ✅ | — |
| [11] | ✅ | — |
| [12] | ✅ | 建议补充 arXiv ID |
| [13] | ✅ | — |
| [14] | 🔴 | 第一作者 Xiao, C. → Xiao, G. |
| [15] | ✅ | — |
| [16] | ⚠️ | 应标注 NeurIPS 2021 |
| [17] | ✅ | — |
| [18] | ⚠️ | 应标注 NeurIPS 2024 |
| [19] | ⚠️ | 标题不完整 + 应标注 EMNLP 2023 |
| [20] | ✅ | — |
| [21] | ✅ | — |
| [22] | ✅ | — |
| [23] | ⚠️ | 应标注 ICML 2023 |
| [24] | ✅ | — |
| [25] | ✅ | — |
| [26] | ✅ | — |
| [27] | ⚠️ | 应标注 ACL 2020 |
| [28] | ✅ | — |
| [29] | ✅ | — |
| [30] | ✅ | — |
| [31] | ✅ | — |
| [32] | ✅ | — |
| [33] | ✅ | — |
| [34] | ⚠️ | 作者首字母 Zhang, Z. 存疑 |
| [35] | 🔴 | 作者完全错误（Xiao → Narayanan），年份/场所错误 |
| [36] | ✅ | — |
| [37] | ✅ | — |
| [38] | ✅ | — |
| [39] | ✅ | — |
| [40] | ⚠️ | 应标注 ICML 2016 |
| [41] | ✅ | — |
| [42] | 🔴 | 标题语法错误（Makes → Making） |
| [43] | ✅ | — |
| [44] | ✅ | — |
| [45] | ✅ | — |
| [46] | ✅ | — |
| [47] | ✅ | — |
| [48] | ✅ | — |
| [49] | ✅ | — |
| [50] | ✅ | — |
| [51] | ✅ | — |
| [52] | ✅ | — |
| [53] | ✅ | — |
| [54] | ✅ | — |
| [55] | ⚠️ | 应标注 ICLR 2024（非 2023） |
| [56] | ✅ | — |
| [57] | ✅ | — |
| [58] | ⚠️ | 元数据基本正确，正文归因全部正确 |

---

## 六、系统性问题

### 1. "arXiv" 发表场所系统性遗漏正式会议
以下论文已在正式顶会发表，但仍标注为 arXiv：

| 引用 | 当前标注 | 应改为 |
|------|---------|--------|
| [6] | arXiv, 2022 | NeurIPS, 2022 |
| [16] | arXiv, 2021 | NeurIPS, 2021 |
| [18] | arXiv, 2023 | NeurIPS, 2024 |
| [19] | arXiv, 2023 | EMNLP, 2023 |
| [23] | arXiv, 2022 | ICML, 2023 |
| [27] | arXiv, 2020 | ACL, 2020 |
| [35] | arXiv, 2023 | SC, 2021 |
| [40] | arXiv, 2016 | ICML, 2016 |
| [55] | ICLR, 2023 | ICLR, 2024 |

### 2. 引用编号 [9] 指向错误论文
- 参考文献列表中 [9] 是 Vision MoE (Riquelme et al.)
- 正文中 [9] 被用作 Mixture of Depths (MoD) 的引用
- 需新增 MoD 论文或将 [9] 替换

### 3. 引用 [4] 的正文归因存在严重学术诚信问题
详见 `QTTT_CITATION_AUDIT.md`：
- 4 处虚构引用数据（correlation、oracle recovery、EMA threshold、gating mechanism）
- 1 处作者错误
- 需立即修正

---

## 七、修正优先级

### P0 — 立即修正（学术诚信风险）
1. **[4]** 修正参考文献条目 + 正文所有归因
2. **[9]** 新增 MoD 引用并修正正文指向
3. **[35]** 修正作者和发表信息

### P1 — 尽快修正（影响论文质量）
4. **[14]** 第一作者 Xiao, C. → Xiao, G.
5. **[42]** 标题语法 Makes → Making
6. **[6][16][18][19][23][27][40][55]** 更新发表场所

### P2 — 建议修正（提升规范性）
7. 统一 arXiv ID 标注风格
8. **[34]** 核实作者首字母
9. **[58]** 考虑补充完整作者列表

---

*报告生成时间: 2026-03-27 11:00 CST*
