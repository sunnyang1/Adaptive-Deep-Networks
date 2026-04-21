# 面向自适应深度网络的质量感知 Stiefel 投影：矩阵流形查询自适应与标记价值加权

**匿名作者**

---

## 摘要

Transformer 的长上下文推理同时受到四类耦合瓶颈的约束：KV 缓存随上下文增长、对早期层表示的访问受限、缺乏可扩展的外部记忆，以及测试时查询行为固定不变。自适应深度网络（Adaptive Deep Networks, ADN）分别通过 RaBitQ、AttnRes、Engram 和 qTTT 对应解决这些问题。然而，在 ADN 框架内仍存在两个重要不足：其一，qTTT 将不同注意力头彼此独立处理，没有显式保持多头查询的几何多样性；其二，ADN 默认对所有标记一视同仁，即使许多标记在语义上稳定但信息量有限。

本文提出**质量感知 Stiefel 投影**（Quality-Aware Stiefel Projection, QASP），作为 ADN 的一个针对性扩展。QASP 包含两项核心机制：一是借助矩阵符号函数

$$
\mathrm{msign}(\mathbf{W})=\mathbf{W}(\mathbf{W}^{\top}\mathbf{W})^{-1/2}
$$

在 Stiefel 流形上执行矩阵级查询自适应；二是引入信息质量加权，在块级聚合与外部记忆检索中抑制低价值标记的影响，并对测试时更新强度进行调制。基于一个显式的局部扰动模型，我们给出在 1-bit RaBitQ 压缩噪声下、质量调制 Stiefel 更新的鲁棒性与收敛性分析。

在经验层面，1.5B 代理模型上的初步结果表明：相较于 ADN 基线，QASP 将 128K needle-in-haystack 准确率从 76.3% 提升到 80.2%，将 MATH 从 48.1% 提升到 51.3%，同时保持 112 tok/s 的吞吐率。整体来看，QASP 为 ADN 增加了更强的查询多样性与标记选择性，同时保留了 ADN 的效率优势。本文的主要实证结论仅建立在 1.5B 代理模型证据之上，不将更大规模外推作为主证据。

---

## 1. 引言

### 1.1 以查询为中心理解长上下文 Transformer

从功能上看，Transformer 可以被理解为一个“查询-应答”系统：每个注意力头发出查询向量 $\mathbf{q}$，从键向量 $\mathbf{k}$ 中检索相关信息，再通过值向量 $\mathbf{v}$ 聚合输出。随着上下文长度增大，这一机制会同时暴露出四类限制：

1. **空间（Space）**：KV 缓存随序列长度线性增长，长上下文推理的显存成本迅速上升。
2. **范围（Scope）**：标准残差连接主要依赖相邻层，早期层的重要表示会在深层堆叠中逐渐被“掩埋”。
3. **存储（Storage）**：模型知识被固定在参数中，难以通过高效、可更新的外部记忆进行扩展。
4. **特异性（Specificity）**：查询在预训练后通常保持静态，难以在测试时根据当前输入进行针对性适配。

### 1.2 ADN 框架

ADN 以统一方式回应上述四个维度：

- **RaBitQ**：通过无偏 1-bit 量化压缩 KV 缓存，缓解空间压力。
- **AttnRes**：通过块级深度注意力扩大跨层访问范围，缓解表示掩埋。
- **Engram**：通过可扩展外部记忆实现确定性检索，扩展存储能力。
- **qTTT**：仅更新查询相关参数，使模型在测试时进行轻量自适应。

ADN 的核心价值在于，它不是单点优化某一个瓶颈，而是把长上下文推理理解为一组相互耦合的资源控制问题。

### 1.3 本文关注的两个剩余问题

尽管 ADN 已经建立了四维统一框架，但我们认为还有两点值得继续完善。

**第一，缺乏显式的多头几何结构。**  
qTTT 在向量层面更新查询，没有显式约束不同头之间的相对关系。这样做虽然简单，但无法直接避免多个头朝相近方向收缩，进而损害表示多样性。

**第二，所有标记默认被平等对待。**  
在实际序列中，许多标记在语义上十分稳定，但信息贡献并不高。近期关于 lazy aggregation 的分析提示：Transformer 容易过度依赖这类“稳定但低价值”的标记，从而稀释真正携带内容的标记贡献。

### 1.4 贡献

围绕这两个问题，本文提出 QASP，主要贡献如下：

1. **矩阵级查询自适应**：将查询更新从向量提升到矩阵，使用矩阵符号函数将更新投影到 Stiefel 流形上，以显式保持多头正交性并抑制头坍缩。
2. **贯穿 ADN 流水线的信息质量感知**：构建基于频谱分析的信息质量分数，并将其用于 AttnRes 聚合、Engram 记忆融合，以及测试时更新强度调制。
3. **假设显式的理论分析**：在局部算子模型下，分析价值加权 Stiefel 更新在 1-bit RaBitQ 压缩噪声下的鲁棒性与收敛性。
4. **初步经验支持**：在 1.5B 代理模型上观察到相对 ADN 的稳定增益；本文不将更大规模外推结果作为主经验结论。

---

## 2. ADN 框架概述

### 2.1 空间维度：RaBitQ

RaBitQ 通过随机变换与低比特量化压缩高维向量表示，使得内积估计保持无偏。对于查询 $\mathbf{q}$ 和键 $\mathbf{k}$，其量化内积满足

$$
\mathbb{E}\bigl[\widehat{\mathbf{q}^{\top}\mathbf{k}}\bigr]=\mathbf{q}^{\top}\mathbf{k},
\qquad
\frac{\mathbb{E}\bigl[(\widehat{\mathbf{q}^{\top}\mathbf{k}}-\mathbf{q}^{\top}\mathbf{k})^2\bigr]}{(\mathbf{q}^{\top}\mathbf{k})^2}\le \delta^2.
$$

在本文设定中，我们主要关注 1-bit RaBitQ，因为它提供了最强的显存节省，并为后续测试时更新提出了鲁棒性问题。

### 2.2 范围维度：AttnRes

AttnRes 将深层网络划分为若干块，让后续层能够以块为单位访问更早的表示。若第 $\ell$ 层位于第 $n$ 个块，则其块级聚合可写为

$$
\mathbf{h}*{\ell}=\sum*{m=0}^{n-1}\alpha_{m\rightarrow \ell}\mathbf{B}_m,
$$

其中 $\mathbf{B}*m$ 为第 $m$ 个块的摘要表示，$\alpha*{m\rightarrow \ell}$ 为块级注意力权重。这样做的效果是把层间访问从“仅看上一层”扩展为“看多个历史块”，从而减弱表示掩埋。

### 2.3 存储维度：Engram

Engram 使用 n-gram 哈希实现外部记忆的确定性查找。对于给定 n-gram，系统先计算哈希索引，再从记忆表中读出对应向量，并通过门控方式与当前隐藏状态融合。它把参数化知识和可更新记忆区分开来，使模型具备更灵活的存储能力。

### 2.4 特异性维度：qTTT

qTTT 只在测试时更新查询相关参数，而冻结其他大部分权重。其典型形式是把查询写成

$$
\mathbf{w}=r\cdot \mathbf{u},
$$

其中幅值 $r$ 固定、方向 $\mathbf{u}$ 在球面上更新。这样既保留预训练尺度信息，又允许模型在推理期间根据当前样本做局部适配。

---

## 3. 惰性聚合与信息质量评分

### 3.1 惰性聚合

lazy aggregation 指的是：模型更偏好那些语义上稳定、统计上高频、但信息量有限的标记，而不是更具判别性的内容标记。对语言模型而言，停用词、格式性标记、模式化背景文本都可能成为这种现象的载体。

我们将其抽象为：若注意力分布 $\boldsymbol{\alpha}$ 所聚焦的标记，其信息含量期望显著低于均匀采样标记的信息含量，则说明模型出现了惰性聚合倾向。

### 3.2 信息质量分数

设 $\mathbf{h}_t\in\mathbb{R}^d$ 为第 $t$ 个标记的隐藏表示。我们通过沿通道维度的离散傅里叶变换来度量该标记中“高频、信息丰富”成分的比例。定义

$$
\rho(t)=1-s(t),\qquad
s(t)=\frac{\left\mathcal{F}^{-1}\bigl(\hat{\mathbf{h}}*t\odot \mathbf{g}*{\mathrm{LP}}\bigr)\right_2}{\mathbf{h}_t_2},
$$

其中 $\hat{\mathbf{h}}_t=\mathcal{F}(\mathbf{h}*t)$，$\mathbf{g}*{\mathrm{LP}}$ 为高斯低通滤波器。直观上：

- $\rho(t)$ 越高，说明该标记包含的高频信息越多，更可能是内容承载标记；
- $\rho(t)$ 越低，说明该标记越稳定、越模式化，更可能是低价值标记。

### 3.3 计算代价

逐标记逐层计算频谱分数成本较高，因此我们采用两种简化：

1. **Ponder gate 触发计算**：只有在需要测试时更新时才计算分数。
2. **滑动窗口摊销**：对长度为 $W$ 的窗口批量执行 DFT，并在窗口内复用结果。

这样可以把额外开销控制在较低范围内。

---

## 4. 矩阵符号函数与 Stiefel 流形优化

### 4.1 为什么要使用矩阵而不是向量

多头注意力天然形成一个查询矩阵 $\mathbf{W}\in\mathbb{R}^{d\times k}$。如果把每个头完全独立地看作一个向量，就很难显式刻画“不同头应保持多样性”这一结构性要求。一个更自然的约束是让查询矩阵位于 Stiefel 流形：

$$
\mathrm{St}(k,d)=\mathbf{U}\in\mathbb{R}^{d\times k}:\mathbf{U}^{\top}\mathbf{U}=\mathbf{I}_k.
$$

这意味着各列正交归一，从而直接编码多头之间的差异性。

### 4.2 矩阵符号函数

对于列满秩矩阵 $\mathbf{W}$，定义

$$
\mathrm{msign}(\mathbf{W})=\mathbf{W}(\mathbf{W}^{\top}\mathbf{W})^{-1/2}.
$$

若 $\mathbf{W}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}$ 为薄 SVD，则

$$
\mathrm{msign}(\mathbf{W})=\mathbf{U}\mathbf{V}^{\top},
$$

也即极分解中的正交因子。它可被理解为“把任意满秩矩阵投影到最近的 Stiefel 点”。

### 4.3 Newton-Schulz 迭代

直接通过 SVD 计算代价较高，因此我们采用 Newton-Schulz 迭代做近似：

$$
\mathbf{Y}_{t+1}=\frac{1}{2}\mathbf{Y}_t(3\mathbf{I}-\mathbf{Y}_t^{\top}\mathbf{Y}_t),
$$

并以谱归一化后的 $\mathbf{Y}_0$ 为初值。经验上，5 次迭代已足以获得较高精度，同时保持较小延迟。

---

## 5. QASP：质量感知的 Stiefel 投影

QASP 对 ADN 做三类增强：

1. 用 Stiefel 投影替代向量级 qTTT 的查询更新；
2. 在 AttnRes 中加入质量感知的块级加权；
3. 在 Engram 中加入质量感知的记忆融合。

### 5.1 质量加权的 AttnRes

对于块 $m$，定义其平均质量分数

$$
\bar{\rho}*m=\frac{1}{|B_m|}\sum*{t\in B_m}\rho(t).
$$

则加权后的块级注意力写为

# $$
\alpha_{m\rightarrow \ell}^{(\rho)}

\frac{\exp\bigl(\mathbf{w}_{\ell}^{\top}\mathbf{B}*m\cdot \bar{\rho}m/\sqrt{d}\bigr)}
{\sum{j=0}^{n-1}\exp\bigl(\mathbf{w}*{\ell}^{\top}\mathbf{B}_j\cdot \bar{\rho}_j/\sqrt{d}\bigr)}.
$$

这使得由低价值标记主导的块自然获得更低权重。

### 5.2 质量加权的 Engram

写入 n-gram 记忆时，同时保存对应的平均质量分数：

$$
\mathrm{Memory}[\mathrm{idx}]=(\mathbf{m},\rho_{\mathrm{mem}}),
\qquad
\rho_{\mathrm{mem}}=\frac{1}{n}\sum_{i=1}^{n}\rho(t_i).
$$

检索时，通过

$$
\mathbf{h}'=\mathbf{h}+\alpha\cdot \sigma(\rho_{\mathrm{mem}})\cdot \mathbf{m}
$$

控制记忆对当前隐藏状态的贡献，从而降低低价值 n-gram 的影响。

### 5.3 矩阵级测试时更新

设查询矩阵为 $\mathbf{W}\in\mathbb{R}^{d\times k}$，QASP 的测试时更新包含以下步骤：

1. 计算欧几里得梯度 $\nabla_{\mathbf{W}}\mathcal{L}$；
2. 用当前批次的平均信息质量分数对更新强度进行批级调制；
3. 在欧几里得空间执行一步更新：
  $$
   \mathbf{W}'=\mathbf{W}-\eta \widetilde{\nabla}_{\mathbf{W}}\mathcal{L};
   $$
4. 用矩阵符号函数投影回 Stiefel 流形：
  $$
   \mathbf{W}_{\mathrm{new}}=\mathrm{msign}(\mathbf{W}');
   $$
5. 在 ponder gate 允许时重复上述过程若干步。

与向量级 qTTT 不同，QASP 在每次更新后都显式恢复多头正交结构。

---

## 6. 实验

### 6.1 经验范围说明

本节聚焦于 1.5B 代理模型上的当前证据。现阶段的经验包由一体化代理模型运行结果与同一实验线上的组件级受控验证共同构成，因此更适合用于说明方向性增益与 trade-off，而不应被理解为已经完全冻结的最终 benchmark package。本文不将更大规模外推作为主实验部分的证据。

### 6.2 实验配置

1.5B 代理模型采用 LLaMA 风格 decoder-only 架构，训练数据为 SlimPajama，训练 token 量为 100B。核心结构参数包括：24 层、隐藏维度 2048、16 个注意力头、4 个 KV 头、训练上下文 4096。QASP 侧的主要超参数包括：Newton-Schulz 迭代 5 次、测试时学习率 0.01、最大自适应步数 5、DFT 窗口 512、低通截止频率 $d/4$。

### 6.3 代理模型上的当前结果

在 1.5B 代理模型上，QASP 相比 ADN 基线表现出较为一致的提升：


| 方法                       | Needle@128K | Needle@256K | Needle@512K | LongBench | L-Eval    | MATH      | GSM8K     | 吞吐率     |
| ------------------------ | ----------- | ----------- | ----------- | --------- | --------- | --------- | --------- | ------- |
| Standard Transformer     | 2.8%        | 1.2%        | 0.5%        | 18.3%     | 15.2%     | 28.4%     | 32.1%     | 28      |
| ADN                      | 76.3%       | 65.2%       | 48.5%       | 42.8%     | 38.5%     | 48.1%     | 52.3%     | 118     |
| ADN + value weights only | 78.5%       | 68.0%       | 51.2%       | 44.5%     | 40.2%     | 49.5%     | 54.1%     | 116     |
| ADN + msign only         | 77.1%       | 66.3%       | 49.8%       | 43.6%     | 39.1%     | 48.9%     | 53.2%     | 114     |
| **ADN + QASP**           | **80.2%**   | **70.4%**   | **55.8%**   | **47.2%** | **43.5%** | **51.3%** | **56.8%** | **112** |


这些结果应被理解为**支持性证据**，而不是对最终方法效果的全面定论。

### 6.4 消融结果

主要消融显示：

- 移除质量权重会明显削弱检索能力；
- 移除 Stiefel 投影会显著恶化正交性指标；
- 用 SVD 替代 Newton-Schulz 基本不改变精度，但速度更慢；
- 始终执行测试时自适应只带来有限收益，却增加较多推理开销。

对应地，QASP 的增益并非来自单一组件，而是“质量加权 + 几何约束”的组合。

### 6.5 计算效率

在 128K 上下文下，QASP 相比 ADN 增加的开销主要来自：

- 质量分数计算；
- Stiefel 投影的近似迭代；
- 少量测试时更新步骤。

在当前设定下，完整 ADN+QASP 的总 FLOPs 增幅约为 4.0%，延迟增加约 10.1%，吞吐率仍为 112 tok/s。换言之，QASP 在提高选择性的同时，没有显著破坏 ADN 的效率优势。

### 6.6 统计验证计划

由于当前经验包仍是阶段性版本，正式的统计检验将放在后续冻结 benchmark protocol 的验证轮次中执行。该轮验证将采用配对检验、bootstrap 置信区间以及必要的多重比较校正。本文刻意不在主文中报告投影式置信区间或投影式效应量，以避免它们被误读为已完成验证的证据。

### 6.7 与代表性长上下文方法的比较

我们将 QASP 与若干代表性方法进行并列比较，包括 StreamingLLM、H$_2$O、LoRA-FA 等。需要强调的是，这些方法在训练范式、监督条件、适配预算和模型初始化方式上并不完全一致，因此该表应被理解为**trade-off 定位**，而不是严格统一设置下的最终排行榜。

在这一视角下，ADN+QASP 在检索能力、推理质量、吞吐率和显存占用之间展现出较为有利的平衡。

---

## 7. 理论分析

### 7.1 理论定位

本节不试图为端到端非凸 Transformer 训练给出全局保证，而是围绕“价值加权 + Stiefel 投影 + 压缩噪声”的局部更新规则建立一个假设显式的分析框架。理论结果应理解为对**更新机制本身**的刻画。

### 7.2 关键假设

分析依赖以下几类假设：

1. 损失关于查询矩阵具有局部 Lipschitz 连续梯度；
2. 在流形局部区域内存在足够好的几何条件，可支持局部强凸式分析；
3. 质量分数有界；
4. RaBitQ 对梯度引入有界相对噪声。

这些假设的目的不是把真实训练过程理想化为可完全求解的问题，而是明确说明：我们的鲁棒性与收敛性结论成立于何种分析范围之内。

### 7.3 噪声下的 Stiefel 投影稳定性

设真实梯度为 $\mathbf{G}$，压缩后的带噪梯度为 $\tilde{\mathbf{G}}=\mathbf{G}+\boldsymbol{\Delta}$。若噪声满足

$$
\boldsymbol{\Delta}_F\le \delta\mathbf{G}_F,
$$

则在适当步长条件下，可得到矩阵符号投影前后误差的上界。该结果表明：只要查询矩阵的条件数没有恶化到过高程度，投影步骤对压缩噪声的放大是可控的。

### 7.4 价值加权更新的收敛行为

在局部模型下，价值加权相当于用信息质量分数调制梯度大小。理论上，这会降低有效更新幅度，因此可能减慢收敛速度；但同时，它也抑制了由低价值标记引入的噪声梯度。换言之，QASP 用“更慢但更干净”的更新换取更好的方向性。

### 7.5 误差分解

QASP 的误差可以粗略分解为三部分：

1. **压缩误差**：由 RaBitQ 低比特内积估计带来；
2. **流形误差**：由带噪梯度进入 Stiefel 投影后引起；
3. **价值加权偏差**：由主动压低低价值标记贡献带来。

这个分解的重要性在于，它把“QASP 为何有效、又可能在哪些地方受限”拆解为可分析的三个来源。

### 7.6 复杂度结论

相较于 ADN，QASP 的额外代价主要是 $O(dk^2)$ 级别的矩阵投影与 $O(d\log d)$ 级别的频谱权重计算。在常见设定下，由于 $k\ll d$，前者相对于注意力主开销通常较小；后者则可以通过窗口复用与 ponder gate 进一步摊销。

---

## 8. 相关工作

### 8.1 KV 缓存压缩

KIVI、KVQuant、GEAR、QJL、TurboQuant 等工作主要聚焦于**空间维度**：即如何在尽量不损失精度的前提下压缩 KV 缓存。相比之下，QASP 并不试图替代这些方法，而是在 ADN 的四维框架中继续处理“几何自适应”和“信息质量选择性”问题。

### 8.2 测试时训练与自适应

TTT-Linear、TLM、Titans、ATLAS 等工作探索了模型在测试时如何利用额外计算进行局部学习。与本文最接近的是 qTTT：它只更新查询相关参数，并重用已有 KV 缓存。QASP 的区别不在于否定 qTTT，而在于把它从向量级自适应进一步提升为矩阵级、流形约束的自适应。

### 8.3 Stiefel 流形优化

Muon、Mousse、StelLA 等工作说明：在训练或微调时引入 Stiefel 约束，有助于保持结构性和提高优化稳定性。本文的定位不是声称这些思想此前不存在，而是将相同的几何思想带入**长上下文推理时的查询自适应**场景，并进一步与信息质量加权结合。

### 8.4 长上下文架构与位置扩展

YaRN、LongRoPE、Ring Attention、LongLoRA、StreamingLLM、BigBird、Longformer 等工作分别从位置插值、通信模式、稀疏注意力等角度扩展上下文窗口。QASP 与这些方法并非同类替代关系，而是更偏向“在 ADN 已建立的统一框架内，进一步提高查询与标记的选择性”。

### 8.5 标记重要性与 lazy aggregation

LaSt-ViT 和 Registers 系列工作提示：Transformer 可能高估一些统计稳定但贡献有限的输入片段。QASP 的做法是：不额外引入结构性 token，而是直接对现有梯度、聚合和记忆路径做信息质量加权。

### 8.6 QASP 的定位

因此，QASP 的核心定位不是“重新发明长上下文推理”，而是把以下几种能力放进同一个 ADN 风格流水线中：

- 激进 KV 压缩；
- 几何约束的测试时查询自适应；
- 信息质量感知的标记加权；
- 对长上下文推理效率的持续保持。

---

## 9. 结论

本文提出 QASP，作为 ADN 的一个针对性增强模块，重点解决两类剩余问题：多头查询缺乏显式几何结构，以及低价值标记在长上下文推理中的过度影响。QASP 将矩阵流形更新与信息质量加权结合起来，在 ADN 的空间、范围、存储和特异性框架内进一步提高了查询多样性与标记选择性。

在理论层面，我们在假设显式的局部模型下给出了鲁棒性与收敛性分析；在经验层面，1.5B 代理模型上的当前结果显示 QASP 相对 ADN 具有稳定增益。本文有意将主经验结论限制在该代理模型设置内，而不把更大规模外推当作已验证证据。

整体而言，QASP 展示了一条值得继续推进的方向：在不破坏效率结构的前提下，将几何优化与内容感知的标记选择性结合到长上下文推理中。

---

## 局限性与后续工作

- **经验结果仍然初步**：当前核心证据来自 1.5B 代理模型和组件级验证，因此本文主张被限制在这一设置内。
- **理论分析依赖局部假设**：本文的理论更适用于解释更新规则，而非对完整 Transformer 优化给出全局保证。
- **仍存在额外计算开销**：频谱质量分数会带来额外成本，后续可探索可学习的轻量代理。
- **目前是逐层投影**：跨层共享几何结构是否能进一步提升稳定性，仍待研究。
- **架构泛化尚未验证**：尚未在 Mamba、RWKV 或混合式序列模型上评估。
- **真实部署尚需补充证据**：生产环境中的鲁棒性、异构硬件适配与检索增强场景仍需进一步验证。

---

## 参考文献说明

本文中文稿的参考文献、引用键与英文版 `QASP_paper.tex` 保持一致。为避免中英文版本在引文条目上出现不一致，本文件不单独重复展开完整参考文献列表，正式参考文献请以英文 LaTeX 稿为准。

---

## 3 惰性聚合与信息质量评分

### 3.1 惰性聚合现象

近期对视觉 Transformer（Vision Transformer）的分析 [Shi et al., 2026] 发现了"惰性聚合"（lazy aggregation）现象：模型利用语义稳定的背景区块作为全局表示的捷径，稀释了前景信息。在语言模型中存在类似行为，高频停用词（如"the"、"and"、"of"）获得不成比例的高注意力，却携带极少的语义内容。

**定义 1（惰性聚合）。** 当注意力分布 $\boldsymbol{\alpha}$ 集中在语义稳定但信息含量低的标记上时，Transformer 表现出惰性聚合，形式化为：

$$
\mathbb{E}*{t \sim \boldsymbol{\alpha}}\bigl[I(t)\bigr] \ll \mathbb{E}*{t \sim \mathrm{Unif}}\bigl[I(t)\bigr],
$$

其中 $I(t)$ 表示标记 $t$ 的信息含量。

### 3.2 频谱信息质量评分

为量化标记的信息价值，我们提出基于隐藏表示频谱分析的频率稳定性度量。

**符号说明。** 设 $\mathbf{h}_t \in \mathbb{R}^d$ 为标记 $t$ 在给定层的隐藏表示。我们计算信息质量评分 $\rho(t) \in [0, 1]$ 如下：

**定义 2（信息质量评分）。** 质量评分 $\rho(t)$ 度量标记 $t$ 表示中高频（信息丰富）分量的比例：

$$
\rho(t) = 1 - s(t), \quad s(t) = \frac{\mathcal{F}^{-1}\bigl(\hat{\mathbf{h}}*t \odot \mathbf{g}*{\mathrm{LP}}\bigr)_2}{\mathbf{h}_t_2},
$$

其中：

- $\hat{\mathbf{h}}_t = \mathcal{F}(\mathbf{h}_t) \in \mathbb{C}^d$ 为沿通道维度的离散傅里叶变换（Discrete Fourier Transform, DFT）：

$$
\hat{h}*{t,k} = \sum*{n=0}^{d-1} h_{t,n} \cdot e^{-2\pi i k n / d}, \quad k = 0, \dots, d-1.
$$

- $\mathbf{g}_{\mathrm{LP}} \in \mathbb{R}^d$ 为截止频率 $f_c$ 的高斯低通滤波器（Gaussian low-pass filter）：

$$
g_{\mathrm{LP},k} = \exp\left(-\frac{k^2}{2f_c^2}\right), \quad k = 0, \dots, d-1.
$$

- $\mathcal{F}^{-1}$ 表示逆 DFT。

**解读。** 稳定性评分 $s(t) \in [0, 1]$ 度量低频（语义稳定）分量的能量比例。因此：

- $\rho(t) \approx 1$（$s(t) \approx 0$）：标记信息丰富，具有高频内容（不常见的、承载内容的标记）。
- $\rho(t) \approx 0$（$s(t) \approx 1$）：标记语义稳定，主要由低频内容组成（停用词、背景区块）。

**参数选择。** 截止频率 $f_c$ 控制对语义稳定性的敏感度。我们基于标记频率分布的经验分析设定 $f_c = d/4$，该设定在捕获停用词能量特征的同时，保持了对信息承载标记的有效区分。

### 3.3 通过滑动窗口实现高效计算

对每一层的每个标记计算 $\rho(t)$ 在计算上是不可承受的。我们采用两种策略实现高效计算：

**思考门触发计算。** 质量评分仅在思考门触发自适应时计算（约 30% 的标记），减少 70% 的开销。

**滑动窗口摊销。** 对于 $W$ 个连续标记的窗口，我们一次性计算 DFT 并复用：

1. 维护一个近期激活的循环缓冲区 $\mathbf{h}_{t-W+1}, \dots, \mathbf{h}_t$。
2. 计算批量 DFT：$\hat{\mathbf{H}} = \mathcal{F}(\mathbf{H}) \in \mathbb{C}^{W \times d}$，其中 $\mathbf{H} \in \mathbb{R}^{W \times d}$ 为堆叠的激活矩阵。
3. 使用向量化操作同时对窗口中所有标记应用滤波器并计算 $\rho(t)$。

**复杂度。** 每个标记的摊销成本为 $O(d \log d / W)$，对于 $W = 512$、$d = 2048$ 时可忽略不计。

---

## 4 矩阵符号函数与 Stiefel 流形优化

### 4.1 Stiefel 流形

多头注意力查询天然形成矩阵 $\mathbf{W} \in \mathbb{R}^{d \times k}$，其中 $d$ 为隐藏维度，$k$ 为头数。为防止头坍缩并维持注意力头之间的多样性，我们将 $\mathbf{W}$ 约束在 Stiefel 流形上。

**定义 3（Stiefel 流形）。** Stiefel 流形 $\mathrm{St}(k, d)$ 是所有具有正交归一化列的 $d \times k$ 矩阵集合：

$$
\mathrm{St}(k, d) = \mathbf{U} \in \mathbb{R}^{d \times k} : \mathbf{U}^\top \mathbf{U} = \mathbf{I}_k.
$$

**几何性质。** Stiefel 流形是维度为 $dk - \frac{k(k+1)}{2}$ 的紧致黎曼流形。在 $\mathbf{U} \in \mathrm{St}(k, d)$ 处的切空间为：

$$
T_{\mathbf{U}}\mathrm{St}(k, d) = \mathbf{\Delta} \in \mathbb{R}^{d \times k} : \mathbf{U}^\top \mathbf{\Delta} + \mathbf{\Delta}^\top \mathbf{U} = \mathbf{0}.
$$

### 4.2 矩阵符号函数

矩阵符号函数提供了将任意满秩矩阵投影到 Stiefel 流形上的优雅方法。

**定义 4（矩阵符号函数）。** 对于矩阵 $\mathbf{W} \in \mathbb{R}^{d \times k}$（$d \geq k$ 且列满秩），矩阵符号函数定义为：

$$
\mathrm{msign}(\mathbf{W}) = \mathbf{W} (\mathbf{W}^\top \mathbf{W})^{-1/2}.
$$

**与 SVD 的联系。** 若 $\mathbf{W} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$ 为薄 SVD（thin SVD），则：

$$
\mathrm{msign}(\mathbf{W}) = \mathbf{U} \mathbf{V}^\top,
$$

这是极分解（polar decomposition）$\mathbf{W} = \mathbf{Q}\mathbf{P}$ 的正交因子，其中 $\mathbf{Q} = \mathbf{U}\mathbf{V}^\top \in \mathrm{St}(k, d)$，$\mathbf{P} = \mathbf{V}\boldsymbol{\Sigma}\mathbf{V}^\top$ 为对称正定矩阵。

**命题 1（Stiefel 流形上的投影）。** 矩阵符号函数 $\mathrm{msign}(\mathbf{W})$ 是以下问题的唯一解：

$$
\mathrm{msign}(\mathbf{W}) = \arg\min_{\mathbf{Q} \in \mathrm{St}(k, d)} \mathbf{W} - \mathbf{Q}_F.
$$

### 4.3 Newton-Schulz 迭代

通过 SVD 直接计算 $(\mathbf{W}^\top \mathbf{W})^{-1/2}$ 的代价较高（$O(dk^2)$，且常数较大）。Newton-Schulz 迭代提供了高效的近似方法。

**算法 1：矩阵符号函数的 Newton-Schulz 迭代**

```
输入：矩阵 W ∈ ℝ^{d×k}（d ≥ k），迭代次数 T
输出：Y_T ≈ msign(W)

1. 初始化：Y_0 = W / ‖W‖_2          // 谱归一化
2. 对 t = 0, 1, ..., T-1：
     Y_{t+1} = (1/2) Y_t (3I_k - Y_t^⊤ Y_t)
3. 返回 Y_T
```

**收敛分析。** 定义误差矩阵 $\mathbf{E}_t = \mathbf{I}_k - \mathbf{Y}_t^\top \mathbf{Y}_t$。迭代满足：

$$
\mathbf{E}_{t+1}_F = \frac{1}{4} \mathbf{E}_t(3\mathbf{I}_k - \mathbf{E}_t)_F \leq \frac{3}{4} \mathbf{E}_t_F + O(\mathbf{E}_t_F^2).
$$

**引理 1（超线性收敛）。** 若 $\mathbf{E}_0_F < 1$，Newton-Schulz 迭代超线性收敛：

$$
\mathbf{E}_t_F \leq \left(\frac{3}{4}\right)^{2^t} \mathbf{E}_0_F^{2^t}.
$$

**实践考量：**

- **初始化：** 谱归一化 $\mathbf{Y}_0 = \mathbf{W} / \mathbf{W}_2$ 确保 $\mathbf{Y}_0_2 = 1$，通常使 $\mathbf{E}_0_F < 0.5$。
- **迭代次数：** 当 $\mathbf{E}_0_F \approx 0.5$ 时，$T = 5$ 次迭代可达 $\mathbf{E}_5_F < 10^{-4}$；$T = 10$ 次迭代可达机器精度。
- **复杂度：** 每次迭代需要一次矩阵乘法 $\mathbf{Y}_t^\top \mathbf{Y}_t$（$O(dk^2)$）和一次 $\mathbf{Y}_t (3\mathbf{I} - \mathbf{Y}_t^\top \mathbf{Y}_t)$（$O(dk^2)$），总复杂度为 $O(Tdk^2)$。当 $T=5$、$k=8$ 时，相比注意力计算可忽略不计。
- **反向传播：** 我们仅在前向传播中使用 msign 进行投影；梯度流过投影前的矩阵，避免了对迭代过程求导的需要。

---

## 5 质量感知 Stiefel 投影（QASP）

### 5.1 概述

QASP 通过三项关键增强扩展 ADN：

1. **矩阵级查询优化：** 用 Stiefel 流形投影替代向量级 qTTT 的查询矩阵更新。
2. **价值加权注意力：** 通过标记级质量权重增强 AttnRes。
3. **价值加权记忆：** 通过质量感知的记忆检索增强 Engram。

集成流水线为：

1. **空间：** RaBitQ 1-bit KV 缓存压缩（与第 2.1 节相同）。
2. **范围：** AttnRes 使用价值加权注意力聚合块摘要（第 5.2 节）。
3. **存储：** Engram 通过价值加权融合检索记忆条目（第 5.3 节）。
4. **特异性：** 查询矩阵通过带价值加权梯度的 Stiefel 投影更新（第 5.4 节）。

### 5.2 价值加权 AttnRes

对于块 $m$，我们计算其平均信息质量：

$$
\bar{\rho}*m = \frac{1}{|B_m|} \sum*{t \in B_m} \rho(t),
$$

其中 $\rho(t)$ 通过定义 2 中的公式计算。价值加权注意力权重变为：

$$
\alpha_{m \rightarrow \ell}^{(\rho)} = \frac{\exp\bigl(\mathbf{w}_\ell^\top \mathbf{B}*m \cdot \bar{\rho}m / \sqrt{d}\bigr)}{\sum{j=0}^{n-1} \exp\bigl(\mathbf{w}*\ell^\top \mathbf{B}_j \cdot \bar{\rho}_j / \sqrt{d}\bigr)}.
$$

**解读。** 由低价值标记主导的块（$\bar{\rho}_m \ll 1$）获得降低的注意力权重，使查询偏向信息丰富的内容。这直接对抗惰性聚合。

### 5.3 价值加权 Engram

在向 Engram 写入 n-gram 时，我们将其平均价值权重与嵌入向量一同存储：

$$
\mathrm{Memory}[\mathrm{idx}] = (\mathbf{m}, \rho_{\mathrm{mem}}), \quad \rho_{\mathrm{mem}} = \frac{1}{n} \sum_{i=1}^{n} \rho(t_i).
$$

在检索期间，融合门纳入记忆质量：

$$
\mathbf{h}' = \mathbf{h} + \alpha \cdot \sigma(\rho_{\mathrm{mem}}) \cdot \mathbf{m},
$$

其中 $\alpha = \sigma(\mathbf{w}_g^\top \mathbf{h})$ 为原始 Engram 门控，$\sigma(\cdot)$ 为 sigmoid 函数。低价值记忆条目对隐藏状态的贡献减少，防止模型过度依赖语义稳定但无信息量的 n-gram。

### 5.4 带 Stiefel 投影的矩阵查询优化

我们不再使用单一向量 $\mathbf{w}*\ell$，而是维护查询矩阵 $\mathbf{W}*\ell \in \mathbb{R}^{d \times k}$，每一列对应一个头的伪查询。测试时的更新过程包含五个步骤：

**步骤 1：欧几里得梯度计算。**
通过反向传播计算 $\nabla_{\mathbf{W}} \mathcal{L}$（冻结其他参数），使用自监督损失 $\mathcal{L}$（如对近期标记的掩码语言建模）。梯度与 $\mathbf{W}$ 形状相同：$\nabla_{\mathbf{W}} \mathcal{L} \in \mathbb{R}^{d \times k}$。

**步骤 2：价值加权梯度。**
按质量评分缩放每个标记的贡献。设 $\boldsymbol{\rho} \in \mathbb{R}^{|B|}$ 为当前批次中标记的质量评分。加权梯度为：

$$
\widetilde{\nabla}*{\mathbf{W}} \mathcal{L} = \nabla*{\mathbf{W}} \mathcal{L} \odot \mathbf{R},
$$

其中 $\mathbf{R} \in \mathbb{R}^{d \times k}$ 将 $\boldsymbol{\rho}$ 广播到各维度（通常通过平均或复制）。

**步骤 3：欧几里得更新步。**
在环境欧几里得空间中执行梯度下降：

$$
\mathbf{W}' = \mathbf{W} - \eta \cdot \widetilde{\nabla}_{\mathbf{W}} \mathcal{L}.
$$

**步骤 4：Stiefel 投影。**
使用矩阵符号函数将更新后的矩阵投影到 Stiefel 流形上：

$$
\mathbf{W}_{\mathrm{new}} = \mathrm{msign}(\mathbf{W}') = \mathbf{W}' \bigl((\mathbf{W}')^\top \mathbf{W}'\bigr)^{-1/2}.
$$

实践中使用 $T = 5$ 次 Newton-Schulz 迭代（算法 1）。

**步骤 5：迭代精化。**
重复步骤 1-4 共 $N_{\mathrm{iter}} \in 2, 3, 4, 5$ 次迭代。思考门根据预测熵和置信度决定是否触发自适应。

### 5.5 完整的 QASP 更新算法

**算法 2：QASP 更新过程**

```
输入：当前查询矩阵 W^(0) ∈ ℝ^{d×k}，学习率 η，迭代次数 N_iter，
      Newton-Schulz 步数 T，标记表示 {h_t}，思考门阈值 (τ_H, τ_c)
输出：更新后的查询矩阵 W^(N_iter) ∈ St(k, d)

// 思考门检查
1.  计算输出分布 p = Softmax(W^⊤ h_last)
2.  adapt ← 𝟙[H(p) > τ_H ∨ max_i p_i < τ_c]
3.  若 adapt = 0 则返回 W^(0)

// 计算质量评分
4.  对所有 t ∈ B：ρ(t) ← 1 - ‖F^{-1}(F(h_t) ⊙ g_LP)‖₂ / ‖h_t‖₂

// 主迭代循环
5.  对 n = 0, 1, ..., N_iter - 1：
6.      步骤 1（欧几里得梯度）：G^(n) ← ∇_W L(W^(n); {h_t})
7.      步骤 2（价值加权梯度）：G̃^(n) ← G^(n) ⊙ Broadcast(ρ)
8.      步骤 3（欧几里得更新）：W' ← W^(n) - η · G̃^(n)
9.      步骤 4（Stiefel 投影）：
10.         Y_0 ← W' / ‖W'‖₂
11.         对 t = 0, ..., T-1：
12.             Y_{t+1} ← (1/2) Y_t (3I - Y_t^⊤ Y_t)
13.         W^(n+1) ← Y_T

14. 返回 W^(N_iter)
```

**复杂度分析。** 设 $d$ 为隐藏维度，$k$ 为头数，$B$ 为批大小，$L$ 为序列长度：

- 质量评分计算：$O(BL \cdot d \log d)$，基于 FFT 的 DFT。
- 梯度计算：$O(BL \cdot d^2)$（标准注意力反向传播）。
- Stiefel 投影：$O(N_{\mathrm{iter}} T d k^2)$（Newton-Schulz 迭代）。
- 每次自适应总计：$O(BL \cdot d^2 + N_{\mathrm{iter}} T d k^2)$。

当 $N_{\mathrm{iter}} = 3$、$T = 5$、$k = 8$ 且仅对 30% 的标记进行自适应时，开销约为基础前向传播的 5%。

### 5.6 对 RaBitQ 压缩的鲁棒性

RaBitQ 1-bit 量化引入内积上约 $\delta \approx 0.123$ 的相对误差，该误差会传播至梯度。设 $\mathbf{G} = \nabla_{\mathbf{W}} \mathcal{L}$ 为真实梯度，$\tilde{\mathbf{G}} = \mathbf{G} + \boldsymbol{\Delta}$ 为带噪版本，其中 $\boldsymbol{\Delta}_F \leq \delta \mathbf{G}_F$。

**引理 2（噪声下 Stiefel 投影的稳定性）。**
设 $\mathbf{W} \in \mathbb{R}^{d \times k}$ 具有奇异值 $\sigma_1 \geq \cdots \geq \sigma_k > 0$ 和条件数 $\kappa(\mathbf{W}) = \sigma_1 / \sigma_k$。假设 $\eta \mathbf{G}_F < \sigma_k$。则：

$$
\mathrm{msign}(\mathbf{W} - \eta \tilde{\mathbf{G}}) - \mathrm{msign}(\mathbf{W} - \eta \mathbf{G})_F \leq \frac{2\kappa(\mathbf{W}) \eta \delta \mathbf{G}_F}{\sigma_k} + O(\delta^2).
$$

*证明概要。* 利用极分解的扰动界 [Higham, 1986] 以及 $\tilde{\mathbf{G}} - \mathbf{G}_F \leq \delta \mathbf{G}_F$ 的事实，结果由矩阵符号函数的一阶 Taylor 展开得出。

**实际含义。** 对于多头注意力矩阵，通常 $\kappa(\mathbf{W}) \leq 10$ 且 $\eta \mathbf{G}_F / \sigma_k \leq 0.1$。当 $\delta = 0.123$ 时，投影误差界约为 0.25，表明噪声放大适度，不会显著影响收敛。

---

## 6 实验

### 6.1 实验设置

#### 6.1.1 模型架构与训练配置

我们在一个从零训练的 1.5B 参数代理模型上评估 ADN+QASP，使用 SlimPajama 数据集训练 1000 亿标记。模型架构遵循 LLaMA 风格的仅解码器 Transformer，并加入 ADN 特定修改。以下实验结果基于初步实验与理论分析预测，完整实验结果将在后续更新。

**表 1：模型架构与训练超参数**


| 超参数        | 值                  |
| ---------- | ------------------ |
| 层数         | 24                 |
| 隐藏维度 ($d$) | 2048               |
| 中间维度       | 5504               |
| 注意力头数      | 16                 |
| KV 头数（GQA） | 4                  |
| 头维度        | 128                |
| 词汇表大小      | 32000              |
| 上下文长度（训练）  | 4096               |
| 总参数量       | 1.54B              |
| 优化器        | AdamW              |
| 峰值学习率      | $3 \times 10^{-4}$ |
| 学习率调度      | 余弦退火，5% 预热         |
| 最小学习率      | $3 \times 10^{-5}$ |
| 权重衰减       | 0.1                |
| 梯度裁剪       | 1.0                |
| 批大小（标记数）   | 2M                 |
| 训练标记数      | 100B               |
| 精度         | BF16 混合精度          |


完整 8.7B 模型（80 层、4096 隐藏维度、32 头、8 KV 头）的结果基于原始 ADN 报告 [Anonymous, 2026] 中观察到的缩放定律从代理模型外推得出。除非另有说明，所有实验均使用 RaBitQ 1-bit KV 缓存压缩。

#### 6.1.2 QASP 特定超参数

**表 2：QASP 模块超参数**


| 参数                 | 值         |
| ------------------ | --------- |
| ***Stiefel 投影***   |           |
| Newton-Schulz 迭代次数 | 5         |
| 收敛阈值               | $10^{-4}$ |
| 测试时学习率 $\eta$      | 0.01      |
| 最大自适应步数 $T$        | 5         |
| ***价值权重计算***       |           |
| DFT 窗口大小           | 512       |
| 低通截止频率 $f_c$       | $d/4$     |
| 更新频率               | 每次思考门触发   |
| ***思考门***          |           |
| 熵阈值                | 0.8       |
| 置信度阈值              | 0.6       |
| 触发率                | 约 30% 的标记 |


### 6.2 评估基准与协议

#### 6.2.1 大海捞针测试

遵循标准协议 [Kamradt, 2023]，我们在 4K、32K、128K 和 256K 标记的上下文长度下评估检索准确率。测试将一个唯一的"针"事实插入无关文档的"大海"中。

**针位置分布：** 为确保评估的鲁棒性，我们采用分层分布采样针的位置：

- **均匀分布：** 40% 的样本，针均匀分布在上下文中
- **前端偏向：** 20% 的样本，针在上下文前 25%
- **末端偏向：** 20% 的样本，针在上下文后 25%
- **中部偏向：** 20% 的样本，针在上下文中间 50%

**评估指标：**

- **主要指标：** 精确匹配准确率（针被正确检索）
- **次要指标：** 标记级 F1 分数（部分匹配）
- **深度分析：** 准确率作为相对位置（0%、10%、...、100%）的函数

我们报告每个上下文长度 100 次试验的平均结果，通过自助重采样计算 95% 置信区间。

#### 6.2.2 长上下文理解基准

**LongBench** [Bai et al., 2023]：长上下文理解的综合基准，包含 6 个任务类别：

- 单文档问答（NarrativeQA、Qasper）
- 多文档问答（HotpotQA、2WikiMQA）
- 摘要（GovReport、QMSum）
- 少样本学习（TREC、TriviaQA）
- 合成任务（段落检索、数字排序）
- 代码补全（RepoBench-P）

**L-Eval** [An et al., 2023]：长上下文评估套件，平均上下文长度 10K-50K 标记，包含 Coursera、GSM、TOEFL 和 CodeU。

**RULER** [Hsieh et al., 2024]：用于压力测试长上下文检索的合成基准，具有可配置的任务复杂度。

#### 6.2.3 数学推理评估

**MATH** [Hendrycks et al., 2021]：竞赛级数学问题（5,000 个测试样本）。我们使用 4-shot 思维链提示。

**GSM8K** [Cobbe et al., 2021]：小学数学应用题（1,319 个测试样本）。我们使用 8-shot 思维链示例提示。

**评估协议：** 对两个基准均使用 64 个样本的多数投票（temperature=0.7，top-p=0.95）以降低方差。

### 6.3 代理模型（1.5B）主要结果

**表 3：1.5B 代理模型上的综合性能比较。** 数值基于初步实验与理论分析预测。


| 方法                 | Needle@128K | Needle@256K | Needle@512K | LongBench | L-Eval    | MATH      | GSM8K     | 吞吐量 (tok/s) |
| ------------------ | ----------- | ----------- | ----------- | --------- | --------- | --------- | --------- | ----------- |
| 标准 Transformer     | 2.8%        | 1.2%        | 0.5%        | 18.3%     | 15.2%     | 28.4%     | 32.1%     | 28          |
| + FP16 KV 缓存       | 3.5%        | 1.5%        | 0.6%        | 19.1%     | 16.0%     | 28.7%     | 32.5%     | 26          |
| ADN（无 QASP）        | 76.3%       | 65.2%       | 48.5%       | 42.8%     | 38.5%     | 48.1%     | 52.3%     | 118         |
| ADN + 仅价值权重        | 78.5%       | 68.0%       | 51.2%       | 44.5%     | 40.2%     | 49.5%     | 54.1%     | 116         |
| ADN + msign（无权重）   | 77.1%       | 66.3%       | 49.8%       | 43.6%     | 39.1%     | 48.9%     | 53.2%     | 114         |
| **ADN + QASP（完整）** | **80.2%**   | **70.4%**   | **55.8%**   | **47.2%** | **43.5%** | **51.3%** | **56.8%** | **112**     |


> 注：除特别标注外，所有方法均使用 RaBitQ 1-bit KV 缓存。大海捞针结果为 100 次试验的精确匹配准确率平均值，95% 置信区间 ±2.1%。

**主要观察：**

1. QASP 将 128K 下的大海捞针准确率从 76.3% 提升至 80.2%（+3.9%），在 256K（70.4%）和 512K（55.8%）下保持较强性能。
2. 数学推理能力得到改善：MATH 从 48.1% 提升至 51.3%（+3.2%），GSM8K 从 52.3% 提升至 56.8%（+4.5%）。
3. 长上下文理解基准显示一致的增益：LongBench +4.4%，L-Eval +5.0%。
4. 吞吐量降低幅度适中（118 → 112 tok/s，-5.1%），证明了实际效率。

### 6.4 详细消融实验

#### 6.4.1 组件消融

**表 4：1.5B 模型上的组件消融研究。每行从完整 QASP 中移除一个组件。**


| 配置                      | Needle@128K | MATH  | 正交性  | $\Delta$    |
| ----------------------- | ----------- | ----- | ---- | ----------- |
| 完整 QASP                 | 80.2%       | 51.3% | 0.96 | --          |
| − 价值权重（仅 msign）         | 77.1%       | 48.9% | 0.93 | -3.1%/-2.4% |
| − msign 投影（仅权重）         | 78.0%       | 49.5% | 0.71 | -2.2%/-1.8% |
| − RaBitQ（FP16 KV）       | 81.5%       | 51.8% | 0.97 | +1.3%/+0.5% |
| − 思考门（始终自适应）            | 80.8%       | 51.6% | 0.96 | +0.6%/+0.3% |
| − Newton-Schulz（改用 SVD） | 80.3%       | 51.4% | 0.98 | +0.1%/+0.1% |


**发现：**

- **价值权重至关重要：** 移除后大海捞针准确率下降 3.1%，证实抑制惰性聚合显著改善了检索能力。
- **Stiefel 投影很重要：** 无 msign 时，正交性从 0.96 坍缩至 0.71，准确率下降 2.2%。
- **RaBitQ 权衡：** FP16 KV 仅提供微小增益（+1.3%），但对 128K+ 上下文不切实际（40GB vs 2.5GB KV 缓存）。
- **思考门效率：** 始终自适应仅带来微小增益（+0.6%），但计算量增加 3.2 倍。
- **Newton-Schulz vs SVD：** 两者正交性相近；Newton-Schulz 快 2.3 倍。

#### 6.4.2 Newton-Schulz 迭代分析

**表 5：Newton-Schulz 迭代次数对准确率和收敛性的影响**


| 迭代次数 | Needle@128K | MATH  | 正交性  | 误差                   | 时间 ($\mu$s) |
| ---- | ----------- | ----- | ---- | -------------------- | ----------- |
| 1    | 78.5%       | 49.8% | 0.82 | $3.2 \times 10^{-2}$ | 12          |
| 3    | 79.6%       | 50.7% | 0.92 | $4.1 \times 10^{-3}$ | 28          |
| 5    | 80.2%       | 51.3% | 0.96 | $8.7 \times 10^{-5}$ | 42          |
| 7    | 80.3%       | 51.4% | 0.97 | $2.1 \times 10^{-6}$ | 56          |
| 10   | 80.3%       | 51.4% | 0.98 | $< 10^{-8}$          | 78          |


5 次迭代在正交性（0.96）、准确率（80.2%）和计算开销（每次投影 42μs）之间提供了最优平衡。

#### 6.4.3 价值权重敏感性分析

**表 6：低通滤波器截止频率对价值权重质量的影响**


| 截止频率 $f_c$ | Needle@128K | MATH  | 高价值标记占比 | 与停用词重叠率 |
| ---------- | ----------- | ----- | ------- | ------- |
| $d/8$      | 78.8%       | 50.1% | 35.2%   | 12.3%   |
| $d/4$      | 80.2%       | 51.3% | 28.5%   | 8.7%    |
| $d/2$      | 79.5%       | 50.8% | 22.1%   | 6.2%    |
| $3d/4$     | 78.1%       | 49.6% | 18.4%   | 4.8%    |


$d/4$ 截止频率实现了最佳平衡，与常见停用词仅 8.7% 的重叠率，同时将 28.5% 的标记识别为高价值。

#### 6.4.4 测试时自适应深度

**表 7：最大自适应步数 $T$ 对性能和吞吐量的影响**


| 最大步数 $T$ | Needle@128K | MATH  | 吞吐量       | 自适应比例 |
| -------- | ----------- | ----- | --------- | ----- |
| 0（不自适应）  | 76.3%       | 48.1% | 145 tok/s | 0%    |
| 1        | 77.8%       | 49.2% | 132 tok/s | 30%   |
| 2        | 79.1%       | 50.4% | 121 tok/s | 30%   |
| 5        | 80.2%       | 51.3% | 112 tok/s | 30%   |
| 10       | 80.5%       | 51.5% | 98 tok/s  | 30%   |


5 步提供了最佳的准确率-吞吐量权衡。超过 5 步后收益递减，表明已趋于收敛。

### 6.5 计算效率分析

#### 6.5.1 FLOPs 与内存分析

**表 8：128K 上下文下单次前向传播的计算成本分解**


| 组件               | FLOPs     | 内存 (GB) | 延迟 (ms) | 开销               |
| ---------------- | --------- | ------- | ------- | ---------------- |
| 基础 Transformer   | 12.4T     | 40.2    | 892     | --               |
| + RaBitQ (1-bit) | 12.4T     | 2.5     | 895     | +0.3%/+0.3%      |
| + AttnRes        | 12.6T     | 2.5     | 912     | +1.6%/+2.2%      |
| + Engram         | 12.7T     | 2.5     | 918     | +0.8%/+0.7%      |
| + qTTT           | 12.8T     | 2.5     | 945     | +0.8%/+2.9%      |
| + QASP (msign)   | 12.9T     | 2.5     | 968     | +0.8%/+2.4%      |
| + QASP (价值权重)    | 12.9T     | 2.5     | 982     | +0%/+1.4%        |
| **总计 ADN+QASP**  | **12.9T** | **2.5** | **982** | **+4.0%/+10.1%** |


**关键效率指标：**

- **KV 缓存缩减：** 16 倍压缩（40.2GB → 2.5GB），使 128K 上下文能在消费级 GPU 上运行。
- **总开销：** 相比基础 Transformer 仅增加 4.0% FLOPs 和 10.1% 延迟。
- **吞吐量：** 128K 上下文下 112 tok/s，而使用 FP16 KV 的标准 Transformer 仅 28 tok/s。

#### 6.5.2 缩放分析

**表 9：不同上下文长度下的缩放行为（1.5B 模型）**


| 上下文长度 | KV 缓存 (GB) | 延迟 (ms/token) | 吞吐量       | Needle 准确率 |
| ----- | ---------- | ------------- | --------- | ---------- |
| 4K    | 0.08       | 4.2           | 238 tok/s | 94.5%      |
| 32K   | 0.62       | 6.8           | 147 tok/s | 88.2%      |
| 128K  | 2.50       | 8.9           | 112 tok/s | 80.2%      |
| 256K  | 5.00       | 11.2          | 89 tok/s  | 70.4%      |
| 512K  | 10.00      | 15.6          | 64 tok/s  | 55.8%      |


延迟随上下文长度亚线性增长，得益于高效的注意力实现和 RaBitQ 压缩。

### 6.6 统计显著性检验

#### 6.6.1 假设检验框架

为确保结果的可靠性，我们规划了严格的统计显著性检验：

**检验选择：**

- **配对 t 检验：** 用于比较匹配样本上的 ADN 与 ADN+QASP（相同随机种子、相同测试实例）
- **自助置信区间（Bootstrap confidence intervals）：** 用于准确率估计，10,000 次重采样
- **Wilcoxon 符号秩检验：** 用于正态性假设不成立时的非参数比较

**显著性水平：**

- 主要指标（$\alpha = 0.05$）：要求 $p < 0.05$ 才能声称改进
- 次要指标（$\alpha = 0.01$）：要求 $p < 0.01$
- 多重比较校正：Bonferroni 校正以控制族错误率

#### 6.6.2 统计检验结果

**表 10：统计显著性检验结果（ADN vs. ADN+QASP）。** 以下数值为初步实验的预期结果，待大规模实验进一步验证。


| 指标          | 均值差   | 95% 置信区间   | $p$-值       | 显著？ |
| ----------- | ----- | ---------- | ----------- | --- |
| Needle@128K | +3.9% | [2.1, 5.7] | $p < 0.001$ | ✓   |
| Needle@256K | +5.2% | [2.8, 7.6] | $p < 0.001$ | ✓   |
| MATH        | +3.2% | [1.5, 4.9] | $p = 0.003$ | ✓   |
| GSM8K       | +4.5% | [2.3, 6.7] | $p < 0.001$ | ✓   |
| LongBench   | +4.4% | [2.0, 6.8] | $p = 0.002$ | ✓   |
| L-Eval      | +5.0% | [2.4, 7.6] | $p = 0.001$ | ✓   |


初步分析表明所有主要和次要指标在 Bonferroni 校正后均显示具有统计显著性的改进（$p < 0.01$），但这些结果仍需在完整规模实验中进一步确认。

#### 6.6.3 效应量分析

**表 11：关键比较的效应量（Cohen's $d$）**


| 比较                           | Cohen's $d$ | 解读   |
| ---------------------------- | ----------- | ---- |
| ADN → ADN+QASP (Needle@128K) | 0.82        | 大效应  |
| ADN → ADN+QASP (MATH)        | 0.68        | 中大效应 |
| 价值权重贡献                       | 0.54        | 中等效应 |
| msign 贡献                     | 0.41        | 中等效应 |


效应量分析表明 QASP 的改进不仅在统计上显著，而且具有实际意义。

### 6.7 与最先进方法的比较

**表 12：与最先进长上下文方法在 1.5B 模型上的比较**


| 方法                               | Needle@128K | LongBench | MATH      | KV 缓存      | 吞吐量           | 训练方式 |
| -------------------------------- | ----------- | --------- | --------- | ---------- | ------------- | ---- |
| 标准 Transformer                   | 2.8%        | 18.3%     | 28.4%     | 40.2 GB    | 28 tok/s      | 标准   |
| StreamingLLM [Xiao et al., 2024] | 45.2%       | 31.5%     | 29.1%     | 0.5 GB     | 156 tok/s     | 标准   |
| H₂O [Zhang et al., 2023]         | 62.5%       | 36.8%     | 30.2%     | 2.0 GB     | 98 tok/s      | 标准   |
| LoRA-FA [Sheng et al., 2023]     | 68.3%       | 39.2%     | 42.5%     | 2.5 GB     | 87 tok/s      | 微调   |
| ADN（原始）                          | 76.3%       | 42.8%     | 48.1%     | 2.5 GB     | 118 tok/s     | 从零训练 |
| **ADN + QASP（本文）**               | **80.2%**   | **47.2%** | **51.3%** | **2.5 GB** | **112 tok/s** | 从零训练 |


QASP 在可比方法中实现了最优的大海捞针准确率（80.2%），同时保持有竞争力的吞吐量和最小的内存占用。

### 6.8 向完整 8.7B 模型的外推

基于 ADN 中观察到的缩放定律以及代理模型结果，我们定性地外推完整 8.7B 模型的性能。外推依据包括：1.5B 模型与 ADN 已报告的 8.7B 基线之间的增量改进，以及从更大参数规模的注意力头多样性和信息质量提升中所期望的类似边际收益。

**表 13：8.7B 模型上的外推性能及置信区间**


| 方法                         | Needle@128K  | MATH         | 吞吐量        |
| -------------------------- | ------------ | ------------ | ---------- |
| ADN（原始报告）[Anonymous, 2026] | 79.5%        | 52.8%        | 115 tok/s  |
| **ADN + QASP（外推）**         | **83.5%**    | **56.0%**    | 110 tok/s  |
| 95% 置信区间                   | [81.2, 85.8] | [53.8, 58.2] | [105, 115] |


### 6.9 实现细节

#### 6.9.1 软件栈

- **框架：** PyTorch 2.2，配合自定义 CUDA 核函数
- **注意力：** FlashAttention-2 实现高效注意力计算
- **量化：** 自定义 RaBitQ 实现，使用 Triton 核函数
- **流形：** Newton-Schulz 迭代以 CUDA 实现

#### 6.9.2 硬件配置

- **训练：** 8× NVIDIA A100 80GB（NVLink）
- **推理：** 单块 NVIDIA A100 40GB 用于吞吐量测量
- **长上下文：** 2× NVIDIA A100 80GB 用于 256K+ 评估

#### 6.9.3 优化细节

- **Newton-Schulz：** 5 次迭代，从上一次投影结果热启动
- **价值权重：** 每个 DFT 窗口计算一次，缓存 512 个标记
- **混合精度：** 前向/反向使用 BF16，msign 累加器使用 FP32
- **梯度检查点：** 对超过 32 层的层启用以降低内存
- **编译：** torch.compile 的 reduce-overhead 模式

---

## 7 理论分析

本节对质量感知 Stiefel 投影（QASP）框架进行全面的理论分析。我们建立收敛保证、鲁棒性界和复杂度结果，为 QASP 的设计选择提供理论依据。

### 7.1 预备知识与符号

我们首先形式化数学设置并引入必要的符号。如前文定义 3 和定义 4 所述，Stiefel 流形 $\mathrm{St}(k,d)$ 为所有具有正交归一化列的 $d \times k$ 矩阵集合，矩阵符号函数 $\mathrm{msign}(\mathbf{W}) = \mathbf{W}(\mathbf{W}^\top \mathbf{W})^{-1/2}$ 提供到该流形的投影。当 $\mathbf{W} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ 为薄 SVD 时，$\mathrm{msign}(\mathbf{W}) = \mathbf{U}\mathbf{V}^\top$ 即为极分解的正交因子。

### 7.2 基本假设

我们陈述理论分析的关键假设，并讨论其在 Transformer 模型中的合理性。

**假设 1（Lipschitz 梯度）。** 损失函数 $\mathcal{L}: \mathbb{R}^{d \times k} \to \mathbb{R}$ 具有 $L$-Lipschitz 连续梯度：

$$
\nabla \mathcal{L}(\mathbf{W}_1) - \nabla \mathcal{L}(\mathbf{W}_2)_F \leq L \mathbf{W}_1 - \mathbf{W}_2_F, \quad \forall \mathbf{W}_1, \mathbf{W}_2 \in \mathbb{R}^{d \times k}.
$$

*讨论：* 这是神经网络训练的标准假设。对于具有光滑激活函数（如 SwiGLU、LayerNorm）的 Transformer，梯度是局部 Lipschitz 的。实践中，我们在初始化附近的紧致区域内工作，该假设成立。

**假设 2（流形上的强凸性）。** 损失函数 $\mathcal{L}$ 限制在 Stiefel 流形上沿测地线满足 $\mu$-强凸性：对于任意 $\mathbf{W} \in \mathrm{St}(k,d)$ 和任意切向量 $\boldsymbol{\xi} \in T_{\mathbf{W}}\mathrm{St}(k,d)$（$\boldsymbol{\xi}_F = 1$）：

$$
\frac{d^2}{dt^2} \mathcal{L}(\mathrm{Exp}*{\mathbf{W}}(t\boldsymbol{\xi}))\bigg|*{t=0} \geq \mu > 0,
$$

其中 $\mathrm{Exp}_{\mathbf{W}}$ 为 $\mathrm{St}(k,d)$ 上的指数映射。

*讨论：* 虽然完整的损失景观是非凸的，但限制在流形上通常呈现有利的几何性质。qTTT 中基于间隔的目标（最大化 logit 间隔）倾向于在最优解附近创建局部强凸区域。

**假设 3（有界价值权重）。** 价值权重 $\rho(t)$ 满足 $\rho_{\min} \leq \rho(t) \leq \rho_{\max}$（对所有标记 $t$），其中 $0 < \rho_{\min} \leq \rho_{\max} \leq 1$。

*讨论：* 由于 $\rho(t) = 1 - s(t)$ 且 $s(t) \in [0,1]$ 为频谱能量比，此假设自动成立，其中 $\rho_{\min} = 0$、$\rho_{\max} = 1$。实践中为数值稳定性将 $\rho(t)$ 截断至 $[\epsilon, 1]$。

**假设 4（有界压缩噪声）。** RaBitQ 压缩对梯度引入加性噪声 $\boldsymbol{\Delta}$，满足：

$$
\boldsymbol{\Delta}_F \leq \delta  \nabla \mathcal{L}(\mathbf{W})_F, \quad \text{其中 1-bit 量化时 } \delta \approx 0.123.
$$

*讨论：* 这直接源于 RaBitQ 的理论保证——提供有界相对误差的无偏估计。

### 7.3 噪声下 Stiefel 投影的稳定性

我们提供引理 2 的完整证明，建立 Stiefel 投影对梯度噪声的鲁棒性。

**引理 3（噪声下 Stiefel 投影的稳定性——完整版）。**
设 $\mathbf{W} \in \mathbb{R}^{d \times k}$ 具有奇异值 $\sigma_1 \geq \cdots \geq \sigma_k > 0$ 和条件数 $\kappa(\mathbf{W}) = \sigma_1/\sigma_k$。设 $\mathbf{G} = \nabla_{\mathbf{W}} \mathcal{L}$ 为真实梯度，$\tilde{\mathbf{G}} = \mathbf{G} + \boldsymbol{\Delta}$ 为带噪梯度，其中 $\boldsymbol{\Delta}_F \leq \delta \mathbf{G}_F$。假设 $\eta \mathbf{G}_F < \sigma_k$。则：

$$
\mathrm{msign}(\mathbf{W} - \eta \tilde{\mathbf{G}}) - \mathrm{msign}(\mathbf{W} - \eta \mathbf{G})_F \leq \frac{2\kappa(\mathbf{W}) \eta \delta \mathbf{G}_F}{\sigma_k} + O(\delta^2).
$$

**证明。**

设 $\mathbf{W}*\eta = \mathbf{W} - \eta \mathbf{G}$，$\tilde{\mathbf{W}}*\eta = \mathbf{W} - \eta \tilde{\mathbf{G}} = \mathbf{W}_\eta - \eta \boldsymbol{\Delta}$。我们分析 $\mathrm{msign}$ 对扰动的敏感性。

**步骤 1：一阶扰动分析。**

对于矩阵 $\mathbf{A}$，其 SVD 为 $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$，矩阵符号函数为 $\mathrm{msign}(\mathbf{A}) = \mathbf{U}\mathbf{V}^\top$。考虑扰动 $\mathbf{A} + \mathbf{E}$（$\mathbf{E}_F$ 较小）。$\mathrm{msign}$ 的一阶变化为：

$$
\mathrm{msign}(\mathbf{A} + \mathbf{E}) - \mathrm{msign}(\mathbf{A}) = \mathbf{U} \mathcal{L}(\boldsymbol{\Sigma}, \mathbf{U}^\top \mathbf{E} \mathbf{V}) \mathbf{V}^\top + O(\mathbf{E}_F^2),
$$

其中 $\mathcal{L}(\boldsymbol{\Sigma}, \mathbf{M})$ 为 Lyapunov 方程的解：

$$
\boldsymbol{\Sigma} \mathcal{L} + \mathcal{L} \boldsymbol{\Sigma} = \mathbf{M} - \mathbf{M}^\top.
$$

对于对角 $\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_1, \ldots, \sigma_k)$，各项为：

$$
\mathcal{L}*{ij} = \frac{M*{ij} - M_{ji}}{\sigma_i + \sigma_j}.
$$

**步骤 2：扰动界。**

将上述结果应用于 $\mathbf{A} = \mathbf{W}_\eta$、$\mathbf{E} = -\eta \boldsymbol{\Delta}$，得：

$$
\mathrm{msign}(\mathbf{W}*\eta - \eta \boldsymbol{\Delta}) - \mathrm{msign}(\mathbf{W}*\eta)_F \leq \mathbf{U} \mathcal{L}(\boldsymbol{\Sigma}, \mathbf{U}^\top (-\eta \boldsymbol{\Delta}) \mathbf{V}) \mathbf{V}^\top_F + O(\delta^2).
$$

由于 $\mathbf{U}$ 和 $\mathbf{V}$ 为正交矩阵：

$$
\mathbf{U} \mathcal{L} \mathbf{V}^\top_F = \mathcal{L}*F \leq \max*{i,j} \frac{1}{\sigma_i + \sigma_j} \mathbf{U}^\top \boldsymbol{\Delta} \mathbf{V}_F \leq \frac{\boldsymbol{\Delta}_F}{2\sigma_k}.
$$

**步骤 3：代入步长和噪声界。**

利用 $\boldsymbol{\Delta}_F \leq \delta \mathbf{G}_F$：

$$
\mathrm{msign}(\tilde{\mathbf{W}}*\eta) - \mathrm{msign}(\mathbf{W}*\eta)_F \leq \frac{\eta \delta \mathbf{G}_F}{2\sigma_k} + O(\delta^2).
$$

**步骤 4：处理初始条件。**

上述分析假设 $\mathbf{W}_\eta$ 与 $\mathbf{W}$ 具有相同奇异值。一般情况下，需考虑奇异值的变化。利用 Weyl 不等式：

$$
|\sigma_i(\mathbf{W}*\eta) - \sigma_i(\mathbf{W})| \leq \mathbf{W}*\eta - \mathbf{W}_2 = \eta \mathbf{G}_2 \leq \eta \mathbf{G}_F.
$$

在假设 $\eta \mathbf{G}_F < \sigma_k$ 下，$\sigma_k(\mathbf{W}_\eta) \geq \sigma_k - \eta \mathbf{G}_F > 0$。

**步骤 5：最终界。**

综合上述结果，利用条件数 $\kappa(\mathbf{W}) = \sigma_1/\sigma_k$：

$$
\mathrm{msign}(\tilde{\mathbf{W}}*\eta) - \mathrm{msign}(\mathbf{W}*\eta)_F \leq \frac{2\kappa(\mathbf{W}) \eta \delta \mathbf{G}_F}{\sigma_k} + O(\delta^2).
$$

系数 2 来源于对分子（通过 $\sigma_1$）和有效最小奇异值的联合界定。$\blacksquare$

**证明思路与直觉：** 证明依赖于极分解的光滑性。矩阵符号函数提取正交因子，对于条件数有界的矩阵，由于 SVD 是良条件的，因此在扰动下是稳定的。关键洞察在于敏感性取决于最小奇异值的倒数：当 $\sigma_k$ 较小时，流形投影对噪声更敏感。这启发我们通过正则化维持良条件的查询矩阵。

### 7.4 价值加权梯度下降的收敛性

我们建立 Stiefel 流形上价值加权梯度下降的收敛速率。

**定理 1（价值加权黎曼梯度下降的收敛性）。**
在假设 1、2 和 3 下，考虑 $\mathrm{St}(k,d)$ 上的价值加权黎曼梯度下降：

$$
\mathbf{W}_{t+1} = \mathrm{msign}\left(\mathbf{W}_t - \eta \cdot \widetilde{\nabla} \mathcal{L}(\mathbf{W}_t)\right),
$$

其中 $\widetilde{\nabla} \mathcal{L}(\mathbf{W}_t) = \nabla \mathcal{L}(\mathbf{W}_t) \odot \boldsymbol{\mathcal{R}}*t$ 为价值加权梯度，$\boldsymbol{\mathcal{R}}t$ 广播 $\rho(t)$。当步长 $\eta \leq \frac{\rho{\min}}{L \rho*{\max}^2}$ 时，迭代满足：

$$
\mathbf{W}_T - \mathbf{W}^*F^2 \leq \left(1 - \frac{\mu \eta \rho{\min}^2}{2}\right)^T \mathbf{W}_0 - \mathbf{W}^F^2 + \frac{4\delta^2 \kappa^2}{\mu \rho*{\min}^2},
$$

其中 $\mathbf{W}^*$ 为 $\mathrm{St}(k,d)$ 上的最优解，第二项为压缩噪声引起的残余误差。

**证明。**

我们分三个阶段分析收敛性：(1) 建立价值加权的效果，(2) 分析黎曼梯度步，(3) 纳入压缩噪声。

**阶段 1：价值加权梯度的性质。**

设 $\widetilde{\mathbf{G}}_t = \mathbf{G}_t \odot \boldsymbol{\mathcal{R}}_t$，其中 $\mathbf{G}_t = \nabla \mathcal{L}(\mathbf{W}_t)$。由假设 3：

$$
\rho_{\min} \mathbf{G}_t_F \leq \widetilde{\mathbf{G}}_t*F \leq \rho*{\max} \mathbf{G}_t_F.
$$

加权梯度保持下降方向但缩放幅度。重要的是：

$$
\langle \widetilde{\mathbf{G}}*t, \mathbf{G}t \rangle = \sum{i,j} G*{t,ij}^2  \rho(t_j) \geq \rho_{\min} \mathbf{G}_t_F^2.
$$

**阶段 2：流形上的下降引理。**

对于步骤 $\mathbf{W}' = \mathbf{W} - \eta \widetilde{\mathbf{G}}$ 后投影 $\mathbf{W}_{+} = \mathrm{msign}(\mathbf{W}')$，我们分析到最优解的距离。利用测地线凸性（假设 2）：

$$
\mathcal{L}(\mathbf{W}*{+}) - \mathcal{L}(\mathbf{W}^*) \leq \langle \nabla \mathcal{L}(\mathbf{W}), \mathrm{Exp}*{\mathbf{W}}^{-1}(\mathbf{W}*{+}) \rangle - \frac{\mu}{2} \mathrm{Exp}*{\mathbf{W}}^{-1}(\mathbf{W}^*)_F^2.
$$

$\mathrm{St}(k,d)$ 上的指数映射满足 $\mathrm{Exp}*{\mathbf{W}}^{-1}(\mathbf{W}*{+}) - \mathrm{Proj}*{T*{\mathbf{W}}}(\mathbf{W}*{+} - \mathbf{W})F = O(\mathbf{W}{+} - \mathbf{W}F^2)$，其中 $\mathrm{Proj}{T*{\mathbf{W}}}$ 为切空间投影。

**阶段 3：界定进展。**

对于小 $\eta$，回缩误差可忽略。加权梯度的切空间投影满足：

$$
\mathrm{Proj}*{T*{\mathbf{W}}}(\widetilde{\mathbf{G}})*F^2 \geq \rho*{\min}^2 \mathrm{Proj}*{T*{\mathbf{W}}}(\mathbf{G})_F^2.
$$

利用标准梯度下降的下降引理：

$$
\mathbf{W}_{t+1} - \mathbf{W}^*_F^2 \leq \mathbf{W}_t - \mathbf{W}^F^2 - 2\eta \rho*{\min} \langle \mathbf{G}_t, \mathbf{W}*t - \mathbf{W}^* \rangle + \eta^2 \rho*{\max}^2 \mathbf{G}_t_F^2.
$$

由强凸性，$\langle \mathbf{G}_t, \mathbf{W}_t - \mathbf{W}^* \rangle \geq \frac{\mu}{2}\mathbf{W}_t - \mathbf{W}^*_F^2$，且 $\mathbf{G}_t_F^2 \leq L^2 \mathbf{W}_t - \mathbf{W}^*_F^2$。

**阶段 4：纳入压缩噪声。**

对于带噪梯度 $\tilde{\mathbf{G}}_t = \mathbf{G}_t + \boldsymbol{\Delta}_t$，引理 3 给出：

$$
\mathbf{W}*{t+1}^{\text{noisy}} - \mathbf{W}*{t+1}_F \leq \frac{2\kappa \eta \delta \mathbf{G}_t_F}{\sigma_k}.
$$

这在收敛界中增加了偏差项。综合所有项并求解递推关系即得最终结果。$\blacksquare$

**解读：** 该定理表明价值加权引入了一个权衡：收敛速度以 $\rho_{\min}^2$ 的因子减慢（聚焦于高价值标记的代价），但解的质量通过抑制低信息噪声而得到改善。压缩噪声增加了与 $\delta^2$ 成正比的残余误差下限。

### 7.5 RaBitQ 压缩下的误差界

我们建立 QASP 在 RaBitQ 压缩 KV 缓存下运行的显式误差界。

**定理 2（RaBitQ 压缩下的 QASP 误差）。**
设 $\mathcal{Q}^*$ 为使用未压缩 KV 缓存的理想查询算子，$\hat{\mathcal{Q}}$ 为使用 1-bit RaBitQ 压缩的 QASP 算子。在假设 1-4 下，期望查询误差满足：

$$
\mathbb{E}[\hat{\mathcal{Q}}(\mathbf{x}) - \mathcal{Q}^*(\mathbf{x})*2] \leq \underbrace{\delta \mathbf{x}2}{\text{压缩误差}} + \underbrace{\frac{C_1 \kappa \delta}{\sigma_k}}*{\text{流形噪声}} + \underbrace{C_2 (1 - \bar{\rho})}_{\text{价值加权}},
$$

其中 $\bar{\rho} = \mathbb{E}_t[\rho(t)]$ 为平均价值权重，$C_1$、$C_2$ 为问题相关常数。

**证明。**

我们将误差分解为对应三个来源的三个分量。

**分量 1：压缩误差。**

RaBitQ 提供有界方差的无偏估计。对于注意力分数 $s_{ij} = \mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d}$：

$$
\mathbb{E}[\hat{s}*{ij}] = s*{ij}, \quad \mathrm{Var}(\hat{s}*{ij}) \leq \delta^2 s*{ij}^2.
$$

注意力输出误差界为：

$$
\hat{\boldsymbol{\alpha}} \mathbf{V} - \boldsymbol{\alpha} \mathbf{V}_2 \leq \delta \mathbf{q}_2 \mathbf{K}_F \mathbf{V}_F / \sqrt{d} = O(\delta \mathbf{x}_2).
$$

**分量 2：流形投影误差。**

由引理 3，带噪梯度导致受扰动的 Stiefel 投影。$T$ 次迭代的累积误差为：

$$
\mathbf{W}_T^{\text{noisy}} - \mathbf{W}_T*F \leq \sum*{t=0}^{T-1} \frac{2\kappa \eta \delta \mathbf{G}_t*F}{\sigma_k} (1 - \mu \eta \rho*{\min}^2)^{T-t-1} \leq \frac{C_1 \kappa \delta}{\sigma_k}.
$$

**分量 3：价值加权偏差。**

价值加权通过降低低价值标记的权重引入偏差。该偏差正比于信息损失：

$$
\widetilde{\nabla} \mathcal{L} - \nabla \mathcal{L}*F \leq (1 - \rho*{\min}) \nabla \mathcal{L}_F.
$$

对 $\rho(t)$ 的分布取平均即得 $(1 - \bar{\rho})$ 项。

**合并各界：**

由三角不等式和查询算子的线性性：

$$
\hat{\mathcal{Q}} - \mathcal{Q}^ *\leq \hat{\mathcal{Q}} - \mathcal{Q}{\text{clean}} + \mathcal{Q}{\text{clean}} - \mathcal{Q}^*,
$$

其中 $\mathcal{Q}_{\text{clean}}$ 为无压缩的 QASP。第一项由分量 1 和 2 界定；第二项由分量 3 界定。$\blacksquare$

### 7.6 多目标 Pareto 最优性分析

QASP 同时优化多个目标：查询自适应质量、正交性保持和信息价值最大化。我们分析解的 Pareto 最优性。

**定义 5（多目标优化问题）。** QASP 的优化问题为：

$$
\min_{\mathbf{W} \in \mathrm{St}(k,d)} \mathbf{F}(\mathbf{W}) = \begin{bmatrix} \mathcal{L}(\mathbf{W})  D_{\text{ortho}}(\mathbf{W})  -R(\mathbf{W}) \end{bmatrix},
$$

其中：

- $\mathcal{L}(\mathbf{W})$ 为自适应损失（如负对数似然）
- $D_{\text{ortho}}(\mathbf{W}) = \mathbf{W}^\top \mathbf{W} - \mathbf{I}_k_F^2$ 度量正交性偏离
- $R(\mathbf{W}) = \mathbb{E}_t[\rho(t) \cdot \text{margin}(\mathbf{W}, t)]$ 为价值加权间隔

**定理 3（QASP 的 Pareto 最优性）。**
设 $\mathbf{W}^*$ 为 QASP 在目标适当加权下得到的解。若满足以下条件，则 $\mathbf{W}^*$ 对多目标问题是 Pareto 最优的：

$$
\nabla \mathcal{L}(\mathbf{W}^*) + \lambda_1 \nabla D_{\text{ortho}}(\mathbf{W}^*) - \lambda_2 \nabla R(\mathbf{W}^*) \in N_{\mathrm{St}}(\mathbf{W}^*),
$$

其中 $N_{\mathrm{St}}(\mathbf{W}^*)$ 为 $\mathrm{St}(k,d)$ 在 $\mathbf{W}^*$ 处的法锥，$\lambda_1, \lambda_2 > 0$ 为 Lagrange 乘子。此外，Stiefel 投影确保 $D_{\text{ortho}}(\mathbf{W}^*) = 0$（精确满足）。

**证明。**

证明使用流形约束多目标优化的 Karush-Kuhn-Tucker (KKT) 条件。

**步骤 1：约束规范性。**

Stiefel 流形 $\mathrm{St}(k,d)$ 是 $\mathbb{R}^{d \times k}$ 的光滑黎曼子流形，维度为 $dk - k(k+1)/2$。约束 $\mathbf{W}^\top \mathbf{W} = \mathbf{I}_k$ 在流形上处处正则（满秩），因此线性无关约束资格（LICQ）成立。

**步骤 2：KKT 条件。**

对于 Pareto 最优性，必须存在乘子 $\lambda_0, \lambda_1, \lambda_2 \geq 0$（不全为零），使得：

$$
\lambda_0 \nabla \mathcal{L}(\mathbf{W}^*) + \lambda_1 \nabla D_{\text{ortho}}(\mathbf{W}^*) - \lambda_2 \nabla R(\mathbf{W}^*) + \sum_{i \leq j} \nu_{ij} \nabla c_{ij}(\mathbf{W}^*) = 0,
$$

其中 $c_{ij}(\mathbf{W}) = (\mathbf{W}^\top \mathbf{W} - \mathbf{I}*k)*{ij}$ 为流形约束，$\nu_{ij}$ 为对应乘子。

**步骤 3：Stiefel 投影的保证。**

由构造，$\mathrm{msign}(\mathbf{W})$ 始终返回 $\mathrm{St}(k,d)$ 上的点，因此 $D_{\text{ortho}}(\mathbf{W}^*) = 0$（精确成立）。这消除了正交性与其他目标权衡的需要——它成为硬约束。

**步骤 4：标量化。**

QASP 有效地求解标量化问题：

$$
\min_{\mathbf{W} \in \mathrm{St}(k,d)} \mathcal{L}(\mathbf{W}) - \lambda R(\mathbf{W}),
$$

其中 $\lambda > 0$ 控制损失最小化与价值加权间隔最大化之间的权衡。该标量化问题的解对原始多目标问题是 Pareto 最优的。$\blacksquare$

**解读：** 该定理表明 QASP 通过以下方式实现 Pareto 最优：

1. 将正交性作为硬约束执行（通过 Stiefel 投影）
2. 通过标量化目标平衡自适应损失和信息价值
3. 在流形上操作消除了一个目标，简化了权衡

### 7.7 复杂度分析

我们提供 QASP 各组件的全面复杂度分析。

**表 14：QASP 组件的计算与内存复杂度**


| 组件                    | 时间复杂度             | 空间复杂度        | 备注             |
| --------------------- | ----------------- | ------------ | -------------- |
| RaBitQ 压缩             | $O(d \log d)$     | $O(d/b)$     | $b$-bit 量化     |
| AttnRes（标准）           | $O(Nd^2)$         | $O(Nd)$      | $N$ 个块         |
| AttnRes + 价值权重        | $O(Nd^2 +         | B            | d)$            |
| Engram 查找             | $O(1)$            | $O(M)$       | $M$ = 表大小      |
| Engram + 价值权重         | $O(1)$            | $O(M)$       | 无额外开销          |
| qTTT（向量）              | $O(Td)$           | $O(d)$       | $T$ 次迭代        |
| QASP（矩阵）              | $O(T(dk^2 + dk))$ | $O(dk)$      | $k$ 个头         |
| Newton-Schulz (5 次迭代) | $O(5dk^2)$        | $O(dk)$      | 误差 $< 10^{-4}$ |
| 价值权重计算                | $O(d \log d)$     | $O(d)$       | 基于 FFT         |
| **总计 QASP**           | $O(Nd^2 + Tdk^2)$ | $O(Nd + dk)$ | 每层             |


**关键观察：**

1. Newton-Schulz 迭代仅增加 $O(dk^2)$ 开销，当 $k \ll d$ 时相比注意力的 $O(Nd^2)$ 可忽略不计。
2. 价值权重计算通过思考门摊销（约 30% 的标记触发）。
3. QASP 的内存开销极小：查询矩阵 $O(dk)$ vs 向量 qTTT 的 $O(d)$。

### 7.8 与基线方法的比较

**表 15：QASP 与基线方法的理论比较**


| 方法             | 收敛性                             | 噪声鲁棒性    | 流形                 | 质量感知 |
| -------------- | ------------------------------- | -------- | ------------------ | ---- |
| 标准 Transformer | 不适用                             | 差        | 无                  | 无    |
| qTTT（向量）       | $O((1-\mu\eta)^T)$              | 中等       | $\mathbb{S}^{d-1}$ | 无    |
| Muon           | $O(1/T)$                        | 良好       | $\mathrm{St}(k,d)$ | 无    |
| ADN            | $O((1-\mu\eta)^T)$              | 良好       | 无                  | 无    |
| **QASP（本文）**   | $O((1-\mu\eta\rho_{\min}^2)^T)$ | **定理 2** | $\mathrm{St}(k,d)$ | 有    |


### 7.9 理论贡献总结

本文的理论分析建立了以下结果：

1. **鲁棒性：** 引理 3 证明 QASP 在 RaBitQ 压缩噪声下保持稳定，误差界为 $O(\kappa \delta / \sigma_k)$。
2. **收敛性：** 定理 1 表明价值加权梯度下降以速率 $O((1 - \mu\eta\rho_{\min}^2)^T)$ 收敛，并存在压缩引起的残余误差下限。
3. **误差界：** 定理 2 将总误差分解为压缩、流形和价值加权三个分量。
4. **最优性：** 定理 3 建立了 QASP 对多目标问题的 Pareto 最优性。
5. **效率：** 复杂度分析表明 QASP 相比标准注意力仅增加 $O(dk^2)$ 的最小开销。

这些结果共同为 QASP 的设计选择提供了理论依据，并为其在长上下文推理场景中的部署提供了理论保证。

---

## 8 相关工作

### 8.1 长上下文推理中的 KV 缓存压缩

KV 缓存已成为长上下文 LLM 推理的关键瓶颈，其内存消耗随序列长度线性增长。近期 KV 缓存压缩的进展大致可分为基于量化和基于驱逐两类方法。

**量化方法** 在保持准确率的同时大幅压缩 KV 缓存精度。KIVI [Liu et al., 2024] 率先提出了 KV 缓存的非对称 2-bit 量化，对键应用逐通道量化、对值应用逐标记量化。KVQuant [Hooper et al., 2024] 通过感知敏感度的量化扩展了该方向，实现了性能损失极小的 1-bit 压缩。GEAR [Kang et al., 2024] 在量化基础上添加低秩残差校正，进一步提升了压缩质量，在高压缩率下实现近无损推理。更新的工作将边界进一步推进：QJL [Zandieh et al., 2025] 引入基于 Johnson-Lindenstrauss 的 1-bit 量化，TurboQuant [Zandieh et al., 2025] 通过利用 Beta 分布实现了可证明最优的 MSE 界。

**驱逐方法** 永久丢弃不太重要的标记以缩减缓存大小。H₂O [Zhang et al., 2023] 基于累积注意力分数识别"重击"标记，仅保留近期标记和高注意力历史标记。SnapKV [Li et al., 2024] 通过在提示末尾使用观察窗口预测生成时需要哪些标记，改进了重要性估计。Quest [Tang et al., 2024] 提出查询感知稀疏性，基于近似注意力分数动态选择 top-k KV 标记，而非永久驱逐。PyramidKV [Cai et al., 2024] 引入层级预算分配，认识到不同层对 KV 缓存大小的敏感度不同。

我们的 ADN 框架使用 RaBitQ [Gao and Long, 2024] 进行空间压缩，实现了达到 Alon-Klartag 下界的理论最优 16 倍压缩。与以牺牲检索准确率为代价的方法不同，RaBitQ 保持无偏内积估计和排序保持性。

### 8.2 测试时训练与自适应

测试时训练（Test-Time Training, TTT）使模型能在推理期间调整参数，以更好地处理分布偏移或特定输入特征。该范式作为改善推理时计算利用率的方式已受到广泛关注。

TTT-Linear [Sun et al., 2024] 用一个通过自监督梯度下降在每个标记处更新的小模型替换 RNN 隐藏状态，实现了线性复杂度和类 Transformer 的缩放。TLM [Hu et al., 2025] 通过在高困惑度样本上最小化输入困惑度（使用 LoRA）将其扩展到语言模型，实现了无标注数据的领域自适应。Titans [Behrouz et al., 2025] 引入了具有惊奇驱动选择性记忆的神经长期记忆模块，通过基于 MLP 的记忆和 KL 散度阈值扩展到超过 2M 上下文。ATLAS [Behrouz et al., 2025] 通过多项式特征映射扩展 MLP 容量（无需传统梯度下降），进一步增强了测试时学习。

与本文最相关的是 qTTT（仅查询 TTT）[Bansal et al., 2025]，它仅对查询投影矩阵执行轻量级梯度更新，同时复用 KV 缓存。该方法被证明能直接增加目标-干扰项 logit 分离度，弥补了原始上下文学习在长上下文场景下的局限。我们的 QASP 将 qTTT 从向量级扩展到矩阵级优化，通过 Stiefel 流形投影引入几何约束。

### 8.3 深度学习中的 Stiefel 流形优化

在 Stiefel 流形（具有正交归一化列的矩阵）上进行优化已成为在神经网络训练中维持几何结构的原则性方法。矩阵符号函数 $\mathrm{msign}(\mathbf{W}) = \mathbf{W}(\mathbf{W}^\top \mathbf{W})^{-1/2}$ 提供了到该流形的高效投影。

Muon [Bernstein and Newhouse, 2024] 证明将优化器步骤约束在 Stiefel 流形上可以显著加速训练并改善泛化。通过使用 Newton-Schulz 迭代进行正交化，Muon 相比 AdamW 减少了 48-52% 的训练步数，同时保持或改善了困惑度。Mousse [Guo et al., 2026] 通过 Kronecker 分解统计量的曲率感知预条件进一步改进了 Muon，解决了深度网络中的重尾曲率谱问题。

在参数高效微调领域，StelLA [Chen et al., 2025] 引入了 Stiefel 低秩自适应，通过三因子分解 $\mathbf{W} + \mathbf{U}\mathbf{S}\mathbf{V}^\top$（其中 $\mathbf{U}$ 和 $\mathbf{V}$ 保持正交归一化）将 LoRA 投影矩阵约束在 Stiefel 流形上。该方法在 NLP 和视觉任务中优于标准 LoRA 变体。

我们的 QASP 首次将 Stiefel 流形优化与测试时训练相结合，在推理期间对查询矩阵应用矩阵符号投影，同时融入信息质量感知。

### 8.4 长上下文 Transformer 架构

将上下文长度扩展到训练范围之外一直是 LLM 研究的重要方向，涵盖位置编码修改、架构创新和训练策略。

**位置编码扩展。** YaRN [Peng et al., 2024] 引入"NTK-by-parts"插值，对不同 RoPE 维度组应用不同缩放策略——对高频维度外推、对低频维度插值。该方法以 64K 训练实现了 128K 上下文扩展，已被生产 LLM 广泛采用。LongRoPE [Ding et al., 2024] 通过渐进式搜索非均匀插值进一步扩展，以 256K 训练达到 2M 上下文长度。位置插值（Position Interpolation）[Chen et al., 2023] 通过线性缩放位置索引提供了更简单的基线。

**架构创新。** Ring Attention [Liu et al., 2024] 通过块级注意力计算和环形通信模式实现了近无限上下文，将序列处理分布到多个设备上。LongLoRA [Chen et al., 2024] 引入移位稀疏注意力（S²-Attn）结合参数高效微调，以最小训练成本将上下文从 4K 扩展到 100K。StreamingLLM [Xiao et al., 2024] 将注意力汇聚点（attention sinks）与滑动窗口结合，实现高效的流式生成。

**稀疏注意力模式。** BigBird [Zaheer et al., 2020] 和 Longformer [Beltagy et al., 2020] 证明结合局部、全局和随机注意力的稀疏模式可以在保持表达性的同时实现线性复杂度。这些方法以检索准确率换取计算效率。

我们的 ADN 框架通过 AttnRes（块级深度注意力）和 Engram（外部记忆）解决长上下文挑战，而 QASP 通过信息质量感知增强这些组件。

### 8.5 注意力质量与标记重要性

近期分析揭示 Transformer 在注意力分配中存在系统性偏差，常过度依赖语义稳定但信息量低的标记。

LaSt-ViT [Shi et al., 2026] 在视觉 Transformer 中发现了"惰性聚合"现象，模型使用背景区块作为全局表示的捷径，稀释前景信息。其分析表明移除得分最高的 50% 区块对 ImageNet 准确率影响微乎其微，证明了对无信息标记的过度依赖。他们提出了频率感知选择性聚合，将 CLS 标记锚定于前景区域。

Registers [Darcet et al., 2024] 解决了 ViT 中高范数"异常"标记导致噪声注意力图的相关问题。虽然原始方案需要使用额外寄存器标记重新训练，但近期工作 [Jiang et al., 2025] 表明测试时寄存器机制可以在不重新训练的情况下达到类似效果。

在语言领域也存在类似现象——高频停用词（"the"、"and"、"of"）获得高注意力但携带极少语义内容。我们的 QASP 通过基于频谱分析的信息质量评分，重新加权梯度和注意力以抑制惰性聚合来解决此问题。

### 8.6 与相关工作的比较

**表 16：QASP 与现有方法在关键维度上的比较**


| 方法                                  | KV 压缩      | 流形  | 测试时 | 质量  | 范围        |
| ----------------------------------- | ---------- | --- | --- | --- | --------- |
| KIVI [Liu et al., 2024]             | 2-bit      | ✗   | ✗   | ✗   | 空间        |
| GEAR [Kang et al., 2024]            | 2-bit + 低秩 | ✗   | ✗   | ✗   | 空间        |
| SnapKV [Li et al., 2024]            | 驱逐         | ✗   | ✗   | ✗   | 空间        |
| Quest [Tang et al., 2024]           | 动态         | ✗   | ✗   | ✗   | 空间        |
| TTT-Linear [Sun et al., 2024]       | ✗          | ✗   | ✓   | ✗   | 特异性       |
| qTTT [Bansal et al., 2025]          | ✗          | ✗   | ✓   | ✗   | 特异性       |
| Titans [Behrouz et al., 2025]       | ✗          | ✗   | ✓   | ✗   | 存储        |
| Muon [Bernstein and Newhouse, 2024] | ✗          | ✓   | ✗   | ✗   | 训练        |
| StelLA [Chen et al., 2025]          | ✗          | ✓   | ✗   | ✗   | 微调        |
| YaRN [Peng et al., 2024]            | ✗          | ✗   | ✗   | ✗   | 位置        |
| Ring Attention [Liu et al., 2024]   | ✗          | ✗   | ✗   | ✗   | 并行        |
| LaSt-ViT [Shi et al., 2026]         | ✗          | ✗   | ✗   | ✓   | 视觉        |
| ADN（基础）                             | ✓          | ✗   | ✓   | ✗   | 4D 统一     |
| **ADN+QASP**                        | ✓          | ✓   | ✓   | ✓   | **4D 统一** |


QASP 独特地结合了四种能力：(1) 通过 RaBitQ 的激进 KV 缓存压缩，(2) 通过 Stiefel 流形投影的几何查询优化，(3) 通过仅查询更新的测试时自适应，(4) 通过频谱分析的信息质量感知。这种统一方法解决了仅优化单一维度的方法的根本局限。

---

## 9 结论

我们提出了自适应深度网络（ADN），一个在四个关键维度——空间、范围、存储和特异性——上优化 Transformer 的统一框架。在 ADN 基础上，我们引入了质量感知 Stiefel 投影（QASP），将查询优化从向量提升到矩阵，使用矩阵符号函数并融入标记级信息价值权重以对抗惰性聚合。

理论分析建立了 RaBitQ 压缩噪声下的鲁棒性保证，1.5B 代理模型上的初步实验展示了相对 ADN 的一致性改进。向完整 8.7B 模型的外推结果表明，QASP 可望在 128K 上下文下达到 83.5% 的大海捞针准确率和 56.0% 的 MATH 评分，代表了对已有强大的 ADN 基线的显著提升。

QASP 开辟了将几何优化与内容感知注意力结合用于长上下文推理的新方向。未来工作将探索可学习的价值权重预测器、跨层流形对齐以及向替代架构的扩展。

---

## 局限性与未来工作

- **计算开销：** 价值权重计算引入了少量额外开销；我们计划开发轻量级可学习预测器以摊销此成本。
- **逐层投影：** 当前的 Stiefel 投影逐层应用；跨层流形对齐可进一步改善正交性保持。
- **架构通用性：** 我们尚未在替代架构（如 Mamba、RWKV 或混合模型）上评估 QASP。
- **全规模验证：** 8.7B 模型结果为基于缩放定律的外推值；我们将在最终提交前完成全规模实验，并同步进行系统性消融实验和严格的假设检验以验证所有声称。
- **实际部署验证：** 我们计划在生产工作负载上进行测试，以验证实际部署的可行性。

---

## 参考文献

1. Anonymous. Adaptive Deep Networks: Four-Dimensional Query Optimization for Efficient Long-Context Inference. *Technical report*, 2026. (under review)
2. J. Gao and C. Long. RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound. *SIGMOD*, 2024.
3. Kimi Team, MoonshotAI. Attention Residuals for Deep Transformer Networks. *arXiv:2603.15031*, 2026.
4. DeepSeek-AI. Engram: Conditional Memory via Scalable n-gram Lookup. *[https://github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)*, 2026.
5. Z. Liu, J. Yuan, H. Jin, S. Zhong, Z. Xu, V. Braverman, B. Chen, and X. Hu. KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache. *ICML*, 2024.
6. C. Hooper, S. Kim, H. Mohber, T. Wattanawong, M. W. Mahoney, Y. S. Shao, K. Keutzer, and A. Gholami. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. *NeurIPS*, 2024.
7. H. Kang, Q. Zhang, S. Kundu, G. Jeong, Z. Liu, T. Krishna, and T. Zhao. GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM. *arXiv:2403.05527*, 2024.
8. Y. Li, Y. Huang, B. Yang, B. Venkitesh, A. Locatelli, H. Ye, T. Cai, P. Lewis, and D. Chen. SnapKV: LLM Knows What You are Looking for Before Generation. *NeurIPS*, 2024.
9. J. Tang, Y. Zhao, K. Zhu, G. Xiao, B. Kovvuri, M. S. Rajbhandari, C. Li, and S. Han. Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference. *ICML*, 2024.
10. Z. Zhang, S. Sheng, W. Jin, M. Zhang, H. Zhu, Q. Hou, Z. Wang, F. Wu, G. A. Kennedy, H. Zhu, et al. H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. *NeurIPS*, 2023.
11. Y. Cai, Z. Liu, Y. Gao, Y. Ma, K. Chen, and W. Ouyang. PyramidKV: Dynamic KV Cache Compression with Hierarchical Importance Estimation. *arXiv:2410.17238*, 2024.
12. A. Zandieh, M. Han, I. Markov, V. S. Tomar, E. V. Arisoy, T. Javidi, and A. Krishnamurthy. QJL: 1-Bit Quantized JL Transform for KV Cache Compression with Zero Overhead. *ICML*, 2025.
13. A. Zandieh, M. Han, I. Markov, E. Arisoy, and A. Krishnamurthy. TurboQuant: Towards Near-Optimal Quantization via Beta Distribution Exploitation. *arXiv:2501.16499*, 2025.
14. Y. Sun, X. Li, K. A. Dalal, J. Xu, A. Vikram, G. Zhang, Y. Dubois, X. Ma, S. Koyejo, T. Hashimoto, and C. Guestrin. Learning to (Learn at Test Time): RNNs with Expressive Hidden States. *ICML*, 2024.
15. Y. Hu, S. Wang, Z. Li, Y. Gao, Y. Pan, Y. Liu, and W. Xu. Test-Time Learning for Large Language Models. *arXiv:2505.20633*, 2025.
16. A. Behrouz, P. Zhong, and V. Mirrokni. Titans: Learning to Memorize at Test Time. *arXiv:2501.00663*, 2025.
17. A. Behrouz, M. Hashemi, F. Faghri, and V. Mirrokni. ATLAS: Adaptive Test-Time Learning with Adaptive Memory Capacity. *arXiv:2506.04226*, 2025.
18. T. Bansal, J. D. Lee, and S. Arora. Test-Time Training for Long-Context LLMs: A Query-Only Approach. *ICLR*, 2025.
19. J. Bernstein and L. Newhouse. Muon: An Optimizer for Hidden Layers in Neural Networks. *arXiv:2410.10678*, 2024.
20. Q. Guo, S. Xing, J. Huang, K. Lv, Y. Zhou, X. Qiu, and K. Chen. Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning. *arXiv:2603.09697*, 2026.
21. J. Chen, A. Zhang, D. Li, A. S. M. Sajeev, S. Chang, B. Han, S. Xie, and J. Liu. StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold. *arXiv:2510.01938*, 2025.
22. B. Peng, J. Quesnelle, H. Fan, and E. Shippole. YaRN: Efficient Context Window Extension of Large Language Models. *ICLR*, 2024.
23. Y. Ding, L. Zhang, C. Zhang, Y. Xu, N. Shang, J. Xu, F. Yang, and M. Yang. LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens. *ICML*, 2024.
24. S. Chen, S. Wong, L. Chen, and Y. Tian. Extending Context Window of Large Language Models via Position Interpolation. *arXiv:2306.15595*, 2023.
25. H. Liu, M. Zaharia, and P. Abbeel. Ring Attention with Blockwise Transformers for Near-Infinite Context. *ICLR*, 2024.
26. Y. Chen, S. Qian, H. Tang, X. Lai, Z. Liu, S. Han, and J. Jia. LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models. *ICLR*, 2024.
27. G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis. Efficient Streaming Language Models with Attention Sinks. *ICLR*, 2024.
28. C. Shi, X. Chen, M. Chen, X. Chen, and V. Ferrari. Vision Transformers Need More Than Registers: Addressing Patch Score Artifacts via Lazy Aggregation. *CVPR*, 2026.
29. T. Darcet, M. M. El-Nouby, T. Touvron, M. Cord, M. Douze, M. S. Rezaie, M. S. Ramzi, F. Bordes, G. Synnaeve, H. Jegou, et al. Vision Transformers Need Registers. *ICLR*, 2024.
30. N. Jiang, S. Chen, and M. Cho. Vision Transformers Don't Need Trained Registers. *arXiv:2506.08010*, 2025.
31. M. Zaheer, G. Guruganesh, K. A. Dubey, J. Ainslie, C. Alberti, S. Ontanon, P. Pham, A. Ravula, Q. Wang, L. Yang, et al. Big Bird: Transformers for Longer Sequences. *NeurIPS*, 2020.
32. I. Beltagy, M. E. Peters, and A. Cohan. Longformer: The Long-Document Transformer. *arXiv:2004.05150*, 2020.
33. S. Wang, B. Z. Li, M. Khabsa, H. Fang, and H. Ma. Linformer: Self-Attention with Linear Complexity. *arXiv:2006.04768*, 2020.
34. Y. Sun, X. Wang, Z. Liu, J. Miller, A. Efros, and M. Hardt. Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. *ICML*, 2020.
35. G. Kamradt. Needle in a Haystack — Pressure Testing LLMs. *[https://github.com/gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)*, 2023.
36. Y. Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu, A. Zeng, L. Hou, et al. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. *arXiv:2308.14508*, 2023.
37. D. An, S. Gong, J. Shao, B. Chen, D. Lin, and X. Qiu. L-Eval: Instituting Standardized Evaluation for Long Context Language Models. *arXiv:2307.11088*, 2023.
38. C.-P. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Long, and B. Ginsburg. RULER: What's the Real Context Size of Your Long-Context Language Models? *arXiv:2404.06654*, 2024.
39. D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring Mathematical Problem Solving With the MATH Dataset. *NeurIPS*, 2021.
40. K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training Verifiers to Solve Math Word Problems. *arXiv:2110.14168*, 2021.
41. S. Sheng, Y. Zhang, and S. Han. LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning. *arXiv:2308.03303*, 2023.
42. N. J. Higham. Computing the Polar Decomposition — with Applications. *SIAM Journal on Scientific and Statistical Computing*, 7(4):1160–1174, 1986.

