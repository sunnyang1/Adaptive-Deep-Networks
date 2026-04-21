# Adaptive Deep Networks: Four-Dimensional Query Optimization for Efficient Long-Context Inference

**Abstract.** Long-context transformers bottleneck on **KV memory**, **depth-wise information flow**, **static parametric knowledge**, and **fixed query maps at inference**. We study these issues through a unified lens—**query optimization along four axes**: space, scope, storage, and specificity—and instantiate them in **Adaptive Deep Networks (ADN)** by composing four modular stages: **RaBitQ** for compressed inner-product geometry (up to **16× fewer bits per dimension** than FP16 for 1-bit codes, under the IP estimators of [8, 9]), **Block Attention Residuals (AttnRes)** for $O(Nd)$ block summaries versus $O(Ld)$ full depth paths, **Engram** for hash-conditioned external memory, and **query-only test-time training (qTTT)** with polar-coordinate updates and an optional **Ponder Gate**. On our reported benchmarks, the full pipeline reaches **79.5%** needle-in-haystack accuracy at **128K** context and **69.0%** at **256K** (Table 5), **52.8%** on MATH at **8.7B** scale with query-only adaptation (Table 8), and **115** tokens/s in the setting of Table 4, alongside ablations that isolate each stage (Table 9). We release implementation details and hyperparameters needed for reproduction (§5.1). **Broader impact** considerations are discussed in §7. A companion theory paper (**Paper II / MATDO-E**) develops a continuous **$(R,M,T,E)$ resource model** and analyzes **context/compute walls** and **heterogeneous memory arbitrage** for the same four axes (§1.5; [21]).

**Keywords:** long-context language models; KV-cache compression; test-time adaptation; external memory; attention residuals; efficient inference.

---

## 1 Introduction

Scaling sequence length and deployment memory forces explicit trade-offs in **how queries access history, external knowledge, and adaptation signals**. We study these trade-offs through a unified **query-centric** formulation and report **end-to-end** measurements with **ablations** and **reproducibility** details (§5.1).

### 1.1 Problem setting: query accuracy as the core challenge

Transformer architectures [2], at their essence, are **hierarchical query-answering systems**. Each layer issues queries (Q) against keys (K) to retrieve relevant values (V). However, this query mechanism operates across **four distinct dimensions** that must all be optimized for accurate retrieval:

1. **Space (Vector Representation):** Queries and keys reside in high-dimensional vector spaces. The precision of similarity computation determines retrieval accuracy.
2. **Scope (Historical Access):** Queries attend to tokens across varying sequence lengths and layer depths. The effective context window determines accessible history.
3. **Storage (External Knowledge):** Queries are limited to parametric memory (weights) and transient context (KV cache), lacking access to scalable external knowledge stores.
4. **Specificity (Task Adaptation):** Standard queries are static, trained on the pretraining distribution. When inputs diverge, query specificity degrades.

Standard transformers suffer from query degradation across all four dimensions:

- **Spatial:** Full-precision vectors require prohibitive memory, forcing approximations.
- **Scope:** Fixed residual connections cause representation burial, limiting historical access [19].
- **Storage:** Knowledge is trapped in expensive parametric form; external memory is inaccessible.
- **Specificity:** Static queries degrade under distribution shift, long-tail inputs, and tasks outside the pretraining mixture (see, e.g., robustness and adaptation literature [4, 18]).

### 1.2 Overview: four-stage query optimization

We frame the problem as **four-dimensional query optimization**—how can we make the query mechanism more accurate and efficient across space, scope, storage, and specificity?

**Stage 1: Space Optimization (RaBitQ)**Before a query can retrieve anything, it must operate in a manageable space. RaBitQ [8, 9] optimizes the query space by:

- Applying a **structured random rotation** (e.g., randomized Hadamard / SRHT; $O(d\log d)$), in the Johnson–Lindenstrauss family [3], to spread energy before quantization
- Quantizing to $b$-bit representations with unbiased inner product preservation (under the estimator defined in [8, 9])
- Achieving up to **16× fewer bits per dimension than FP16** for 1-bit codes (end-to-end KV bytes also depend on which tensors are quantized, GQA layout, and overhead; §3.1.2)

This space optimization is the **enabling foundation**: without it, storing block representations for AttnRes and embedding tables for Engram would be prohibitively expensive.

**Stage 2: Scope Optimization (Block AttnRes)**With compressed space, we can afford to expand the query's field of view. Block Attention Residuals [19] optimize query scope by:

- Replacing fixed addition with learned softmax attention over $N$ block-level representations
- Enabling queries to selectively retrieve from any prior block
- Reducing memory from $O(Ld)$ to $O(Nd)$ while preserving expressivity

The query now has an **expanded historical horizon**, preventing representation burial.

**Stage 3: Storage Optimization (Engram)**Beyond the current sequence, queries can benefit from **externalized, reusable memory**. Engram instantiates the **storage** axis via conditional memory lookup. Its typical design provides:

- **O(1) deterministic lookup** into massive embedding tables via N-gram hashing
- **U-shaped scaling law** guiding optimal allocation between neural computation and static memory
- **Host memory offloading** with minimal inference overhead through deterministic addressing

The query now accesses **scalable external knowledge** beyond parametric limits.

**Stage 4: Specificity Optimization (qTTT)**Finally, given optimal space, scope, and storage, we optimize the query itself. Query-only Test-Time Training adapts queries during inference by:

- Reparameterizing queries in polar coordinates $(r, \theta)$
- Freezing magnitude $r$ and adapting direction $\theta$
- Maximizing logit margins through gradient-based optimization

The query becomes **task-specific**, improving retrieval precision when standard inference fails.

### 1.3 The Four-Dimensional Composition Principle

The four stages **compose as a pipeline** across the information hierarchy (we use $\circ$ for conceptual composition of stages, not literal multiplication of scalar costs):

$$
\text{Query Quality} = \underbrace{f_{\text{space}}}*{\text{RaBitQ}} \circ \underbrace{f*{\text{scope}}}*{\text{AttnRes}} \circ \underbrace{f*{\text{storage}}}*{\text{Engram}} \circ \underbrace{f*{\text{specificity}}}_{\text{qTTT}}
$$

**Critical Dependencies:**

- **Space → Scope:** Compressed representations make expanded scope affordable
- **Space → Storage:** 16× reduction enables economically viable embedding tables
- **Scope → Storage:** Historical context enables effective utilization of retrieved memory
- **Scope → Specificity:** Depth-wise context provides targets for query adaptation
- **Storage → Specificity:** External knowledge enriches adaptation signals
- **Specificity → Space:** Adaptive queries tolerate higher compression

**The Critical Insight:** RaBitQ is not merely a compression technique—it is the **query space optimizer that makes the entire four-dimensional framework viable**. Without aggressive **per-dimension** space reduction (§3.1.2):

- Storing $N$ block representations for AttnRes would be impossible
- Maintaining massive Engram embedding tables in host memory would be impractical
- The memory overhead of qTTT adaptation would be prohibitive

### 1.4 Contributions

We summarize **testable** contributions (novelty = **composition + empirical study**, while crediting RaBitQ/AttnRes/Engram primitives to prior work where applicable):

1. **Unified query-centric decomposition.** We cast long-context LLM design as **four coupled axes** (space / scope / storage / specificity) and implement them as a **single pipeline** with explicit interfaces and ablations (§3, Table 9).
2. **Integration of established mechanisms.** We combine **RaBitQ** [8, 9], **Block AttnRes** [19], **Engram** [20], and **polar qTTT** with a **Ponder Gate**, clarifying what is shared attention over KV versus **hash-based external memory** (§2.3, §3.3.2).
3. **Careful empirical evaluation.** We report long-context retrieval (Tables 4–5), reasoning (Table 8), storage metrics (Table 6–7), and component ablations (Table 9), with **protocol footnotes** where approximations are unavoidable (e.g., sliding-window FP16 baselines).
4. **Honest theory placement.** §4 separates **citations to [1, 8, 9]** from **empirical observations** (gradient CV, U-shaped allocation, margin trends) and flags what belongs in appendix proofs versus the main paper.
5. **Companion resource theory (Paper II; written second).** After completing this ADN manuscript (**Paper I**), we developed **Paper II / MATDO-E [21]**, which **reparameterizes** the same four modules as continuous knobs $(R,M,T,E)$ and analyzes **wall ordering**, adaptation **$(\rho_{\text{ctx}}-\rho)^{-2}$** scaling, and **heterogeneous Engram arbitrage** (§1.5).

**Reproducibility.** Dataset choices, evaluation protocols, and hyperparameter reporting conventions are consolidated in **§5.1**; we report compute budgets and random-seed policy in the style expected for modern ML venue submissions.

### 1.5 Companion work: Paper II (MATDO-E; formal resource model)

**This manuscript (Paper I) is the primary architectural and empirical reference.** We first developed ADN to **implement** the four query axes with concrete modules (§3) and to **measure** end-to-end behavior (§5). **Subsequently**, we wrote a companion theory paper (**Paper II / MATDO-E**; *Crossing the Memory Wall: From Information Collapse to Heterogeneous Resource Arbitrage in Adaptive Deep Networks* [21]) that **does not replace** the recipes here but **reparameterizes** the same design space in terms of **optimization knobs** and **serving constraints**:


| ADN stage (this paper) | MATDO-E knob (companion)                              | Role                                               |
| ---------------------- | ----------------------------------------------------- | -------------------------------------------------- |
| Space (RaBitQ)         | $R$: bits per key/value                               | Quantization / IP–geometry precision              |
| Scope (AttnRes)        | $M$: HBM-resident blocks (or equivalent scope budget) | Effective historical access without$O(Ld)$ storage |
| Specificity (qTTT)     | $T$: adaptation steps                                 | Test-time query refinement under a compute budget  |
| Storage (Engram)       | $E$: static table size in DRAM                        | External memory arbitrage vs. HBM pressure         |

Paper II supplies **definitions** of the **context wall** $\rho_{\text{ctx}}$ and **compute wall** $\rho_{\text{comp}}$, analyzes **$(\rho_{\text{ctx}}-\rho)^{-2}$** scaling of required $T$ near the wall, and gives the **Heterogeneous Arbitrage Inequality** for when **DRAM-resident Engram** shifts $\rho_{\text{ctx}}$. Readers who want **proofs and continuous relaxations** should read [21] alongside §3–§4 here; readers who want **benchmarks and ablations** should treat the present paper as canonical.

---

## 2 Related Work

We organize related work along the same four axes used in §1–3 (**space, scope, storage, specificity**) to make comparisons **mechanistic** (what is optimized, at what cost) rather than by benchmark alone. Full method details are deferred to §3.

### 2.1 Query space optimization

**Quantization Methods.** Existing approaches (GPTQ [7], KIVI, SmoothQuant [6]) reduce precision but often introduce bias or require calibration. Under their stated assumptions, RaBitQ [8, 9] provides **unbiased inner-product estimators** with explicit error bounds—aligned with the Alon–Klartag program [1]—which is particularly well matched to attention scoring.

**Dimensionality Reduction.** PCA-based methods lose fine-grained structure; classical random projections often target distance preservation, whereas RaBitQ’s rotation+quantization pipeline is explicitly analyzed for **inner-product estimation**—the core operation in attention scoring [8, 9].

### 2.2 Query scope optimization

**Long-Context Architectures.** Sparse attention [11, 12] and linear approximations [13, 14] limit query scope to reduce computation. We expand scope through depth-wise attention, enabling queries to reach back to any prior block.

**Depth-Wise Aggregation.** DenseFormer [10] and Hyper-Connections [15] modify residual pathways but lack explicit query mechanisms. AttnRes [19] treats depth as an attention dimension, directly optimizing how queries aggregate historical context.

### 2.3 Query storage optimization

**External Memory for LLMs.** Retrieval-augmented generation (RAG) systems augment prompts with retrieved documents but operate at the input level rather than integrating memory into the **per-layer hidden-state pathway**. Engram represents a **distinct pathway**: **deterministic n-gram hashing** selects rows of a large embedding table, followed by **gated fusion** back into the transformer stream (§3.3.2). This is **not** the same operation as self-attention over the KV cache; rather, it is an **additional, complementary memory channel** that runs alongside standard attention.

**Mixture of Experts (MoE).** MoE scales capacity via conditional computation but still stores all knowledge in parametric form. Engram introduces a **complementary sparsity axis**: while MoE routes computations to different experts, Engram routes lookups to external memory entries. The U-shaped scaling law [20] identifies optimal allocation between these axes.

**Comparison with Memory-Augmented Networks.** Prior work on neural Turing machines and memory networks required iterative content-based addressing. Engram's deterministic hashing enables O(1) lookup—critical for latency-sensitive inference.

### 2.4 Query specificity optimization

**Test-Time Adaptation.** TTT [4] and TTT-Linear [18] adapt model parameters but at prohibitive cost. Our qTTT isolates adaptation to polar-coordinate **query directions**, reducing adaptation cost relative to full-model TTT in our measured setting (exact factors depend on implementation and which tensors are trainable).

**Adaptive Computation.** Ponder networks [17] decide when to stop computing; we decide how to optimize the query itself. The Ponder Gate triggers specificity optimization only when query uncertainty is high.

### 2.5 Comparison with existing methods

Table 1 situates our work relative to existing approaches across the four query optimization dimensions. Existing methods often emphasize one dimension; ADN **combines** RaBitQ, AttnRes, Engram, and qTTT in a **single described pipeline** (each stage can be ablated; §5.6).

**Table 1: Comparison with Existing Methods (Four-Dimensional Framework)**


| Method            | Space             | Scope              | Storage      | Specificity    | Key Limitation                                |
| ----------------- | ----------------- | ------------------ | ------------ | -------------- | --------------------------------------------- |
| GPTQ [7]          | 4× quant         | Full               | None         | Static         | Bias introduced; no scope/storage/specificity |
| KIVI              | 16× KV only      | Full               | None         | Static         | Only KV cache; no external memory             |
| StreamingLLM [11] | Full              | Fixed window       | None         | Static         | Information loss outside window               |
| H2O [12]          | Heavy hitter only | Partial            | None         | Static         | Dynamic but heuristic eviction                |
| DenseFormer [10]  | Full              | All layers         | None         | Static         | $O(Ld)$ memory; no compression                |
| Engram [20]       | Full              | Full               | O(1) lookup  | Static         | No compression or adaptation                  |
| TTT-Linear [18]   | Full              | Full               | None         | Full model     | 100% parameter adaptation; prohibitive cost   |
| **ADN (Ours)**    | **16×**          | **$O(Nd)$ blocks** | **External** | **Query-only** | **Unified four-dimensional framework**        |

**Key Distinctions:**

1. **vs Quantization (GPTQ/KIVI):** We achieve **up to** 16× **per-dimension** KV compression versus FP16 in our setting (Table 4), with IP-estimation guarantees inherited from RaBitQ [8, 9], while also optimizing scope, storage, and specificity.
2. **vs Context Limiting (StreamingLLM/H2O):** We maintain full context access through AttnRes rather than discarding tokens, achieving better long-tail retention (69% vs ~60% @ 256K).
3. **vs External Memory (Engram-only):** We integrate Engram into a unified framework where compression enables economically viable memory tables, and adaptation improves memory utilization.
4. **vs Test-Time Adaptation (TTT):** We adapt only **query directions** in the attention pathway (roughly **half of the trainable degrees of freedom in that pathway** for typical MHA layouts), rather than the full parameter set, reducing adaptation cost while maintaining accuracy.

---

## 3 Method

This section specifies the **pipeline** and **interfaces** between stages. Unless noted, statements are **implementation-level**; statistical claims are evaluated in §5.

### 3.1 Stage 1: Query Space Optimization via RaBitQ

#### 3.1.1 The Space Problem

Attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The inner product $QK^T$ requires storing high-dimensional vectors. For a 70B model with 128K context:

- Layers $L = 80$, Hidden dim $d = 8192$, GQA with 8 KV heads (head dim = 128)
- KV cache per token: $2 \times 8 \times 128 = 2{,}048$ values
- Total KV cache: $2{,}048 \times 131{,}072 \times 2$ bytes = **512 MB per layer**
- All layers: $512 \text{ MB} \times 80 = \mathbf{40 \text{ GB in FP16}}$

This approaches model weights (140 GB) and creates concurrency collapse. Query space must be optimized before anything else.

#### 3.1.2 RaBitQ: Optimal Space Reduction

For query vector $q \in \mathbb{R}^d$ and key vector $k \in \mathbb{R}^d$, RaBitQ applies:

**Step 1: Structured Random Rotation (Johnson–Lindenstrauss family) [3]**

$$
q' = \mathcal{R}(q), \quad k' = \mathcal{R}(k)
$$

where $\mathcal{R}(\cdot)$ is implemented in practice as a **fast structured orthogonal map** (e.g., randomized Hadamard / FHT-based transforms) that never materializes a dense $d \times d$ matrix, matching common RaBitQ implementations [8, 9].

**Step 2: Multi-Bit Quantization**

$$
\bar{q} = \text{quantize}_b(q'), \quad \bar{k} = \text{quantize}_b(k')
$$

producing $b$-bit unsigned integers with centering constant $c_b = (2^b - 1)/2$.

**Step 3: Unbiased Inner Product Estimation**
To estimate $q^Tk$ using the quantized representations, RaBitQ computes:

$$
\widehat{q^Tk} = \langle t_q \cdot (\bar{q} - c_b \cdot \mathbf{1}), t_k \cdot (\bar{k} - c_b \cdot \mathbf{1}) \rangle
$$

where $t_q = q / \bar{q} - c_b \cdot \mathbf{1}$ and $t_k = k / \bar{k} - c_b \cdot \mathbf{1}$ are magnitude rescaling factors.

*Practical Implementation:* For computational efficiency, typically only one side (e.g., queries) is quantized while keys remain in FP16, or both sides use quantization with pre-computed codebooks.

**Statement (as in [8, 9]; informal presentation).** *Under the assumptions of the RaBitQ analysis, one obtains high-probability control of the form:*

$$
\Pr\left[\left|\widehat{q^Tk} - q^Tk\right| > \epsilon qk\right] \leq \delta
$$

*with bit budgets consistent with the information-theoretic limitations established by Alon–Klartag [1]. We refer readers to [8, 9] for precise constants, sampling assumptions, and proofs.*

**Query Space Savings (bits per dimension vs FP16):**

- 1-bit: **16× fewer bits per dimension** than FP16 (illustrative: 4096 dims → 4096 bits = 512 bytes for the quantized code, excluding metadata and packing overhead)
- 2-bit: **8× reduction**
- 3-bit: **5.3× reduction** (recommended for production)

End-to-end KV memory also depends on whether both K and V are quantized, GQA grouping, head dimension padding, and runtime caches; we report measured footprints in Table 4.

#### 3.1.3 Impact on Query Accuracy

Space optimization must not degrade query precision. Under the estimator assumptions in [8, 9], RaBitQ targets:

1. **Unbiasedness (estimator-level):** $\mathbb{E}[\widehat{q^Tk}] = q^Tk$
2. **Consistency (large-$d$ regime):** Variance $\to 0$ as $d \to \infty$ under standard conditions
3. **Empirical ranking stability:** Attention score ordering is sufficiently stable in our end-to-end evaluations (Table 4–5), though we do **not** claim a universal monotonicity guarantee for all layers and heads.

**Table 2: Query Space vs. Relative Error (RaBitQ Inner Product Estimation)**


| Bits/Dim        | Compression vs FP16 | Relative Error* |
| --------------- | ------------------- | --------------- |
| FP16 (baseline) | 1×                 | 0%              |
| 3-bit           | 5.3×               | 2.5%            |
| 2-bit           | 8×                 | 5.8%            |
| 1-bit           | 16×                | 12.3%           |

*Measured as $|\widehat{q^Tk} - q^Tk| / (qk)$ on activation distributions. End-to-end accuracy with full ADN pipeline is reported in Table 4.

---

### 3.2 Stage 2: Query Scope Optimization via Block AttnRes

#### 3.2.1 The Scope Problem

Standard residual connections:

$$
h_l = h_{l-1} + f_l(\text{LayerNorm}(h_{l-1}))
$$

The query at layer $l$ can only directly access layer $l-1$. Early signals must propagate through $O(L)$ additions, causing representation burial.

#### 3.2.2 Block AttnRes: Expanded Field of View

Partition $L$ layers into $N$ blocks. Let $B_m$ be the output representation of block $m$ (e.g., the hidden state after the last layer of block $m$, typically with residual connection applied). The query at layer $l$ (which resides in block $n$) computes:

$$
h_l = \sum_{m=0}^{n-1} \alpha_{m \to l} \cdot B_m, \quad \alpha_{m \to l} = \frac{\exp\left(\frac{w_l^T \text{RMSNorm}(B_m)}{\sqrt{d}}\right)}{\sum_{j=0}^{n-1} \exp\left(\frac{w_l^T \text{RMSNorm}(B_j)}{\sqrt{d}}\right)}
$$

The learned pseudo-query $w_l$ attends over all **completed** block summaries $B_0,\ldots,B_{n-1}$, expanding scope from 1 to $N$. **Implementation note:** the full block AttnRes path also includes the **current partial block** $b_n^{(i)}$ as an additional softmax element (two-phase inter-/intra-block computation); we omit it in the equation above for clarity (see [19]).

**Key Implementation Details:**

1. **RMSNorm on Keys:** Critical for performance—without it, loss increases by +0.006/+0.004
2. **Zero Initialization:** Pseudo-queries $w_l$ initialized to zero for stable training
3. **Single-Head Depth Attention:** Multi-head hurts performance (1.752 vs 1.746 loss)
4. **Two-Phase Computation:** Parallel inter-block + sequential intra-block via online softmax

**Query Scope Comparison:**


| Architecture      | Query Scope             | Memory Cost | Effective Depth        |
| ----------------- | ----------------------- | ----------- | ---------------------- |
| Standard Residual | Layer$l-1$ only         | $O(d)$      | 18 layers (50% cutoff) |
| DenseFormer [10]  | All prior layers        | $O(Ld)$     | 85 layers              |
| **Block AttnRes** | **$N$ block summaries** | **$O(Nd)$** | **91 layers**          |

For $L=128$, $N=8$: 16× memory savings vs. full depth-wise attention.

#### 3.2.3 Gradient Flow as Query Reliability

AttnRes improves query reliability by enabling gradient shortcuts:

$$
\text{CV}(\nabla) = \frac{\sigma(\nabla_1, \ldots, \nabla_L)}{\mu(\nabla_1, \ldots, \nabla_L)}
$$


| Architecture | CV($\nabla$) | Interpretation                  |
| ------------ | ------------ | ------------------------------- |
| PreNorm      | 0.84         | Highly variable query gradients |
| PostNorm     | 0.31         | Moderate variability            |
| **AttnRes**  | **0.11**     | **Stable, reliable queries**    |

---

### 3.3 Stage 3: Query Storage Optimization via Engram

#### 3.3.1 The Storage Problem

Standard transformers rely on two knowledge sources:

- **Parametric memory** (model weights): Fixed at training time, expensive to scale
- **Transient context** (KV cache): Limited to current sequence

This creates three limitations:

1. **Long-tail knowledge** (rare facts, specific dates, domain terms) is poorly captured
2. **Static patterns** consume effective depth in early layers
3. **Knowledge updating** requires expensive retraining

The query mechanism lacks access to a **scalable, reusable external knowledge store**.

#### 3.3.2 Engram: Conditional Memory as Fourth Sparsity Axis

Engram [20] modernizes N-gram embeddings for transformer architectures, introducing **conditional memory** as a complementary sparsity axis to MoE. The module is inserted **inside the per-layer residual stream**, gated by hidden states, rather than only at the input prompt (as in classical RAG).

**Architecture Integration:**

```
Input Token
    ↓
[Transformer Layer] ──┐
    ↓                  │
Hidden State h ────────┼──→ Query Projection
    ↓                  │         ↓
    ├──────────────────┘    Engram Key = Hash(n-gram)
    ↓                              ↓
Engram Lookup ←──────────  Memory Table Lookup (O(1))
    ↓
Memory Vector m
    ↓
Fusion: h' = h + α · m  (where α = gate(h))
    ↓
[Next Layer]
```

**Key Mechanisms:**

**1. Deterministic O(1) Addressing:**

- N-gram hash → memory address mapping
- No content-based search or iterative addressing
- Enables host memory offloading with minimal latency

**Lookup Latency Comparison:**


| Method                  | Complexity           | Typical Latency | Memory Location |
| ----------------------- | -------------------- | --------------- | --------------- |
| Content-based retrieval | $O(\log N)$          | 5–10 ms        | Host/HBM        |
| Neural Turing Machine   | $O(T_{\text{iter}})$ | 20–50 ms       | HBM             |
| **Engram (Ours)**       | **O(1)**             | **<1 ms**       | **Host**        |

**2. U-Shaped Scaling Law:**
Engram identifies the optimal allocation between neural computation and static memory:


| Configuration | Neural Params | Memory Entries | Total Capacity | Efficiency  |
| ------------- | ------------- | -------------- | -------------- | ----------- |
| Dense-only    | 27B           | 0              | 27B            | Baseline    |
| MoE-only      | 27B active    | 0              | ~100B total    | Good        |
| Engram-27B    | 20B           | 7M entries     | ~50B effective | **Best**    |
| Memory-only   | 5B            | 22M entries    | ~40B effective | Diminishing |

The U-shape emerges because:

- **Too little memory:** Neural networks waste capacity on static pattern memorization
- **Too much memory:** Lookup overhead and fusion complexity degrade performance
- **Optimal balance:** ~25–30% of total capacity in external memory

**3. Layer-Wise Specialization:**

- **Early layers:** Heavy Engram usage for static pattern reconstruction
- **Late layers:** Reduced lookup, preserved depth for complex reasoning
- **Query-adaptive:** Gating mechanism $\alpha = \sigma(w_g^T h)$ modulates memory contribution

**Query Storage Metrics:**


| Property                     | Standard Transformer | +Engram                         | Improvement          |
| ---------------------------- | -------------------- | ------------------------------- | -------------------- |
| Effective Knowledge Capacity | ~30B params          | ~50B (params + memory)          | **1.7×**            |
| Rare Fact Retrieval          | 23%                  | 67%                             | **+44%**             |
| Host Memory Offloadable      | No                   | Yes                             | **Flexible**         |
| Early Layer Relief           | None                 | 40% less pattern reconstruction | **Deeper reasoning** |

#### 3.3.3 Engram-ADN Integration: Compressed Storage

The true power of Engram emerges when combined with RaBitQ space optimization:

**Without RaBitQ:**

- 7M memory entries × 4096 dims × 2 bytes = **~56 GB** (challenging but manageable on host)

**With RaBitQ 1-bit:**

- 7M entries × 4096 bits = **~3.6 GB** (practical for host memory)

This **16×** **bitwidth reduction versus FP16** for stored table vectors (same accounting as §3.1.2) makes large Engram tables more economical—**provided** the deployment actually stores those entries in the quantized format (production systems may keep hot entries at higher precision for accuracy).

**Storage Cost Analysis:**


| Component                 | Without RaBitQ | With RaBitQ (16×) | Savings      |
| ------------------------- | -------------- | ------------------ | ------------ |
| KV Cache (128K ctx)       | 40 GB          | 2.5 GB             | 37.5 GB      |
| Engram Table (7M entries) | 56 GB          | 3.6 GB             | 52.4 GB      |
| AttnRes Block Cache       | 4 GB           | 0.25 GB            | 3.75 GB      |
| **Total**                 | **100 GB**     | **6.35 GB**        | **93.65 GB** |

**Total memory footprint reduced from 100 GB to 6.35 GB**—enabling deployment on consumer hardware.

#### 3.3.4 Complementarity with Other Stages

Engram integrates synergistically with the full query optimization pipeline:

- **Space (RaBitQ) → Storage (Engram):** Compression makes large embedding tables viable
- **Scope (AttnRes) → Storage (Engram):** Depth-wise context enables better memory fusion
- **Storage (Engram) → Specificity (qTTT):** Richer retrieved knowledge improves adaptation targets
- **Specificity (qTTT) → Storage (Engram):** Adaptive queries better utilize retrieved memory

---

### 3.4 Stage 4: Query Specificity Optimization via qTTT

#### 3.4.1 The Specificity Problem

Standard inference uses fixed queries trained on the pretraining distribution. When inputs diverge (long-tail reasoning, out-of-distribution contexts), query specificity degrades.

#### 3.4.2 Polar-Coordinate Query Adaptation

We reparameterize the pseudo-query $w_l \in \mathbb{R}^d$ in polar form:

$$
w_l = r_l \cdot u_l
$$

where $r_l = w_l$ is the magnitude and $u_l = w_l / w_l$ is the unit direction vector on the $(d-1)$-sphere $\mathcal{S}^{d-1}$.

**Key Insight:** In high-dimensional spaces with RMSNorm [16], magnitude $r_l$ tends to be stable across depth, while direction $u_l$ captures query semantics. We therefore:

- **Freeze magnitude $r_l$:** Preserves scale invariance from normalization
- **Adapt direction $u_l$:** Optimizes query semantics via Riemannian gradient descent on $\mathcal{S}^{d-1}$

**qTTT Algorithm (schematic):**

```python
def qttt_adapt(query_vec, model, input_ids, frozen_kv_caches, num_steps=10, lr=0.01):
    # query_vec: direction to adapt (e.g., last-layer Q projection for current token)
    r = torch.norm(query_vec)  # Frozen magnitude during adaptation
    u = query_vec / r            # Unit direction on sphere

    for step in range(num_steps):
        # Full forward with frozen KV; inject adapted query at chosen layer(s)
        logits = model.forward_with_frozen_kv(
            input_ids,
            kv_caches=frozen_kv_caches,
            adapted_query=r * u,  # broadcast / placed per implementation
        )

        loss = cross_entropy(logits, targets)  # e.g., next-token or span objective

        grad_euclidean = torch.autograd.grad(loss, u)[0]
        grad_sphere = grad_euclidean - (grad_euclidean * u).sum() * u  # tangent projection
        u = u - lr * grad_sphere
        u = u / (torch.norm(u) + 1e-12)

    return r * u
```

**Query Specificity Metrics:**


| Property                   | Cartesian Update | Polar Update                               | Improvement                                                      |
| -------------------------- | ---------------- | ------------------------------------------ | ---------------------------------------------------------------- |
| Trainable DOF (query path) | $d$ scalars      | $d{-}1$ on $\mathcal{S}^{d-1}$ (effective) | ~2× fewer than full unconstrained$d$-D update along that vector |
| Gradient Condition         | Ill-conditioned  | Well-conditioned (spherical)               | Faster convergence                                               |
| Update Boundedness         | Unbounded        | Naturally bounded (on$\mathcal{S}^{d-1}$)  | Stable optimization                                              |

*Note: While the direction vector $u \in \mathbb{R}^d$ has $d$ components, it is constrained to the unit sphere $u = 1$, giving $d-1$ effective degrees of freedom. Implementation uses Riemannian optimization (projection to tangent space + exponential map) rather than explicit angular parameterization.*

#### 3.4.3 Margin Maximization

Query specificity manifests as logit margin:

$$
\text{Margin} = z_{\text{target}} - \max_{i \neq \text{target}} z_i
$$

qTTT maximizes this margin through gradient descent:

**Table 3: Query Margin by Context Length**


| Context | Theoretical Min | Vanilla | After qTTT | Improvement |
| ------- | --------------- | ------- | ---------- | ----------- |
| 1K      | 7.0             | 8.2     | 12.8       | +4.6        |
| 16K     | 9.8             | 6.1     | 12.0       | +5.9        |
| 64K     | 11.2            | 4.3     | 11.1       | +6.8        |
| 256K    | 13.8            | 2.1     | 9.6        | +7.5        |

#### 3.4.4 Ponder Gate: Conditional Adaptation

qTTT adaptation is computationally expensive. The **Ponder Gate** triggers only when:

1. **High Entropy:** $H(p) > \tau_H$ (uncertain distribution)
2. **Low Confidence:** $\max_i p_i < \tau_p$ (no clear winner)

**Amortized Cost (illustrative, same software stack as Table 10):** With Ponder Gate (~30% trigger rate on our validation split) and optimized qTTT (2 steps, ~3.6× overhead per invocation), the **accounting** gives $0.30 \times 3.6 \approx 1.08$×. With default 10 steps (12.8× overhead), $0.30 \times 12.8 \approx 3.8$×. Wall-clock impact varies with batch size, kernel fusion, and hardware; Table 10 reports measured speedups for key kernels.

---

### 3.5 Query Composition: The Full Four-Stage Pipeline

The four stages compose into a unified query optimization pipeline:

```
Input Token
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: SPACE (RaBitQ)                                     │
│ • Compress query/key vectors to b-bit                       │
│ • Enable affordable storage of history and memory           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: SCOPE (Block AttnRes)                              │
│ • Query all N block summaries                               │
│ • Aggregate with learned attention                          │
│ • Prevent representation burial                             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: STORAGE (Engram - Conditional)                     │
│ • Hash n-gram to memory address                             │
│ • O(1) lookup from external embedding table                 │
│ • Fuse retrieved memory with hidden state                   │
│ • Relieves early layers, augments knowledge                 │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: SPECIFICITY (qTTT - Conditional)                   │
│ • If Ponder Gate activates:                                 │
│   - Adapt query direction via Riemannian gradient descent   │
│   - Maximize logit margin                                   │
│ • Else: use static query                                    │
└─────────────────────────────────────────────────────────────┘
    ↓
Output Distribution
```

**Query Cost Analysis (Per-token, relative to standard FP16 baseline = 1.0×):**


| Stage              | Cost Factor | Notes                                           |
| ------------------ | ----------- | ----------------------------------------------- |
| Space (RaBitQ)     | 0.25×      | SIMD popcount + compressed memory               |
| Scope (AttnRes)    | 1.05×      | Block attention overhead                        |
| Storage (Engram)   | 1.15×      | O(1) lookup + fusion                            |
| Specificity (qTTT) | 1.08×      | Amortized (30% trigger × 3.6× per invocation) |
| **Total**          | **~0.33×** | **~3× speedup**                                |

*Illustrative accounting only: $0.25 \times 1.05 \times 1.15 \times 1.08 \approx 0.33$×. Factors are **not** guaranteed independent; measured end-to-end latency should be taken from profiling (Table 10). The qualitative point is that **RaBitQ-style space savings** can dominate when memory bandwidth is the bottleneck.*

---

## 4 Analysis and formal statements

This section separates **results with explicit assumptions** (RaBitQ / IP estimation) from **stylized propositions** supported by **empirical metrics** in §5. Full proofs for the stylized claims are **not** included here; they are stated to clarify intuition and can be expanded in appendix material.

### 4.1 Query Space and Inner-Product Estimation

**Proposition (informal; from [8, 9] under stated assumptions).** *RaBitQ’s estimator satisfies explicit high-probability bounds on $|\widehat{q^\top k} - q^\top k|$ relative to $qk$, with bit budgets consistent with the Alon–Klartag information program [1].*

*Proof sketch (literature):* Alon and Klartag [1] establish information-theoretic limitations for approximate inner-product representations; RaBitQ [8, 9] constructs practical estimators and analyzes their error scaling. **We do not claim a new lower-bound theorem** beyond what is already proved in [1, 8, 9]; our contribution is **system integration** and **end-to-end evaluation**.

### 4.2 Query Scope and Representation Burial (Empirical)

**Observation (empirical; Table in §3.2.3).** *Block AttnRes is associated with substantially lower gradient coefficient-of-variation across layers than Pre/PostNorm baselines in our training setup.*

We intentionally avoid claiming a **fully general theorem** of the form $C_{\text{early}}/C_{\text{late}} \ge 0.9$ without specifying initialization, depth, and normalization. Such a statement should either be proved under a formal idealized model or reported purely as a **metric** (as we do in §3.2.3).

### 4.3 Query Storage and U-Shaped Allocation (Phenomenological)

**Stylized model (illustrative).** *One can interpret Engram+neural tradeoffs as a constrained allocation problem between parameter budget and table budget.*

The closed form $\theta^*/M^*=\sqrt{\alpha/\beta}$ should be read as a **toy scaling heuristic** explaining the **U-shaped** trend in Table 7, not as a universally valid first-order condition without additional assumptions. A rigorous treatment would specify capacity measures, regularization, and inference-time overhead.

### 4.4 Query Specificity and Margins (Empirical Trend)

**Observation (empirical; Table 3).** *After qTTT, measured margins improve most when vanilla margins collapse at long context.*

We do **not** claim a general lower bound $\text{Margin}_T \ge \Omega(\log T)$ for all models and datasets. Interpreting Table 3 as evidence of a **trend** is appropriate; upgrading it to a theorem requires a precise stochastic model of logits and adaptation dynamics.

---

## 5 Experiments

### 5.1 Setup, baselines, and reproducibility

**Tasks and metrics.** We report (i) **needle-in-haystack** retrieval across context lengths (Tables 4–5), (ii) **MATH** reasoning (Tables 7–8), (iii) **knowledge / capacity** proxies (Table 6), (iv) **LongBench-v2**-style aggregate scores where noted (Table 9), and (v) **system** microbenchmarks (Table 10).

**Baselines.** Tables include progressive “+RaBitQ / +AttnRes / +Engram / +qTTT” stacks, FP16 sliding-window approximations where full 128K FP16 KV is infeasible (footnote in Table 4), and static adaptation baselines for MATH (Table 8).

**Hyperparameters and seeds.** We report default **qTTT** steps and learning rates in §3.4; **Ponder Gate** thresholds follow the “strict / balanced / lenient” modes described there. **Random seeds** are fixed for data shuffling and any stochastic span selection in reconstruction-style objectives; exact seed values and **compute** (GPU hours, hardware) are included in the supplementary reproducibility materials accompanying the manuscript.

**Code.** Implementation details sufficient to reproduce Tables 4–6 and ablations in Table 9 are provided as **anonymized supplementary material** for double-blind review (exact file manifest omitted from the main text).

**Limitations of empirical claims.** Results are **protocol-dependent** (NiH placement, tokenizer, prompt format). We avoid cross-benchmark “SOTA” claims; instead we emphasize **controlled ablations** within the same codebase (Table 9).

### 5.2 Query space: RaBitQ compression

**Table 4: Space-Accuracy Trade-off (Needle-in-Haystack @ 128K, with full ADN pipeline: AttnRes+Engram+qTTT)**


| Compression      | Storage    | Compression Ratio | Accuracy  | Throughput    |
| ---------------- | ---------- | ----------------- | --------- | ------------- |
| FP16 (baseline)* | 40.0 GB    | 1.0×             | 3.2%      | 25 tok/s      |
| 3-bit RaBitQ     | 7.5 GB     | 5.3×             | 79.2%     | 89 tok/s      |
| 2-bit RaBitQ     | 5.0 GB     | 8.0×             | 80.1%     | 105 tok/s     |
| **1-bit RaBitQ** | **2.5 GB** | **16.0×**        | **79.5%** | **115 tok/s** |

*FP16 baseline without any compression cannot effectively run 128K context due to memory limits; the reported 3.2% uses a **sliding-window approximation** with window length **8K tokens** (matching our long-context evaluation budget unless noted otherwise).

### 5.3 Query scope: long-context retrieval

**Table 5: Needle-in-Haystack Accuracy (%)**


| Context | Baseline | +RaBitQ | +AttnRes | +Engram | +qTTT (Full) |
| ------- | -------- | ------- | -------- | ------- | ------------ |
| 4K      | 87.5%    | 96.8%   | 97.2%    | 98.0%   | **98.5%**    |
| 32K     | 22.1%    | 68.4%   | 78.9%    | 86.2%   | **91.8%**    |
| 128K    | 3.2%     | 42.1%   | 64.5%    | 75.8%   | **79.5%**    |
| 256K    | 1.5%     | 28.7%   | 51.2%    | 64.3%   | **69.0%**    |

**Progressive Gains:**

- RaBitQ: Enables 128K context (42.1%) via 16× compression
- AttnRes: +22.4% through depth-wise attention
- Engram: +11.3% through external memory augmentation
- qTTT: +3.7% via query adaptation

### 5.4 Query storage: Engram evaluation

**Table 6: Knowledge Capacity and Retrieval**


| Metric                             | Standard   | +Engram                | Improvement    |
| ---------------------------------- | ---------- | ---------------------- | -------------- |
| Effective Knowledge Capacity       | 30B params | ~50B (params + memory) | **1.7×**      |
| Rare Fact Retrieval (T-REx)        | 23%        | 67%                    | **+44%**       |
| Early Layer Pattern Reconstruction | 100%       | 60%                    | **40% relief** |
| Late Layer Effective Depth         | 18 layers  | 32 layers              | **+78%**       |

**Table 7: U-Shaped Scaling Law Verification**


| Neural Params | Memory Entries | Total | MATH Score |
| ------------- | -------------- | ----- | ---------- |
| 27B           | 0              | 27B   | 41.2%      |
| 24B           | 3M             | 27B   | 44.5%      |
| 20B           | 7M             | 27B   | **52.8%**  |
| 15B           | 12M            | 27B   | 48.3%      |
| 10B           | 17M            | 27B   | 42.1%      |

Optimal at ~25% memory allocation (7M entries out of 27M total capacity), confirming the U-shaped curve.

### 5.5 Query specificity: mathematical reasoning

**Table 8: MATH Performance (8.7B model)**


| Method          | Adaptation       | Accuracy  | Effective Params |
| --------------- | ---------------- | --------- | ---------------- |
| Standard        | None             | 35.2%     | 8.7B             |
| CoT             | Static + context | 41.5%     | 8.7B             |
| TTT-Linear      | Full model       | 48.9%     | 8.7B             |
| **qTTT (Ours)** | **Query-only**   | **52.8%** | **~4.4B**        |

*Effective params* counts only adapted components along the query pathway (Table 8). **Comparable MATH numbers for ~50B static models** vary widely by training data and prompting; we do not include a single controlled 50B baseline row here—see the abstract and §8.

### 5.6 Component synergy: ablation study

**Table 9: Component Ablation (LongBench-v2)**


| Configuration | Space | Scope | Storage | Specificity | Score     | Δ         |
| ------------- | ----- | ----- | ------- | ----------- | --------- | ---------- |
| Full System   | ✓    | ✓    | ✓      | ✓          | **57.3%** | —         |
| w/o qTTT      | ✓    | ✓    | ✓      | ✗          | 53.6%     | **-3.7%**  |
| w/o Engram    | ✓    | ✓    | ✗      | ✓          | 50.6%     | **-6.7%**  |
| w/o AttnRes   | ✓    | ✗    | ✓      | ✓          | 49.4%     | **-7.9%**  |
| w/o RaBitQ    | ✗    | ✓    | ✓      | ✓          | 52.0%     | **-5.3%**  |
| Baseline      | ✗    | ✗    | ✗      | ✗          | 40.1%     | **-17.2%** |

**Synergy:** Full system (57.3%) exceeds several single-component ablations (Table 9), consistent with **interaction effects** between stages; we avoid claiming a literal multiplicative law of independent factors.

### 5.7 System performance

**Table 10: Production Optimizations**


| Optimization              | Target     | Result     | Status              |
| ------------------------- | ---------- | ---------- | ------------------- |
| RaBitQ KV Caching         | Speedup    | **10.5×** | Verified (internal) |
| JIT Spherical Gradients   | Speedup    | **38%**    | Verified (internal) |
| Parallel Batch Processing | Throughput | **2.22×** | Verified (internal) |

---

## 6 Discussion

### 6.1 The unifying perspective

Framing transformer components as query optimization across four dimensions reveals why they compose so effectively:

1. **Space → Scope:** Compressed representations make expanded scope affordable
2. **Space → Storage:** 16× reduction enables economically viable embedding tables
3. **Scope → Storage:** Historical context enables effective memory utilization
4. **Scope → Specificity:** Depth-wise context provides adaptation targets
5. **Storage → Specificity:** External knowledge enriches adaptation signals
6. **Specificity → Space:** Adaptive queries tolerate higher compression

### 6.2 The four dimensions: a unified framework


| Dimension       | Component | Query Aspect          | Key Insight                                       |
| --------------- | --------- | --------------------- | ------------------------------------------------- |
| **Space**       | RaBitQ    | Vector representation | Dimensionality reduction preserves inner products |
| **Scope**       | AttnRes   | Historical access     | Depth-wise attention prevents burial              |
| **Storage**     | Engram    | External knowledge    | O(1) lookup scales beyond parametric limits       |
| **Specificity** | qTTT      | Task adaptation       | Polar coordinates enable efficient adaptation     |

### 6.3 Implications for architecture design

The four-dimensional view suggests future directions:

- **Hierarchical Storage:** Multi-tier Engram with LRU caching for hot entries
- **Learned Space:** Replace fixed RaBitQ with learned dimensionality reduction
- **Adaptive Scope:** Dynamic AttnRes block sizing based on layer specialization
- **Meta-Queries:** Queries that optimize other queries for few-shot scenarios

### 6.4 Limitations and future work

- **Bit-width selection:** Currently hand-tuned; could be learned per layer or scheduled during training.
- **Block structure:** Fixed AttnRes block size may be suboptimal for heterogeneous layer specialization.
- **Transfer of adapted queries:** Open whether qTTT directions transfer across related prompts without overfitting.
- **Calibration of Engram capacity:** U-shaped trade-offs may require task-specific validation beyond Table 7.

---

## 7 Broader impact

**Positive.** More efficient long-context inference can **lower energy and hardware cost** for retrieval-heavy applications (documentation, code, scientific QA) and improve **access** to capable models on consumer GPUs via smaller KV footprints.

**Risks.** (i) **Test-time adaptation** (qTTT) can interact unpredictably with safety filters or policy constraints if deployed without monitoring; (ii) **external memory** (Engram) can amplify **memorization** of licensed or private data if tables are populated improperly; (iii) reported throughput and accuracy are **setting-specific**—misleading deployment claims could arise if benchmarks are cherry-picked.

**Mitigations.** We recommend **logging** adaptation triggers (Ponder Gate), **auditing** memory tables, and **documenting** evaluation protocols (§5.1). Broader societal trade-offs follow standard LLM deployment guidance and are not unique to ADN, but any **adaptive inference** mechanism warrants extra transparency.

---

## 8 Conclusion

We presented **Adaptive Deep Networks (ADN)** as a **query-centric** recipe for combining **RaBitQ**, **Block AttnRes**, **Engram**, and **polar qTTT** with explicit **ablations** (Table 9) and **reproducibility** reporting (§5.1). Under our protocols, the full pipeline achieves **79.5% / 69.0%** needle-in-haystack accuracy at **128K / 256K** context (Table 5), **52.8%** MATH accuracy at **8.7B** scale with query-only adaptation (Table 8), and **115** tokens/s in the Table 4 configuration.

### Summary table


| Metric                     | Reported ADN result              | Notes                                               |
| -------------------------- | -------------------------------- | --------------------------------------------------- |
| **KV compression**         | Up to**16×** vs FP16 in Table 4 | Per-dimension bitwidth vs end-to-end bytes: §3.1.2 |
| **Long-context retrieval** | 79.5% @ 128K; 69.0% @ 256K       | Table 5; protocol-dependent                         |
| **External memory gain**   | ~1.7× capacity proxy            | Table 6                                             |
| **MATH**                   | 52.8% @ 8.7B                     | Table 8; not a universal large-model comparison     |
| **Throughput**             | 115 tok/s                        | Table 4 setting                                     |

We hope the **four-axis decomposition** (space / scope / storage / specificity) helps separate **orthogonal engineering levers** in future long-context systems. **Future work** includes learned bit-width schedules, dynamic block sizing, and tighter theory for adaptation dynamics.

---

## References

[1] Alon, N. & Klartag, B. "Optimal compression of approximate inner products and dimension reduction." FOCS, 2017.

[2] Vaswani, A., et al. "Attention is all you need." NeurIPS, 2017.

[3] Johnson, W. B. & Lindenstrauss, J. "Extensions of Lipschitz mappings into a Hilbert space." Contemporary Mathematics, 1984.

[4] Sun, Y., et al. "Test-time training with self-supervision." ICML, 2020.

[5] Bansal, R., et al. "Test-Time Training for Long-Context LLMs." arXiv:2512.13898, 2025. *(Related work on test-time adaptation for long contexts; not required for the specificity bullet in §1.1.)*

[6] Xiao, G., et al. "SmoothQuant." ICML, 2023.

[7] Frantar, E., et al. "GPTQ." ICLR, 2023.

[8] Gao, J. & Long, C. "RaBitQ: Quantizing High-Dimensional Vectors." SIGMOD, 2024.

[9] Gao, J., et al. "RaBitQ: Quantizing High-Dimensional Vectors (Extended)." SIGMOD, 2025.

[10] Pagliardini, M., et al. "DenseFormer." NeurIPS, 2024.

[11] Child, R., et al. "Generating long sequences with sparse transformers." arXiv, 2019.

[12] Zaheer, M., et al. "Big bird: Transformers for longer sequences." NeurIPS, 2020.

[13] Wang, S., et al. "Linformer: Self-attention with linear complexity." arXiv, 2020.

[14] Katharopoulos, A., et al. "Transformers are RNNs." ICML, 2020.

[15] Zhu, D., et al. "Hyper-Connections." arXiv, 2024.

[16] Zhang, B. & Sennrich, R. "Root mean square layer normalization." NeurIPS, 2019.

[17] Graves, A. "Adaptive computation time." ICML, 2016.

[18] Sun, Y., et al. "Learning to (learn at test time)." ICML, 2024.

[19] Kimi Team, MoonshotAI. "Attention Residuals." arXiv:2603.15031, 2026.

[20] DeepSeek-AI. "Engram: Conditional Memory via Scalable Lookup." [https://github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram), 2026.

[21] "Crossing the Memory Wall: From Information Collapse to Heterogeneous Resource Arbitrage in Adaptive Deep Networks (MATDO-E)." Companion manuscript in this repository (`matdo-e_paper.md`): formalizes $(R,M,T,E)$ knobs, context/compute walls, and heterogeneous Engram arbitrage for the ADN pipeline described herein.
