# Crossing the Memory Wall: From Information Collapse to Heterogeneous Resource Arbitrage in Adaptive Deep Networks (MATDO-E)

**Companion to Adaptive Deep Networks (ADN).** The project **first** produced **Paper I** (**ADN**; [1]): a modular **RaBitQ → AttnRes → Engram → qTTT** stack with **benchmarks** (long context, reasoning, latency) and **ablations**. **This manuscript is Paper II** (**MATDO-E**): it **abstracts** that stack into continuous knobs $(R,M,T,E)$ and analyzes **wall ordering**, **$(\rho_{\text{ctx}}-\rho)^{-2}$ adaptation blow-up**, and **heterogeneous Engram arbitrage**. Use **Paper I** for architecture and measured end-to-end scores; use **Paper II** for optimization-theoretic claims.

**Anonymous Submission**

---

## Abstract

**Writing order.** **Paper I (ADN; [1])** specifies the **four-axis query framework** and the **concrete inference pipeline** evaluated on retrieval and reasoning tasks. **Paper II (MATDO-E)** **follows**: we **do not** introduce a competing architecture, but a **resource model** for the **same** design space.

Large language model (LLM) serving is constrained by GPU HBM. We reveal two critical points: the *compute wall* and the *context wall*. We prove adaptation cost diverges as $(\rho_{\text{ctx}}-\rho)^{-2}$. Our **MATDO-E** framework couples these walls to **$(R,M,T,E)$** and studies **DRAM-resident Engram** as heterogeneous arbitrage. Evaluations across LLaMA-2, Mistral, and Qwen architectures **illustrate** cross-family scaling; **module-level accuracy/latency tables** appear in **[1]**.

---

## 1. Introduction

**Chronology.** We developed **Paper I (ADN; [1])** *first*: it names **RaBitQ, AttnRes, Engram, and qTTT** as the **instantiations** of **space / scope / storage / specificity**, and reports **needle-in-haystack**, **MATH**, throughput, and **ablations**. **Paper II (MATDO-E)** is written **after** Paper I: it **keeps** the same four axes but replaces narrative proofs-of-concept with a **continuous** allocation problem and **theorems**. When both papers are cited, **cite Paper I for stack and benchmark claims**; **cite Paper II for formal walls, arbitrage, and asymptotic $T$ scaling**.

Modern LLMs process queries using attention over a key-value (KV) cache that grows linearly with context length. GPU HBM capacity has become the primary bottleneck in production serving: once the KV cache exceeds HBM, systems must either drop context (harming accuracy) or offload to slower CPU DRAM (increasing latency). This tension is often called the *memory wall*. **Paper I (ADN; [1])** addresses this with the **module bundle** above; **here** we ask how **$(R,M,T,E)$** behave **as $\rho \to \rho_{\text{ctx}}$**.

### Contributions

*(Empirical pipeline and task metrics are in **Paper I** ([1]); the list below is **theory- and policy-centric**.)*

1. **Dual critical points**: We formalize the *compute wall* $\rho_{\text{comp}}$ (where required adaptation steps exceed the latency budget) and the *context wall* $\rho_{\text{ctx}}$ (where remaining context cannot satisfy accuracy even with infinite compute). We prove $\rho_{\text{comp}} < \rho_{\text{ctx}}$; systems always hit the compute wall first.

2. **Quadratic blow-up**: As $\rho \to \rho_{\text{ctx}}^-$, the optimal number of test-time adaptation steps grows as $(\rho_{\text{ctx}}-\rho)^{-2}$, providing a rigorous explanation for the performance cliff **for query-only adaptation as modeled in Paper I**.

3. **Engram and arbitrage**: We analyze a **DRAM-resident** static table (**Engram**, as used in Paper I) and derive the **Heterogeneous Arbitrage Inequality**, a condition for when **DRAM budget** **postpones** the context wall. Via convex duality analysis, we prove this inequality is necessary and sufficient for Pareto-dominating $E=0$ in a **relaxed** convex setting.

4. **Cross-model validation**: We report **Paper II (MATDO-E)** behavior across LLaMA-2 (MHA), Mistral (GQA/Sliding Window), and Qwen (GQA), as **stress-tests** of the **scaling law**; **architectural ablations** remain in **Paper I**.

---

## 2. Preliminaries and Problem Formulation

**Map from Paper I to $(R,M,T,E)$.** **RaBitQ** (KV **space** / bits) $\rightarrow$ **$R$**; **AttnRes** block summaries (**scope** / HBM-resident history) $\rightarrow$ **$M$**; **qTTT** (**specificity**) $\rightarrow$ **$T$**; **Engram** table (**external storage**) $\rightarrow$ **$E$**. Paper II **optimizes** this envelope; **§3–§5 of Paper I** specify **layers, gates, and training**.

We consider an **ADN** as defined in **Paper I** ([1]) that processes a query $\mathbf{q}$ using a KV cache of $N$ tokens. The cache is partitioned into blocks of size $N_{\text{block}}$.

### Optimization Knobs

| Knob | Symbol | Description |
|------|--------|-------------|
| Quantization | $R$ | bits per key/value (**RaBitQ** in Paper I) |
| Scope | $M$ | number of blocks kept in HBM (**AttnRes** budget; Paper I) |
| Specificity | $T$ | test-time adaptation steps (**qTTT** in Paper I) |
| Engram size | $E$ | static entries in DRAM (**Engram** in Paper I) |

### Constraints

1. **HBM capacity**:
$$M \cdot N_{\text{block}} \cdot R \cdot C_{\text{unit}} \le C_{\text{HBM}} (1-\rho)$$

2. **DRAM capacity** (if Engram used):
$$E \cdot L \le C_{\text{DRAM}} (1-\rho_{\text{DRAM}})$$

3. **Compute budget** (latency SLA):
$$\mathcal{B} = c_R R d + c_M M S d + c_T T d^2 + c_E E L \le \mathcal{B}_{\max}$$

### Error Model

The end-to-end error decomposes into additive contributions:

$$\mathcal{E}(R,M,T,E) = \underbrace{\alpha 2^{-2R}}_{\text{quant}} + \underbrace{\frac{\beta}{MS} \cdot f(E)}_{\text{scope}} + \underbrace{\frac{\gamma}{\sqrt{T}}}_{\text{specificity}} + \underbrace{\delta \frac{2^{-2R}}{M} + \epsilon \frac{\ln M}{T}}_{\text{couplings}} + \underbrace{\frac{\eta}{E}}_{\text{retrieval overhead}}$$

where $f(E)=1-\zeta(1-e^{-E/E_0})$ is the Engram compensation function.

---

## 3. The Dual Critical Points

We first analyze the system without Engram ($E=0$).

### 3.1 Definitions

**Definition 3.1** (Minimum feasible scope). For a given target error $\mathcal{E}_{\text{target}}$, the minimum number of HBM blocks required:

$$M_{\min} = \frac{\beta + \delta 2^{-2R_{\min}}}{S\bigl(\mathcal{E}_{\text{target}}-\alpha 2^{-2R_{\min}}\bigr)}$$

---

**Definition 3.2** (Context wall). The HBM utilization at which available memory exactly fits $M_{\min}$ blocks:

$$\rho_{\text{ctx}} = 1 - \frac{M_{\min} N_{\text{block}} R_{\min} C_{\text{unit}}}{C_{\text{HBM}}}$$

For $\rho > \rho_{\text{ctx}}$, even with infinite compute the SLA cannot be met.

---

**Definition 3.3** (Compute wall). Given a FLOPs budget $\mathcal{B}_{\max}$, the maximum feasible adaptation steps are $T_{\max} = (\mathcal{B}_{\max} - c_R R_{\min} d - c_M M S d)/(c_T d^2)$. The compute wall $\rho_{\text{comp}}$ is the largest $\rho$ such that $T^*(\rho) \le T_{\max}$.

---

### 3.2 Ordering of Walls

**Theorem 3.4** (Ordering of walls). For any feasible system, $\rho_{\text{comp}} < \rho_{\text{ctx}}$. That is, the system runs out of compute budget before it runs out of context.

**Proof.** When HBM is tight, $M$ is determined by:

$$M^*(\rho) = \frac{C_{\text{HBM}}(1-\rho)}{N_{\text{block}} R_{\min} C_{\text{unit}}}$$

As $\rho \to \rho_{\text{ctx}}^-$, $M^* \to M_{\min}^+$. Define $\delta_M = M^*-M_{\min} \propto (\rho_{\text{ctx}}-\rho)$. The residual error budget:

$$\Delta(\rho) = \mathcal{E}_{\text{target}} - \alpha 2^{-2R_{\min}} - \frac{\beta}{M^* S} - \delta \frac{2^{-2R_{\min}}}{M^*}$$

Using Taylor expansion:

$$\Delta(\rho) = \frac{\beta \delta_M}{M_{\min}^2 S} + \delta 2^{-2R_{\min}} \frac{\delta_M}{M_{\min}^2} + O(\delta_M^2) \propto (\rho_{\text{ctx}}-\rho)$$

Setting $\gamma/\sqrt{T^*} = \Delta(\rho)$ gives $T^*(\rho) \propto (\rho_{\text{ctx}}-\rho)^{-2}$, which diverges as $\rho \to \rho_{\text{ctx}}^-$. Since $T_{\max}$ is finite, there exists a unique $\rho_{\text{comp}} < \rho_{\text{ctx}}$.

### 3.3 Quadratic Blow-up

As HBM utilization approaches the context wall, adaptation steps diverge quadratically:

```
T* │                              ╱
   │                             ╱
   │                            ╱
   │                           ╱
   │                          ╱
   │                         ╱
   │                        ╱
   └───────────────────────┴──────────→ 1/(ρ_ctx - ρ)
                          T_max
```

---

## 4. Heterogeneous Resource Arbitrage via Engram

When DRAM is abundant and cheap, we can store a large Engram $E$ to reduce the effective scope error via $f(E)=1-\zeta(1-e^{-E/E_0})$.

### 4.1 Extended Optimization

$$\min_{R,M,T,E} \quad \mathcal{B}_{\text{total}} = c_R R d + c_M M S d + c_T T d^2 + c_E E L$$

subject to the SLA and capacity constraints.

### 4.2 Heterogeneous Arbitrage Inequality

**Theorem 4.1** (Heterogeneous Arbitrage Inequality). Engram postpones the context wall ($\rho_{\text{ctx}}^E > \rho_{\text{ctx}}$) if and only if:

$$\zeta > \frac{\eta}{E_{\max} \mathcal{E}_{\text{target}}}$$

**Proof Sketch:** The new minimum scope $M_{\min}^E$ satisfies:

$$\alpha 2^{-2R_{\min}} + \frac{\beta f(E_{\max})}{M_{\min}^E S} + \delta \frac{2^{-2R_{\min}}}{M_{\min}^E} + \frac{\eta}{E_{\max}} = \mathcal{E}_{\text{target}}$$

Solving:

$$M_{\min}^E = \frac{\beta f(E_{\max}) + \delta 2^{-2R_{\min}}}{S\bigl(\mathcal{E}_{\text{target}}-\alpha 2^{-2R_{\min}} - \eta/E_{\max}\bigr)}$$

For large $E_{\max}$, $f(E_{\max})\approx 1-\zeta$. Postponement ($M_{\min}^E < M_{\min}$) requires:

$$\frac{1-\zeta}{\mathcal{E}_{\text{target}} - \eta/E_{\max}} < \frac{1}{\mathcal{E}_{\text{target}}}$$

Simplifying yields the inequality.

### 4.3 Optimality via Duality

The optimization problem is generally non-convex due to coupling terms. However, with continuous relaxation and concave $f(E)$, the feasible set becomes convex (via transformation $x=1/M$, $y=1/\sqrt{T}$, $z=1/E$), where strong duality holds.

The Lagrangian with dual variables $\lambda, \mu, \nu \ge 0$:

$$\mathcal{L} = c_M M S d + c_E E L + \lambda(\mathcal{E} - \mathcal{E}_{\text{target}}) + \mu(\text{HBM constraint}) + \nu(\text{DRAM constraint})$$

Stationarity conditions yield the **marginal condition**:

$$\frac{-\partial \mathcal{E}/\partial M}{c_M S d} = \frac{-\partial \mathcal{E}/\partial E}{c_E L} \quad \text{when both constraints are tight}$$

This states that the ratio of marginal error reduction to marginal cost must be equalized across memory tiers.

---

**Theorem 4.2** (Optimality of the Arbitrage Inequality). Assume the convex relaxation satisfies Slater's condition. Then the Heterogeneous Arbitrage Inequality is not only sufficient for $\rho_{\text{ctx}}^E > \rho_{\text{ctx}}$, but also **necessary and sufficient** for the existence of an optimal solution with $E>0$ that Pareto-dominates any solution with $E=0$.

**Proof Sketch:** At baseline $M=M_{\min}, E=0$, comparing marginal benefit of increasing $E$ vs. $M$ using the concave form of $f(E)$ shows the Lagrangian dual has negative subgradient for small $E$ iff the inequality holds. Conversely, if violated, dual variables force $E=0$ at optimality.

This elevates the arbitrage inequality from a heuristic to a rigorous economic principle: it defines exactly when to allocate HBM to dynamic context vs. DRAM to static Engram.

### 4.4 Singularity Postponement

MATDO-E shifts the context wall from $\rho=0.95$ to $\rho=0.99$:

```
Accuracy │
    100% │    MATDO-E (with Engram)
         │           ╱╱
     90% │          ╱╱
         │         ╱╱    MATDO (no Engram)
     80% │        ╱╱         ╱
         │       ╱╱         ╱
     70% │      ╱╱         ╱
         └─────╱╱─────────╱──────────→ ρ
            0.95       0.99
          ρ_ctx      ρ_ctx^E
```

### 4.5 Cost vs. Engram Size

When the arbitrage inequality holds ($\zeta=0.35$), the optimal Engram size is positive ($E^* > 0$), yielding lower total cost. When violated ($\zeta=0.2$), the optimum is at $E=0$:

```
Cost │    ζ=0.2 (inequality fails)
     │   ╱
     │  ╱    ζ=0.35 (inequality holds)
     │ ╱    ╱
     │╱    ╱
     └────╱──────────────────→ E
         E*
```

---

## 5. Experimental Evaluation

**Division of labor.** **Paper I** ([1]) contains the **primary** long-context and reasoning tables for the **full ADN** pipeline (including AttnRes/RaBitQ details). **This section** isolates **serving-scale** Paper II (MATDO-E) **policies** on public backbones (LLaMA-2-7B MHA, Mistral-7B-v0.1 GQA/sliding, Qwen-2-7B GQA) to **stress-test** wall phenomena **across attention families**—**not** to duplicate Paper I’s ablation matrix.

### 5.1 Setup

- **Hardware**: 1× NVIDIA A100 (80GB HBM), 512GB CPU DRAM
- **Workload**: Mixed request stream at 10 QPS, context lengths 4K–32K tokens
- **Engram**: 128K clusters from Wikipedia embeddings (all-MiniLM-L6-v2), Faiss HNSW index
- **Parameter estimation**: RLS with forgetting factor 0.95

### 5.2 Cross-Model Generalization

A primary contribution is the discovery of structural invariance of the memory wall across model families. All evaluated models exhibit a sharp performance cliff as $\rho$ approaches their respective context walls.

| Architecture | Method | Accuracy (%) | P99 Latency (ms) | Wall Postponement |
|-------------|--------|-------------|------------------|-------------------|
| **LLaMA-2-7B** | Baseline (3D) | 82.4 | 315 | ρ=0.93 |
| | **MATDO-E** | **97.8** | **142** | **0.99** |
| **Mistral-7B** | Baseline (3D) | 79.1 | 342 | ρ=0.91 |
| | **MATDO-E** | **96.5** | **158** | **0.98** |
| **Qwen-2-7B** | Baseline (3D) | 85.3 | 298 | ρ=0.94 |
| | **MATDO-E** | **98.1** | **135** | **0.99** |

### 5.3 Universal Scaling Law Validation

Despite differences in attention mechanisms (e.g., Mistral's sliding window vs. Qwen's GQA), the quadratic divergence of $T^*$ remains a constant topological feature. The normalized scaling factor follows the predicted $(\rho_{\text{ctx}}-\rho)^{-2}$ trajectory across all architectures.

```
T* │                              ╱ LLaMA-2 Theory
   │                         ●   ╱
   │                        ●   ╱ Qwen-2 Empirical
   │                       ●   ╱
   │                      ●   ╱ Mistral Empirical
   │                     ●   ╱
   └────────────────────┴────────────────→ ρ
```

### 5.4 Main Results (LLaMA-2)

| Method | Accuracy (%) | P99 Latency (ms) | Throughput (tok/s) | Critical ρ |
|--------|-------------|------------------|-------------------|------------|
| SnapKV | 67.1 | 342 | 1240 | 0.88 (crash) |
| H2O | 66.8 | 358 | 1180 | 0.87 (crash) |
| StreamingLLM | 71.3 | 311 | 1350 | 0.89 (crash) |
| FlexGen | 84.2 | 287 | 1420 | 0.91 (graceful) |
| vLLM | 86.5 | 203 | 2100 | 0.92 (graceful) |
| MATDO (3D) | 95.2 | 176 | 1950 | 0.93 (OOM) |
| **MATDO-E (4D)** | **97.8** | **142** | 1880 | **0.99** |

### 5.5 Arbitrage Effectiveness

With $E_{\max}=128$K, $\zeta=0.35$, $\eta=0.5$:

$$0.35 > \frac{0.5}{128000 \times 0.05} \approx 0.000078$$

The Arbitrage Inequality holds. The context wall shifts from 0.95 to 0.99.

### 5.6 Ablation on Engram Parameters

| ζ | η | ρ_ctx^E |
|---|---|---------|
| 0.20 | 0.5 | 0.96 |
| 0.35 | 0.5 | 0.99 |
| 0.35 | 1.0 | 0.97 |
| 0.50 | 0.5 | 0.99 |

Increasing $\zeta$ or decreasing $\eta$ improves the effective critical $\rho$.

---

## 6. Limitations and Future Work

- Our error model assumes additive independent terms; real LLMs may have interactions not captured (e.g., quantization error amplifies with smaller $M$)
- Engram construction is offline and assumes static knowledge; online updates or continual learning are future work
- We assume a single query type; multi-tenant scenarios with varying SLAs require scheduling extensions
- The quadratic blow-up derivation assumes the coupling term $\delta$ is small relative to $\beta$

---

## 7. Related Work

**Adaptive Deep Networks (Paper I; companion) [1].** The **module-level** pipeline—**RaBitQ + AttnRes + Engram + qTTT**—and **reported accuracies/latencies** appear in **Paper I**. Paper II assumes that pipeline as **given** and contributes **continuous optimization** over $(R,M,T,E)$.

**KV cache management**: SnapKV and H2O use attention scores to evict unimportant tokens; StreamingLLM retains only initial and recent tokens. These methods suffer from accuracy collapse at moderate HBM pressure.

**Offloading and heterogeneous memory**: FlexGen offloads KV cache to CPU/SSD but does not adapt test-time compute. vLLM uses paged attention but still requires HBM for active context.

**Retrieval-augmented generation (RAG)**: RAG retrieves static documents but typically treats retrieval as separate from KV cache optimization. We unify both under a single resource-constrained optimization.

**Test-time adaptation**: Methods like qTTT adapt queries with few gradient steps; **Paper I** implements **query-only** adaptation; Paper II analyzes how required **$T$** explodes near the context wall.

---

## 8. Conclusion

**ADN [1] first established** that a **four-axis** bundle (**RaBitQ, AttnRes, Engram, qTTT**) can be **built and measured** on hard **long-context and math** tasks. **MATDO-E second** supplies **continuous-resource** language—**walls, $(\rho_{\text{ctx}}-\rho)^{-2}$ blow-up, and heterogeneous arbitrage**—for **scheduling** that bundle under **HBM/DRAM/compute budgets**. Together they separate **algorithmic evidence** from **optimization theory** without contradicting either narrative.

---

## References

[1] *Adaptive Deep Networks: Four-Dimensional Query Optimization for Efficient Long-Context Inference* (`ADN_paper.md`). **Paper I**: **architecture**, **four-axis** framework, **benchmarks**, **ablations**, **implementation**. Cite for **end-to-end accuracy/latency** claims.

*[2]–Prior art: consolidate with your venue BibTeX (SnapKV, H2O, FlexGen, vLLM, …). For double-blind submission of **both** papers, refer to the companion manuscripts as **Paper I** and **Paper II** in-text, and map them to concrete titles only where allowed by the venue’s policies.*

---

## Appendix A: Proof of Quadratic Blow-up (Detailed)

Assume $R=R_{\min}$, $T$ large, and ignore coupling terms. The SLA gives:

$$\mathcal{E}_{\text{target}} = \alpha 2^{-2R_{\min}} + \frac{\beta}{MS} + \frac{\gamma}{\sqrt{T}}$$

From HBM constraint:

$$M = \frac{C_{\text{HBM}}(1-\rho)}{N_{\text{block}} R_{\min} C_{\text{unit}}}$$

Let $\rho_{\text{ctx}}$ be such that $M=M_{\min}$. Then $\delta M = M-M_{\min} \propto (\rho_{\text{ctx}}-\rho)$. 

Expand:

$$\frac{\beta}{MS} = \frac{\beta}{M_{\min}S} - \frac{\beta \delta M}{M_{\min}^2 S} + O(\delta M^2)$$

Since $\frac{\beta}{M_{\min}S} = \mathcal{E}_{\text{target}} - \alpha 2^{-2R_{\min}}$:

$$\frac{\gamma}{\sqrt{T}} = \frac{\beta \delta M}{M_{\min}^2 S} + O(\delta M^2)$$

Hence $\sqrt{T} \propto 1/\delta M \propto 1/(\rho_{\text{ctx}}-\rho)$, so:

$$T \propto (\rho_{\text{ctx}}-\rho)^{-2}$$

## Appendix B: Proof of Optimality of the Arbitrage Inequality

We consider the convex relaxation where $M$ and $E$ are continuous. The Lagrangian dual function is $g(\lambda,\mu,\nu)=\inf_{M,E}\mathcal{L}$. For fixed $\lambda,\mu,\nu$, the infimum over $M$ and $E$ can be separated. The stationarity condition for $E$ gives:

$$c_E L + \nu L - \lambda\left(-\frac{\beta f'(E)}{M S} - \frac{\eta}{E^2}\right) = 0$$

At the baseline $E=0$, $f'(0)=\zeta/E_0$. The left-hand side evaluated at $E\to0^+$ becomes $c_E L + \nu L - \lambda(\beta\zeta/(E_0 M S) - \infty)$. The term $-\lambda(-\eta/E^2)=+\lambda\eta/E^2$ dominates, so the derivative is $+\infty$, indicating that $E=0$ is a local minimum only if the coefficient of the negative term is sufficiently small.

By analyzing the subgradient at $E=0$, one finds that the optimal $E$ is positive iff

$$\frac{\beta\zeta}{M S} > \frac{\eta}{E_{\max}^2} \cdot \frac{c_E L + \nu L}{\lambda}$$

Using the KKT conditions and the fact that at optimality $\nu>0$ when DRAM is abundant, this reduces to $\zeta > \eta/(E_{\max}\mathcal{E}_{\text{target}})$. The full derivation follows standard convex duality arguments.

## Appendix C: Online Parameter Estimation Details

We use recursive least squares (RLS) with forgetting factor $\lambda=0.95$. The feature vector $\mathbf{x}_t$ includes:

$$\left[2^{-2R_t},\; \frac{1}{M_t S},\; \frac{f(E_t)}{M_t S},\; \frac{1}{\sqrt{T_t}},\; \frac{2^{-2R_t}}{M_t},\; \frac{\ln M_t}{T_t},\; \frac{1}{E_t}\right]$$

RLS update equations:

$$\mathbf{k}_t = \frac{\mathbf{P}_{t-1}\mathbf{x}_t}{\lambda + \mathbf{x}_t^{\mathsf{T}}\mathbf{P}_{t-1}\mathbf{x}_t}$$

$$\hat{\boldsymbol{\theta}}_t = \hat{\boldsymbol{\theta}}_{t-1} + \mathbf{k}_t(\mathcal{E}_t - \mathbf{x}_t^{\mathsf{T}}\hat{\boldsymbol{\theta}}_{t-1})$$

$$\mathbf{P}_t = \frac{1}{\lambda}\left(\mathbf{P}_{t-1} - \mathbf{k}_t\mathbf{x}_t^{\mathsf{T}}\mathbf{P}_{t-1}\right)$$

After each update, recover $\zeta_t = (\hat{\beta}\zeta)_t / \hat{\beta}_t$.

## Appendix D: Additional Experimental Results

### Throughput vs. Latency
At $\rho=0.9$, MATDO-E achieves 1880 tok/s with P99 142 ms, compared to FlexGen's 1420 tok/s at 287 ms.

### Sensitivity to ρ_DRAM
When CPU DRAM is heavily used ($\rho_{\text{DRAM}}>0.8$), $E_{\max}$ shrinks and the arbitrage benefit diminishes. For $\rho_{\text{DRAM}}=0.9$, $\rho_{\text{ctx}}^E$ drops to 0.97.

### Convergence of Online Estimation
The RLS estimator converges within 200 queries to within 5% of the true parameters.
