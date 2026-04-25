# PRISM 框架补充实验验证设计文档

**文档目的**：回应 ICLR Review 中提出的需实验佐证的五个核心问题（Q1–Q5），为返修阶段提供可直接执行的实验方案。每个实验均包含：目标、设置、预期样本量、测量指标、代码框架与判定准则。

---

## 实验1：误差模型函数形式的拟合对比（回应 Q1）

### 目标
验证第3.3节中加性误差模型各分量的函数形式选择（$2^{-2q}$、$1/\sqrt{t}$、$1/c$）是否显著优于合理的替代形式（$e^{-\kappa q}$、$1/t$、$e^{-\nu c}$）。若当前形式并非最优，需根据实验结果修正误差方程并重新评估定理1的鲁棒性。

### 实验设置

**平台**：单节点 A100 80GB，PyTorch 2.2+，vLLM 0.4+。
**模型**：LLaMA-2-7B（主实验），Mistral-7B-v0.1（跨模型验证）。
**数据集**：LongBench 子集（2K–32K 上下文任务，避免极端长上下文引入高阶耦合）。
**基线配置**：$q=16$（FP16）、$c=c_{\max}$、$t=0$、$e=0$。

### 扫描网格

| 变量 | 扫描值 | 固定其他变量 |
|------|--------|-------------|
| $q$ | {2, 3, 4, 6, 8, 16} | $c=c_{\max}$, $t=0$, $e=0$ |
| $t$ | {1, 2, 4, 8, 16, 32} | $q=8$, $c=c_{\max}$, $e=0$ |
| $c$ | {$0.05, 0.1, 0.2, 0.5, 1.0$} $\times c_{\max}$ | $q=8$, $t=0$, $e=0$ |

### 测量指标

对每个配置运行3次，记录：
- **误差代理**：$\mathcal{E} = 1 - \text{Accuracy}_{\text{LongBench}}$（归一化到$[0,1]$）。
- **拟合优度**：非线性最小二乘（`scipy.optimize.curve_fit`，Levenberg-Marquardt）得到调整$R^2$、AICc、BIC。
- **交叉验证**：留一法（LOO）残差均方根（RMSE）。

### 拟合模型列表

对每个维度，拟合以下候选模型：

**量化维度**：
- M1（当前）：$\mathcal{E}(q) = \alpha \cdot 2^{-2q} + C_0$
- M2（替代）：$\mathcal{E}(q) = \alpha \cdot e^{-\kappa q} + C_0$
- M3（替代）：$\mathcal{E}(q) = \alpha \cdot q^{-1} + C_0$

**适配维度**：
- M1（当前）：$\mathcal{E}(t) = \gamma / \sqrt{t} + C_0$
- M2（替代）：$\mathcal{E}(t) = \gamma / t + C_0$
- M3（替代）：$\mathcal{E}(t) = \gamma \cdot e^{-\mu t} + C_0$

**上下文维度**：
- M1（当前）：$\mathcal{E}(c) = \beta / c + C_0$
- M2（替代）：$\mathcal{E}(c) = \beta / c^2 + C_0$
- M3（替代）：$\mathcal{E}(c) = \beta \cdot e^{-\nu c} + C_0$

### 判定准则

1. **若 M1 的调整 $R^2$ 在所有维度均显著优于 M2/M3（差距 > 0.03 或 AICc 低 > 10）**：
   - 结论：当前函数形式选择合理。
   - 论文修改：第3.3节补充一段拟合对比摘要，附录C.1放完整表格。

2. **若 M2 或 M3 在某一维度显著优于 M1**：
   - 结论：该维度函数形式需修正。
   - 论文修改：修正第3.3节方程，更新定理1证明（通常不影响定性结论，因为定理依赖单调性和发散行为，而非精确幂次）。若适配维度变为$1/t$，则近墙发散可能从二次变为线性，需修正第4.5节并更新实测标度预期。

3. **若不同模型间最优形式不一致**：
   - 结论：函数形式具有模型依赖性。
   - 论文修改：第3.3节增加"模型敏感性"讨论，将当前形式定位为7B模型的经验最优拟合，并建议更大规模模型需重新标定。

### 伪代码框架

```python
from scipy.optimize import curve_fit
import numpy as np

# 误差代理：1 - accuracy
def load_results(grid_config):
    # 从vLLM运行时日志加载accuracy
    return accuracy_grid

# 候选模型
models = {
    'quant': {
        'm1': lambda q, a, C0: a * 2**(-2*q) + C0,
        'm2': lambda q, a, k, C0: a * np.exp(-k*q) + C0,
        'm3': lambda q, a, C0: a / q + C0,
    },
    'adapt': {
        'm1': lambda t, g, C0: g / np.sqrt(t) + C0,
        'm2': lambda t, g, C0: g / t + C0,
        'm3': lambda t, g, m, C0: g * np.exp(-m*t) + C0,
    },
    'context': {
        'm1': lambda c, b, C0: b / c + C0,
        'm2': lambda c, b, C0: b / c**2 + C0,
        'm3': lambda c, b, n, C0: b * np.exp(-n*c) + C0,
    }
}

def fit_and_eval(dim, x, y):
    results = {}
    for name, func in models[dim].items():
        try:
            popt, _ = curve_fit(func, x, y, maxfev=10000)
            y_pred = func(x, *popt)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res / ss_tot
            # AICc, LOO-RSME 略
            results[name] = {'R2_adj': r2, 'params': popt}
        except Exception as e:
            results[name] = {'error': str(e)}
    return results

# 执行拟合
for dim in ['quant', 'adapt', 'context']:
    x, y = load_scan_data(dim)
    print(fit_and_eval(dim, x, y))
```

---

## 实验2：外部记忆规模 $e$ 的直接参数扫描（回应 Q4）

### 目标
直接验证第4.6节的套利条件 "$A/B < A_0/B_0 \iff$ 外部记忆推迟上下文墙"，而非仅依赖固定$e$的间接证据。

### 实验设置

**平台**：同实验1，额外配置 512GB CPU DRAM 与 Faiss HNSW 索引（`faiss-cpu`，`hnsw` 参数 `M=16`, `efConstruction=200`）。
**模型**：LLaMA-2-7B。
**外部记忆表**：Wikipedia embeddings 子集，按 $e$ 规模截断。
**固定配置**：$q=6$（中等量化），$t=4$（浅适配），避免高阶耦合干扰。
**目标误差**：$E_{\text{target}}$ 设为 PRISM-3D 在 $c_{\max}$ 时的误差（即"无压缩无记忆基线"的允许退化阈值）。

### 扫描网格

- $e \in \{0, 32K, 64K, 128K, 256K, 512K\}$ 条目。
- 对每个 $e$，通过**二分搜索**找到满足 $E_{\text{target}}$ 的最小 $c$（即 $c_{\min}(e)$）。
- 二分搜索范围：$c \in [c_{\min}^{(0)} / 4, c_{\max}]$，容差 $0.01 \times c_{\max}$。

### 测量指标

对每个 $(e, c)$ 配置运行5次：
- **准确率**：LongBench 平均准确率 $\to \mathcal{E} = 1 - \text{acc}$。
- **检索质量**：Top-5 检索命中率（用于估计 $\zeta$）。
- **时延**：P99 端到端时延（分解为生成时延 + 检索时延）。
- **HBM 占用**：nvidia-smi 峰值内存（用于计算实际 $\rho$）。

### 判定准则

1. **若 $c_{\min}(e)$ 随 $e$ 单调不增，且在 $e \geq 128K$ 后趋于饱和**：
   - 结论：外部记忆确实提供覆盖收益，且边际收益递减。
   - 论文修改：第4.6节和摘要增加"附录C.2直接验证"的引用；在图/表中增加 $c_{\min}$ 对 $e$ 的曲线。

2. **若存在某一 $e$ 使得 $c_{\min}(e) > c_{\min}(0)$（即外部记忆反而损害性能）**：
   - 结论：检索噪声过大，存在"负套利"区域。
   - 论文修改：第4.6节增加"负套利"讨论，修正 $f(e)$ 为带阈值的S型函数（如 $f(e) = \max(0, 1 - \zeta(1-e^{-e/e_0}))$），并增加检索质量门控的工程建议。

3. **若 $c_{\min}(e)$ 基本不变（$e$ 不影响墙位）**：
   - 结论：在当前检索质量下，外部记忆无法有效补偿上下文缩减。
   - 论文修改：弱化第4.6节的普遍性声称，将其改为"在高质量检索条件下"的条件结论，并讨论检索质量的决定性作用。

### 伪代码框架

```python
import faiss
import numpy as np

# 构建外部记忆索引
class ExternalMemory:
    def __init__(self, embeddings, e_size):
        subset = embeddings[:e_size]
        self.index = faiss.IndexHNSWFlat(d, 16)
        self.index.add(subset)

    def retrieve(self, query, k=5):
        D, I = self.index.search(query, k)
        return I, D

# 二分搜索最小可行c
def find_c_min(model, E_target, e, q=6, t=4):
    lo, hi = c_min_estimated // 4, c_max
    while hi - lo > 0.01 * c_max:
        mid = (lo + hi) / 2
        acc = evaluate(model, q=q, c=mid, t=t, e=e)
        E = 1 - acc
        if E <= E_target:
            hi = mid
        else:
            lo = mid
    return hi

# 主扫描
for e in [0, 32_000, 64_000, 128_000, 256_000, 512_000]:
    c_min = find_c_min(llama2, E_target, e)
    rho_ctx = compute_rho_exact(c_min, q=6)  # 使用精确HBM公式
    print(f"e={e}: c_min={c_min:.2f}, rho_ctx={rho_ctx:.3f}")
```

---

## 实验3：$t_{\text{req}}$ 与 $t_{\text{max}}$ 的直接分离测量（回应 Q3 / 墙顺序）

### 目标
直接测量"达到目标误差所需的适配步数 $t_{\text{req}}$"与"时延预算允许的最大适配步数 $t_{\text{max}}$"，绘制两者的交点图，将墙顺序的验证从"间接时延观测"升级为"直接预算分离测量"。

### 实验设置

**平台**：同实验1，额外要求系统级时延精确测量（`CUDA events` + `torch.cuda.synchronize()`）。
**模型**：LLaMA-2-7B。
**固定配置**：$q = q_{\min} = 4$（4-bit量化，保证HBM压力主要来自$c$）。
**$c$ 扫描**：在 $[c_{\min}, c_{\max}]$ 中取15个点，对数均匀分布。

### 测量协议

对每个 $c$：

**测量 $t_{\text{max}}(c)$**：
1. 在固定 $c$ 和 $q=4$ 下，逐步增加 $t$（$t=1,2,4,8,16,32,64$）。
2. 对每个 $t$ 测量端到端 P99 时延（100次重复）。
3. 定义 $t_{\text{max}}$ 为满足 P99 时延 $< B_{\text{latency}}$ 的最大 $t$。本文使用 $B_{\text{latency}} = 200$ ms（对应表3中PRISM-3D的P99水平）。

**测量 $t_{\text{req}}(c)$**：
1. 在固定 $c$ 和 $q=4$ 下，逐步增加 $t$。
2. 对每个 $t$ 测量 LongBench 准确率。
3. 定义 $t_{\text{req}}$ 为满足 $\mathcal{E} \leq E_{\text{target}}$ 的最小 $t$。

### 判定准则

1. **若两曲线在某一 $c_{\text{comp}}$ 处相交，且对应的 $\rho_{\text{comp}}$ 与表3的理论预测偏差 $< 0.03$**：
   - 结论：墙顺序得到直接实验验证。
   - 论文修改：第6.5节增加 $t_{\text{req}}$-$t_{\text{max}}$ 交点图；将"间接时延证据"表述升级为"直接预算分离测量"。

2. **若 $t_{\text{req}}(c) < t_{\text{max}}(c)$ 对所有扫描 $c$ 恒成立（无交点）**：
   - 结论：系统在观测范围内始终满足目标误差而不触碰计算边界；表3中的 $\rho_{\text{comp}}$ 实为外推预测。
   - 论文修改：第6.3节表3中将 $\rho_{\text{comp}}$ 标注改为"extrapolated"；第4.4节增加"在本文实验的 $\rho$ 范围内，计算边界未被直接触及，定理1预测为外推"的说明。

3. **若交点存在但位置与理论预测偏差 $> 0.05$**：
   - 结论：计算约束方程（第3.4节）遗漏了系统级开销。
   - 论文修改：检查 $t_{\text{max}}$ 公式，增加修正项（如梯度同步开销 $t_{\text{sync}} \approx 2$ ms/步，或KV缓存重排开销），并在第3.4节增加"有效计算预算" $B_{\max}^{\text{eff}} = B_{\max} - O_{\text{sys}}$ 的说明。

### 伪代码框架

```python
import torch

def measure_t_max(model, c, q, latency_budget_ms=200, num_runs=100):
    for t in [1, 2, 4, 8, 16, 32, 64]:
        latencies = []
        for _ in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_inference(model, q=q, c=c, t=t)
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
        p99 = np.percentile(latencies, 99)
        if p99 > latency_budget_ms:
            return t // 2  # 返回上一个满足预算的t
    return 64

def measure_t_req(model, c, q, E_target, max_t=64):
    for t in range(1, max_t + 1):
        acc = evaluate_longbench(model, q=q, c=c, t=t)
        E = 1 - acc
        if E <= E_target:
            return t
    return float('inf')

# 主扫描
c_values = np.logspace(np.log10(c_min), np.log10(c_max), 15)
t_reqs, t_maxs = [], []
for c in c_values:
    t_reqs.append(measure_t_req(model, c, q=4, E_target=0.05))
    t_maxs.append(measure_t_max(model, c, q=4))

# 寻找交点
# 绘制 t_req(c), t_max(c) 曲线
```

---

## 实验4：异构硬件瓶颈的 Profiler 验证（回应 Q5）

### 目标
验证第3.4节提出的"计算预算应分解为FLOPs、HBM带宽、DRAM带宽、PCIe带宽"的论断，确定在不同 $(q,c,t,e)$ 区域中实际的硬件瓶颈子系统。

### 实验设置

**平台**：A100 80GB，Nsight Systems 2024.x，PyTorch 2.2+（编译时启用 `USE_CUDA=1`）。
**模型**：LLaMA-2-7B。
**Profiler 配置**：`nsys profile -t cuda,nvtx,osrt -s none -o report.qdrep python run_inference.py`。

### 扫描网格（瓶颈分区设计）

选择6个代表性配置，预期覆盖不同瓶颈区域：

| 配置 | $q$ | $c$ | $t$ | $e$ | 预期瓶颈 |
|------|-----|-----|-----|-----|---------|
| A（带宽型解码） | 8 | $c_{\max}$ | 0 | 0 | HBM带宽 |
| B（FLOP型适配） | 8 | $c_{\max}$ | 16 | 0 | TensorCore FLOPs |
| C（PCIe型检索） | 8 | $c_{\max}/2$ | 0 | 256K | PCIe/DRAM带宽 |
| D（混合型近墙） | 4 | $c_{\min} \times 1.2$ | 8 | 128K | 混合 |
| E（极致量化） | 2 | $c_{\max}$ | 0 | 0 | HBM带宽（但计算密度变化） |
| F（无压缩基线） | 16 | $c_{\max}$ | 0 | 0 | HBM带宽 |

### 测量指标

从 Nsight Systems 报告中提取：
- **HBM 带宽利用率**：`dram__bytes.sum / elapsed_time` vs A100峰值带宽（2039 GB/s）。
- **TensorCore FLOPs**：`sm__inst_executed_pipe_tensor.sum` $\times$ 每指令FLOPs / elapsed_time。
- **PCIe 带宽**：`pcie__bytes.sum` / elapsed_time。
- **CPU-DRAM 带宽**：通过 `perf` 或 `pcm-memory` 采样。
- **瓶颈判定**：若某子系统利用率 > 80%，判定为该子系统瓶颈；若多子系统均 > 60%，判定为混合瓶颈。

### 判定准则

1. **若配置A/F为HBM带宽瓶颈，B为FLOP瓶颈，C为PCIe瓶颈**：
   - 结论：第3.4节的异构预算分解具有实证基础。
   - 论文修改：第3.4节增加"附录C.4的profiler分析验证"的引用；将标量 $B_{\max}$ 重新表述为"在混合约束下的简化聚合"。

2. **若配置C仍显示GPU FLOP为主瓶颈（PCIe未饱和）**：
   - 结论：检索在CPU完成且数据预取充分，GPU未感知PCIe压力。
   - 论文修改：第3.4节调整 $k_e$ 的物理含义说明，指出"当检索异步执行时，$k_e$ 主要表征DRAM带宽而非PCIe"。

3. **若配置D（近墙混合区）呈现不可预测的瓶颈跳变**：
   - 结论：近墙区域的异构行为复杂，标量预算近似误差最大。
   - 论文修改：第4.5节增加"近墙区域瓶颈跳变"的说明，将二次发散模型定位为"HBM压力主导假设下的近似"。

### 伪代码框架

```python
# 使用 PyTorch Profiler + NVTX 标记
import torch.profiler as profiler

def profile_config(model, q, c, t, e):
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        # NVTX 区间标记各子系统
        torch.cuda.nvtx.range_push("attention_decode")
        run_attention(model, q, c)
        torch.cuda.nvtx.range_pop()

        if t > 0:
            torch.cuda.nvtx.range_push("test_time_adapt")
            run_adapter(model, t)
            torch.cuda.nvtx.range_pop()

        if e > 0:
            torch.cuda.nvtx.range_push("external_retrieval")
            run_retrieval(model, e)
            torch.cuda.nvtx.range_pop()

    # 导出给 Nsight Systems 做更细粒度分析
    prof.export_chrome_trace(f"trace_q{q}_c{c}_t{t}_e{e}.json")
    return prof

# 主扫描
configs = [
    ('A', 8, c_max, 0, 0),
    ('B', 8, c_max, 16, 0),
    ('C', 8, c_max//2, 0, 256_000),
    ('D', 4, int(c_min*1.2), 8, 128_000),
    ('E', 2, c_max, 0, 0),
    ('F', 16, c_max, 0, 0),
]
for name, q, c, t, e in configs:
    profile_config(model, q, c, t, e)
    # 后续用 nsys 分析 .qdrep
```

---

## 实验5：精确存储模型下墙位置的重新计算（回应 Q2）

### 目标
使用第3.4节修正后的精确HBM公式（含 $L_{\text{layers}}$、$n_h$、$d_h$、batch size），重新计算表3中的 $\rho_{\text{ctx}}$ 和 $\rho_{\text{comp}}$，评估简化公式引入的偏差。

### 实验设置

**无需新运行**：本实验为**重分析（re-analysis）**，基于已有实验的 $c_{\min}$ 和 $c_{\text{comp}}$ 测量值，用两种公式分别计算 $\rho$。
**输入数据**：表3中各配置下的 $c_{\min}$、$c_{\text{comp}}$、$q_{\min}$。
**公式对比**：
- **简化公式**：$\rho = 1 - c q N_{\text{block}} C_{\text{unit}} / C_{\text{HBM}}$
- **精确公式**：$\rho = 1 - 2 L_{\text{layers}} n_h d_h B_{\text{batch}} (c N_{\text{block}}) (q/8) / C_{\text{HBM}}$

### 判定准则

1. **若两公式偏差 $< 0.01$（对 $B_{\text{batch}}=1$ 且 $C_{\text{unit}}$ 恰好校准）**：
   - 结论：简化公式在单序列场景下是精确公式的良好近似。
   - 论文修改：第3.4节说明"$C_{\text{unit}}$ 在此被校准为等效字节数"，表3保留当前数值。

2. **若偏差 $> 0.03$**：
   - 结论：简化公式显著低估/高估了HBM占用。
   - 论文修改：用精确公式重新计算表3的所有$\rho$值，并更新正文第4.2–4.3节的公式引用。

3. **若 batch size 增加时偏差急剧扩大**：
   - 结论：简化公式无法扩展至多序列并发。
   - 论文修改：第3.4节显式增加"本分析针对 $B_{\text{batch}}=1$；多序列并发需将 $B_{\text{batch}}$ 纳入"的说明。

### 伪代码框架

```python
def rho_approx(c, q, N_block=256, C_unit=2, C_HBM=80e9):
    return 1 - c * q * N_block * C_unit / C_HBM

def rho_exact(c, q, L_layers=32, n_h=32, d_h=128,
              N_block=256, B_batch=1, C_HBM=80e9):
    bytes_used = 2 * L_layers * n_h * d_h * B_batch * (c * N_block) * (q / 8)
    return 1 - bytes_used / C_HBM

# 重分析表3数据
table3_data = {
    'LLaMA-2-7B': {'PRISM-3D': {'c_min': ..., 'q_min': 4},
                   'PRISM-full': {'c_min': ..., 'q_min': 4}},
    # ...
}

for model, configs in table3_data.items():
    for cfg, vals in configs.items():
        rho_a = rho_approx(vals['c_min'], vals['q_min'])
        rho_e = rho_exact(vals['c_min'], vals['q_min'])
        print(f"{model}/{cfg}: approx={rho_a:.3f}, exact={rho_e:.3f}, diff={abs(rho_a-rho_e):.3f}")
```

---

## 实验执行优先级与时间安排建议

| 优先级 | 实验 | 预估时间 | 阻塞风险 |
|--------|------|---------|---------|
| P0 | 实验1（函数形式对比） | 2–3 GPU天 | 低（纯自包含扫描） |
| P0 | 实验2（$e$ 参数扫描） | 3–4 GPU天 | 中（依赖Faiss索引构建） |
| P1 | 实验3（$t_{\text{req}}$/$t_{\text{max}}$分离） | 2 GPU天 | 低 |
| P1 | 实验5（精确公式重分析） | 4小时（纯计算） | 无 |
| P2 | 实验4（Profiler瓶颈分区） | 1 GPU天 | 低（但需Nsight安装） |

**建议执行顺序**：实验5（立即执行，无成本）$\to$ 实验1 + 实验2（并行）$\to$ 实验3 $\to$ 实验4（可选）。

---

## 通用数据管理与复现要求

1. **日志格式**：所有实验输出使用统一JSON Lines格式，字段至少包含 `{model, q, c, t, e, accuracy, latency_ms, hbm_bytes, timestamp}`。
2. **随机种子**：PyTorch `torch.manual_seed(42)`，CUDA `torch.cuda.manual_seed_all(42)`，Faiss `faiss.seed(42)`。
3. **版本锁定**：PyTorch 2.2.x、Transformers 4.38.x、vLLM 0.4.x、Faiss 1.7.x。
4. **数据归档**：原始profiler trace（.qdrep）和chrome trace（.json）需上传至项目存储，供审稿人抽查。
