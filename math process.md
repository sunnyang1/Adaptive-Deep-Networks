# 数学推导过程

我们一步步推导第三章和第四章的核心数学公式。从最基础的假设开始，详细写出每个等式的来源，并解释每一步的物理含义。

## 一、基础设定与误差模型

我们有四个可调变量：$R$（量化比特数）、$M$（HBM中保留的块数）、$T$（自适应步数）、$E$（Engram表大小）。

总推理误差模型为（式1）：

# $$

\mathcal{E}(R,M,T,E)

\alpha 2^{-2R}

- \frac{\beta f(E)}{MS}
- \frac{\gamma}{\sqrt{T}}
- \delta \frac{2^{-2R}}{M}
- \epsilon \frac{\ln M}{T}
- r(E)
$$

其中：

- $\alpha, \beta, \gamma, \delta, \epsilon, \eta$ 是正常数（通过实验拟合）
- $S$ 是每个块的大小（元素个数）
- $f(E) = 1 - \zeta(1 - e^{-E/E_0})$，$\zeta\in[0,1]$ 是 Engram 最大补偿比例，$E_0$ 是饱和常数
- $r(E)=0$ 当 $E=0$，否则 $r(E)=\eta/E$

目标：给定目标精度 $\mathcal{E}_{\text{target}}$，在硬件约束下最小化总成本（或等价地，在给定硬件下最大化吞吐）。我们将分析逼近硬件极限时的行为。

## 二、无 Engram 情形（$E=0$）的双临界点推导

此时 $r(0)=0$，$f(0)=1$。为了找到理论极限，我们考虑系统在极端压力下的配置：

- 量化比特数已降到最低允许值 $R = R_{\min}$（不能再低，否则量化误差直接超标）
- 自适应步数 $T$ 可以任意大（分析 $T \to \infty$ 的行为）
- 忽略小的耦合项 $\delta$ 和 $\epsilon$（它们对渐近阶数无影响，后文会验证）

简化后的误差模型：

$$
\mathcal{E}
\approx
\alpha 2^{-2R_{\min}}

- \frac{\beta}{MS}
- \frac{\gamma}{\sqrt{T}}
$$

### 2.1 推导最小块数 $M_{\min}$

设目标精度为 $\mathcal{E}_{\text{target}}$。当 $T \to \infty$ 时，特异性误差项 $\gamma/\sqrt{T}\to 0$，于是：

# $$

\mathcal{E}_{\text{target}}

\alpha 2^{-2R_{\min}}

- \frac{\beta}{M_{\min}S}
$$

解出：

# $$

M_{\min}

\frac{\beta}{S\left(\mathcal{E}*{\text{target}}-\alpha 2^{-2R*{\min}}\right)}
$$

物理含义：为了在“自适应误差项在极限下消失”（$T\to\infty$）时达到目标精度，至少需要保留 $M_{\min}$ 个块。若 $M < M_{\min}$，即使 $T \to \infty$ 也无法弥补范围误差缺失（自适应只能减少特异性误差，不能恢复被丢弃的上下文信息）。

若保留耦合项 $\delta 2^{-2R}/M$，分子变为 $\beta+\delta 2^{-2R_{\min}}$，即论文式(2)。

### 2.2 定义上下文墙 $\rho_{\text{ctx}}$

HBM 容量约束：

$$
M \cdot N_{\text{block}} \cdot R_{\min}\cdot C_{\text{unit}}
\le
C_{\text{HBM}}(1-\rho)
$$

其中 $\rho$ 是已被模型权重等占用的 HBM 比例，$(1-\rho)$ 是剩余可用比例。

当 $M=M_{\min}$ 时取等号，得临界利用率：

# $$

\rho_{\text{ctx}}

1-\frac{M_{\min}N_{\text{block}}R_{\min}C_{\text{unit}}}{C_{\text{HBM}}}
$$

物理含义：若 $\rho>\rho_{\text{ctx}}$，即使把所有剩余 HBM 都用于 KV 缓存，也装不下 $M_{\min}$ 个块，因此系统无法满足精度要求（与计算资源无关）。

### 2.3 推导 $T^*$ 与 $\rho$ 的关系（二次发散）

当 $\rho<\rho_{\text{ctx}}$ 时，HBM 中实际可容纳块数：

# $$

M^*(\rho)

\frac{C_{\text{HBM}}(1-\rho)}
{N_{\text{block}}R_{\min}C_{\text{unit}}}
$$

精度约束：

# $$

\mathcal{E}_{\text{target}}

\alpha 2^{-2R_{\min}}

- \frac{\beta}{M^*S}
- \frac{\gamma}{\sqrt{T^*}}
$$

移项：

# $$

\frac{\gamma}{\sqrt{T^*}}

\mathcal{E}*{\text{target}}
-\alpha 2^{-2R*{\min}}
-\frac{\beta}{M^*S}
$$

由 $M_{\min}$ 定义有
$\mathcal{E}*{\text{target}}-\alpha2^{-2R*{\min}}=\frac{\beta}{M_{\min}S}$，代入得

# $$

\frac{\gamma}{\sqrt{T^*}}

\frac{\beta}{S}\left(\frac{1}{M_{\min}}-\frac{1}{M^*}\right)
$$

令 $\delta_M=M^*-M_{\min}$。当 $\rho\to\rho_{\text{ctx}}^-$ 时，$\delta_M$ 很小，对 $1/M^*$ 在 $M_{\min}$ 一阶展开：

$$
\frac{1}{M^*}
\approx
\frac{1}{M_{\min}}-\frac{\delta_M}{M_{\min}^2}
$$

因此：

$$
\frac{\gamma}{\sqrt{T^*}}
\approx
\frac{\beta}{S}\cdot\frac{\delta_M}{M_{\min}^2}
$$

再求 $\delta_M$ 与 $\rho$ 的关系：

# $$

\delta_M

# M^*-M_{\min}

\frac{C_{\text{HBM}}}{N_{\text{block}}R_{\min}C_{\text{unit}}}
(\rho_{\text{ctx}}-\rho)
\propto (\rho_{\text{ctx}}-\rho)
$$

于是：

$$
\frac{\gamma}{\sqrt{T^*}}
\propto
(\rho_{\text{ctx}}-\rho)
\Rightarrow
\sqrt{T^*}\propto \frac{1}{\rho_{\text{ctx}}-\rho}
\Rightarrow
T^*\propto (\rho_{\text{ctx}}-\rho)^{-2}
$$

结论：当 HBM 压力逼近上下文墙时，为维持精度所需的自适应步数以平方反比速度爆炸，这就是性能悬崖的数学根源。

### 2.4 证明 $\rho_{\text{comp}} < \rho_{\text{ctx}}$

系统有计算延迟预算 $\mathcal{B}*{\max}$，因此 $T$ 有上限 $T*{\max}$。由于 $T^*(\rho)$ 在 $\rho\to\rho_{\text{ctx}}$ 时发散，必存在某个 $\rho_{\text{comp}}<\rho_{\text{ctx}}$ 使得 $T^*(\rho_{\text{comp}})=T_{\max}$。对 $\rho>\rho_{\text{comp}}$，所需 $T^*$ 超预算，无法满足 SLA。因此计算墙先于上下文墙到达。

## 三、引入 Engram 后的套利推导

现在 $E>0$。为简化，固定 $E=E_{\max}$（DRAM 可分配最大条目数），并取 $T\to\infty$ 极限以找新的最小块数。

忽略耦合项、保留检索开销时：

# $$

\mathcal{E}_{\text{target}}

\alpha 2^{-2R_{\min}}
+\frac{\beta f(E_{\max})}{M_{\min}^E S}
+\frac{\eta}{E_{\max}}
$$

注意 $\eta/E_{\max}$ 是固定检索惩罚，会直接占用误差预算。

解得：

# $$

M_{\min}^E

\frac{\beta f(E_{\max})}
{S\left(\mathcal{E}*{\text{target}}-\alpha2^{-2R*{\min}}-\eta/E_{\max}\right)}
$$

若保留耦合项，分子再加 $\delta2^{-2R_{\min}}$，即论文式(4)。

### 3.1 推导套利不等式

Engram 能推迟上下文墙当且仅当 $M_{\min}^E < M_{\min}$。

写成不等式：

$$
\frac{\beta f(E_{\max})}
{S\left(\mathcal{E}*{\text{target}}-\alpha2^{-2R*{\min}}-\eta/E_{\max}\right)}
<
\frac{\beta}
{S\left(\mathcal{E}*{\text{target}}-\alpha2^{-2R*{\min}}\right)}
$$

约去 $\beta/S$，令
$D=\mathcal{E}*{\text{target}}-\alpha2^{-2R*{\min}}$，并要求
$D>0$ 且 $D-\eta/E_{\max}>0$（即引入检索惩罚后仍有正的可用误差预算），得

$$
\frac{f(E_{\max})}{D-\eta/E_{\max}}<\frac{1}{D}
\Longleftrightarrow
D\left(1-f(E_{\max})\right)>\frac{\eta}{E_{\max}}
$$

由
$f(E)=1-\zeta(1-e^{-E/E_0})$，当 $E_{\max}$ 足够大时，
$e^{-E_{\max}/E_0}\approx0$，故
$f(E_{\max})\approx1-\zeta$，即
$1-f(E_{\max})\approx\zeta$。于是

$$
D\zeta>\frac{\eta}{E_{\max}}
\Longrightarrow
\zeta>\frac{\eta}{E_{\max}D}
\approx
\frac{\eta}{E_{\max}\mathcal{E}_{\text{target}}}
$$

其中最后一步近似使用了 $\alpha2^{-2R_{\min}}\ll\mathcal{E}*{\text{target}}$，即
$D=\mathcal{E}*{\text{target}}-\alpha2^{-2R_{\min}}\approx\mathcal{E}_{\text{target}}$。

物理含义：左边 $\zeta$ 是 Engram 对范围误差的补偿能力，右边是检索开销相对目标误差的归一化成本。当补偿能力大于归一化成本时，用 DRAM 换 HBM 是划算的，上下文墙可被推迟。

### 3.2 关于必要性与凸对偶性

上述推导先给出充分条件。要得到必要性（在给定松弛下），可将原问题写成对 $(M,E)$ 的成本最优化问题，通过变量替换 $x=1/M,z=1/E$，并对 $f(E)$ 做凹松弛后得到凸问题。利用 KKT 条件分析 $E=0$ 边界方向导数：

- 若不等式不成立，最优解停在边界 $E=0$；
- 若不等式成立，存在内点最优解 $E>0$。

因此在“凸松弛 + KKT 正则条件（如 Slater 条件）+ 大表近似”这一组假设下，该不等式可作为充要判据。

### 3.3 数值例子

实验估计：

- $\zeta=0.35$
- $\eta=0.5$
- $E_{\max}=128{,}000$
- $\mathcal{E}_{\text{target}}=0.05$

右侧：

# $$

\frac{0.5}{128000\times0.05}

\frac{0.5}{6400}
\approx
0.000078
$$

显然 $0.35>0.000078$ 成立，因此 Engram 有效，上下文墙可从约 $0.95$ 推迟到约 $0.99$。

## 四、总结关键步骤

1. 建立加性误差模型，将各组件贡献分离。
2. 固定 $R=R_{\min}$，令 $T\to\infty$，求满足精度的最小 $M$（$M_{\min}$）。
3. 由 HBM 约束将 $M_{\min}$ 映射为临界利用率 $\rho_{\text{ctx}}$。
4. 在 $\rho<\rho_{\text{ctx}}$ 时，用 $M^*(\rho)$ 反解所需 $T^*$，并经一阶展开得到
  $T^*\propto(\rho_{\text{ctx}}-\rho)^{-2}$。
5. 引入 $E>0$，重算 $M_{\min}^E$，比较 $M_{\min}^E$ 与 $M_{\min}$ 得到套利不等式。
6. 用凸对偶性在松弛下将该条件扩展为充要判据。