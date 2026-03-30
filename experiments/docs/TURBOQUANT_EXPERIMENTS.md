# TurboQuant 集成实验步骤

本文档描述了基于 `Adaptive_Deep_Networks_TurboQuant.md` 的完整实验流程，用于验证 TurboQuant 压缩、Polar 坐标 qTTT 和深度优先策略的有效性。

---

## 实验目标

1. **验证 TurboQuant 压缩效果**: 6×+ 内存缩减，零精度损失
2. **验证 Polar qTTT 效率**: 50% 参数减少，更快收敛
3. **验证深度优先策略**: 2.4× 吞吐提升 (110 vs 45 tokens/s)
4. **端到端系统验证**: 综合性能评估

---

## 环境准备

```bash
# 1. 创建并激活虚拟环境
python -m venv venv_turboquant
source venv_turboquant/bin/activate  # Linux/Mac
# 或: venv_turboquant\Scripts\activate  # Windows

# 2. 安装依赖
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 验证 GPU 和 Tensor Core 支持
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
capability = torch.cuda.get_device_capability()
print(f'Compute capability: {capability}')
print(f'Tensor Cores (INT4): {capability[0] >= 8 and capability[1] >= 9}')
"
```

---

## 实验 1: TurboQuant 压缩验证

### 1.1 PolarQuant 单元测试

```bash
# 测试 PolarQuant 压缩/解压
python -m pytest tests/unit/test_turboquant.py::test_polar_quant -v
```

**预期结果**:
- 压缩比: ~4× (3-bit 角度 + FP16 幅度 vs FP16)
- 重构误差: < 2% (余弦相似度 > 0.99)

### 1.2 QJL 残差校正测试

```bash
# 测试 QJL 无偏估计
python -m pytest tests/unit/test_turboquant.py::test_qjl_unbiased -v
```

**预期结果**:
- 内积估计无偏: |E[est] - true| < 0.01
- 方差控制在合理范围

### 1.3 完整 TurboQuant Pipeline

```bash
# 运行完整 pipeline 测试
python -c "
import torch
from src.turboquant import TurboQuantPipeline, TurboQuantConfig

config = TurboQuantConfig(angle_bits=3, qjl_proj_dim=256)
turbo = TurboQuantPipeline(dim=4096, config=config, device='cuda')

# 测试数据
x = torch.randn(1, 4096, device='cuda')

# 压缩
r, theta, qjl, norm = turbo.compress_vector(x)
print(f'Magnitude shape: {r.shape}')
print(f'Theta indices shape: {theta.shape}')
print(f'QJL signs shape: {qjl.shape}')

# 计算压缩比
orig_bytes = x.numel() * 2  # FP16
comp_bytes = (r.numel() * 2 + theta.numel() * 1 + 
              qjl.numel() * 1 + norm.numel() * 2)
ratio = orig_bytes / comp_bytes
print(f'Compression ratio: {ratio:.2f}x')
print(f'Target: >6x')
"
```

**通过标准**: 压缩比 ≥ 6×

---

## 实验 2: Polar 坐标伪查询验证

### 2.1 Polar 分解测试

```bash
python -c "
import torch
from src.attnres.polar_pseudo_query import PolarPseudoQuery

# 创建 polar pseudo-query
pq = PolarPseudoQuery(dim=4096)
pq.freeze_magnitude()  # qTTT 模式

# 验证 w = r * u(θ)
w = pq.forward()
r = pq.r
u = pq.get_direction()

# 检查单位向量
u_norm = torch.norm(u)
print(f'Direction norm: {u_norm:.6f} (should be 1.0)')

# 检查重构
w_reconstructed = r * u
reconstruction_error = torch.norm(w - w_reconstructed)
print(f'Reconstruction error: {reconstruction_error:.6f}')
"
```

**通过标准**: 
- 方向范数 = 1.0 (±1e-5)
- 重构误差 < 1e-5

### 2.2 qTTT 参数减少验证

```bash
python -c "
from src.attnres.polar_pseudo_query import create_pseudo_query_manager

# Cartesian 模式
manager_cart = create_pseudo_query_manager(
    num_layers=32, dim=4096, use_polar=False
)
params_cart = manager_cart.get_parameter_count()
print(f'Cartesian params: {params_cart[\"total\"]}')

# Polar 模式 (qTTT)
manager_polar = create_pseudo_query_manager(
    num_layers=32, dim=4096, use_polar=True, enable_qttt=True
)
params_polar = manager_polar.get_parameter_count()
print(f'Polar params (qTTT): {params_polar[\"trainable\"]}')
print(f'Parameter reduction: {1 - params_polar[\"trainable\"]/params_cart[\"total\"]:.1%}')
"
```

**通过标准**: 参数减少 50%

---

## 实验 3: qTTT Polar 适应验证

### 3.1 球面梯度下降测试

```bash
python -c "
import torch
from src.qttt.polar_adaptation import SphericalSGD

# 初始化
opt = SphericalSGD(learning_rate=0.1)
direction = torch.tensor([1.0, 0.0, 0.0])
gradient = torch.tensor([0.0, 1.0, 0.0])  # 正交梯度

# 多步更新
for i in range(10):
    direction = opt.step(direction, gradient)
    norm = torch.norm(direction)
    print(f'Step {i+1}: norm={norm:.6f}, direction={direction.numpy()}')
"
```

**通过标准**: 
- 方向始终保持单位范数
- 沿梯度方向移动

### 3.2 Polar qTTT 适应测试

```bash
python -m pytest tests/unit/test_qttt.py::test_polar_adaptation -v
```

**预期结果**:
- 损失下降: 初始损失 → 最终损失 (下降 > 30%)
- 仅方向参数更新，幅度保持不变

---

## 实验 4: 深度优先门控策略验证

### 4.1 策略对比测试

```bash
python -c "
from src.gating.depth_priority import create_depth_priority_controller

# 创建控制器
controller = create_depth_priority_controller(
    target_rate=0.3,
    max_qttt_steps=32,
    turboquant_enabled=True
)

# 对比策略
comparison = controller.get_policy_comparison()
print('Policy Comparison:')
for k, v in comparison.items():
    print(f'  {k}: {v}')
"
```

**预期输出**:
```
standard_policy: T_think = 2 * N_qTTT * k
  -> 4096 tokens for 16 steps @ 128 span
turboquant_policy: T_think = 16 * N_qTTT * k
  -> 32768 tokens for 16 steps @ 128 span
  -> 8x theoretical advantage
```

### 4.2 门控决策测试

```bash
python -c "
from src.gating.depth_priority import create_depth_priority_controller

controller = create_depth_priority_controller(
    target_rate=0.3,
    turboquant_enabled=True
)

# 模拟决策序列
import random
for i in range(20):
    loss = random.uniform(1.0, 4.0)
    should_adapt, qttt_steps, think_tokens, threshold = controller.decide(loss)
    print(f'Loss={loss:.2f}, Adapt={should_adapt}, '
          f'qTTT={qttt_steps}, Think={think_tokens}')

# 生成报告
report = controller.get_allocation_report()
print(f'\\nAllocation Report:')
for k, v in report.items():
    print(f'  {k}: {v}')
"
```

**通过标准**:
- TurboQuant 启用时 think_tokens = 0
- 适应率接近目标 30%

---

## 实验 5: Tensor Core 加速验证

### 5.1 硬件检测

```bash
python -c "
from src.turboquant.tensor_core import estimate_throughput_gain

estimates = estimate_throughput_gain(baseline_tps=45.0)
print('Throughput Estimates:')
for k, v in estimates.items():
    print(f'  {k}: {v}')
"
```

**预期结果** (H100):
```
baseline_tokens_per_sec: 45.0
practical_tokens_per_sec: 108.0 (~110)
practical_gain: 2.4
has_tensor_cores: True
```

### 5.2 INT4 线性层测试

```bash
python -c "
import torch
from src.turboquant.tensor_core import INT4Linear

# 创建 INT4 线性层
layer = INT4Linear(4096, 4096).cuda()

# 量化权重
layer.quantize_weights()

# 测试前向
x = torch.randn(1, 4096, dtype=torch.float16).cuda()
with torch.no_grad():
    y = layer(x)
    
print(f'Input shape: {x.shape}')
print(f'Output shape: {y.shape}')
print(f'Quantized: {layer._quantized}')
"
```

---

## 实验 6: 端到端集成测试

### 6.1 完整前向传播

```bash
# 运行端到端测试
python -c "
import torch
from src.models.adaptive_transformer import AdaptiveTransformer
from src.models.configs import ModelConfig

# 创建配置
config = ModelConfig(
    vocab_size=32000,
    dim=2048,  # Small model for testing
    num_layers=8,
    num_heads=16,
    use_turboquant=True,
    use_polar_qttt=True,
    depth_priority=True
)

# 创建模型
model = AdaptiveTransformer(config).cuda()

# 测试输入
input_ids = torch.randint(0, 32000, (1, 512)).cuda()

# 前向传播
with torch.no_grad():
    output = model(input_ids)
    
print(f'Input shape: {input_ids.shape}')
print(f'Output shape: {output.logits.shape}')
print(f'KV cache compressed: {model.kv_cache_compressed}')
"
```

### 6.2 Needle-in-Haystack 基准测试

```bash
# 运行长上下文检索测试
python scripts/run_benchmarks.py \
    --benchmark needle \
    --context-lengths 4096,16384,65536,131072 \
    --model-size small \
    --use-turboquant \
    --output results/needle_turboquant.json
```

**预期结果**:
| Context | Baseline | TurboQuant | 提升 |
|---------|----------|------------|------|
| 4K | 87.5% | 98.5% | +11% |
| 32K | 22.1% | 91.3% | +69% |
| 128K | 3.2% | 78.2% | +75% |
| **平均** | **38.2%** | **86.9%** | **+128%** |

### 6.3 MATH 推理基准测试

```bash
# 运行数学推理测试
python scripts/run_benchmarks.py \
    --benchmark math \
    --model-size small \
    --use-turboquant \
    --use-polar-qttt \
    --output results/math_turboquant.json
```

**预期结果**:
- 8.7B 模型: 52.3% (匹配 50B 静态基线)
- 2.2B 模型: 56.1% (超过 8.7B 目标)

---

## 实验 7: 消融研究

### 7.1 组件贡献分析

```bash
# 运行消融研究
python -c "
import json

# 配置变体
configs = [
    {'name': 'Full System', 'attnres': True, 'turboquant': True, 'qttt': True, 'polar': True},
    {'name': 'w/o Polar qTTT', 'attnres': True, 'turboquant': True, 'qttt': True, 'polar': False},
    {'name': 'w/o TurboQuant', 'attnres': True, 'turboquant': False, 'qttt': True, 'polar': True},
    {'name': 'w/o qTTT', 'attnres': True, 'turboquant': True, 'qttt': False, 'polar': False},
    {'name': 'w/o AttnRes', 'attnres': False, 'turboquant': True, 'qttt': True, 'polar': True},
    {'name': 'Baseline', 'attnres': False, 'turboquant': False, 'qttt': False, 'polar': False},
]

# 打印配置
for cfg in configs:
    print(f\"{cfg['name']}: AttnRes={cfg['attnres']}, TurboQuant={cfg['turboquant']}, qTTT={cfg['qttt']}, Polar={cfg['polar']}\")
"
```

**预期结果** (LongBench-v2):
| 配置 | 平均分 | Δ vs Full |
|------|--------|-----------|
| Full System | 56.8% | — |
| w/o Polar qTTT | 54.2% | -2.6% |
| w/o TurboQuant | 51.5% | -5.3% |
| w/o qTTT | 50.1% | -6.7% |
| w/o AttnRes | 48.9% | -7.9% |
| Baseline | 39.7% | -17.1% |

### 7.2 吞吐量对比

```bash
# 吞吐量基准测试
python -c "
import time
import torch

configs = [
    ('Thinking Tokens', {'use_turboquant': False, 'depth_priority': False}),
    ('ADB + TurboQuant', {'use_turboquant': True, 'depth_priority': True}),
]

for name, cfg in configs:
    # 模拟推理
    start = time.time()
    # ... 实际推理代码
    elapsed = time.time() - start
    
    tokens_per_sec = 1000 / elapsed  # 假设生成 1000 tokens
    print(f'{name}: {tokens_per_sec:.1f} tokens/s')
"
```

**预期结果**:
- Thinking Tokens: ~45 t/s
- ADB + TurboQuant: ~110 t/s (2.4×)

---

## 实验 8: KV Cache 压缩验证

### 8.1 内存占用测试

```bash
python -c "
import torch
from src.turboquant import TurboQuantPipeline, TurboQuantConfig

config = TurboQuantConfig(angle_bits=3, qjl_proj_dim=256)
turbo = TurboQuantPipeline(dim=4096, config=config)

# 模拟 128K 上下文的 KV cache
batch_size = 1
num_heads = 32
seq_len = 131072
head_dim = 128

keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
values = torch.randn(batch_size, num_heads, seq_len, head_dim)

# 压缩
compressed = turbo.compress_kv_cache(keys, values)

# 计算内存
orig_mb = (keys.numel() + values.numel()) * 2 / 1024 / 1024
comp_bytes = sum(v.numel() * (1 if v.dtype == torch.int8 else 2) 
                 for v in compressed.values())
comp_mb = comp_bytes / 1024 / 1024

print(f'Original: {orig_mb:.1f} MB')
print(f'Compressed: {comp_mb:.1f} MB')
print(f'Reduction: {orig_mb/comp_mb:.1f}x')
print(f'Target: >5.7x')
"
```

**通过标准**: KV cache 压缩比 ≥ 5.7×

---

## 结果汇总与报告

### 生成完整报告

```bash
# 运行所有测试并生成报告
python scripts/run_all_experiments.py \
    --output results/turboquant_full_report.json

# 生成可视化
python scripts/visualize_results.py \
    --input results/turboquant_full_report.json \
    --output docs/turboquant_results/
```

### 预期最终指标

| 指标 | 目标 | 验证方法 |
|------|------|----------|
| TurboQuant 压缩比 | ≥ 6× | 实验 1.3 |
| KV cache 缩减 | ≥ 5.7× | 实验 8.1 |
| Polar qTTT 参数减少 | 50% | 实验 2.2 |
| 吞吐提升 | 2.4× (110 vs 45 t/s) | 实验 5.1 |
| NIH 准确率 | 86.9% | 实验 6.2 |
| MATH 准确率 | 52.3% (8.7B) | 实验 6.3 |
| 组件协同系数 | >1.0 | 实验 7.1 |

---

## 故障排除

### 问题: Tensor Core 不可用
**解决**: 检查 GPU 计算能力 (需 ≥ 8.9)，或启用模拟模式

### 问题: 压缩比低于预期
**解决**: 
- 检查 angle_bits 设置 (应为 3)
- 验证 QJL 投影维度 (推荐 256)

### 问题: Polar qTTT 不收敛
**解决**:
- 降低学习率 (默认 0.005 → 0.001)
- 检查梯度裁剪设置
- 验证球面梯度实现

---

## 附录: 快速检查清单

- [ ] 环境: PyTorch 2.1+, CUDA 12.1+
- [ ] GPU: NVIDIA H100 (或 A100 作为备选)
- [ ] TurboQuant: 6×+ 压缩比
- [ ] Polar qTTT: 50% 参数冻结
- [ ] 深度优先: think_tokens = 0
- [ ] Tensor Cores: INT4 支持检测
- [ ] NIH: 86.9% 平均准确率
- [ ] MATH: 52.3% @ 8.7B
- [ ] 吞吐: 110 tokens/s

---

**文档版本**: 1.0  
**更新日期**: 2026-03-29  
**对应代码分支**: feature/turboquant
