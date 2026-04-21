# Adaptive Deep Networks — 重构规范 SPEC.md

## 1. 目标

将当前混乱的代码库重构为模块化、清晰、可维护的结构，同时保持所有论文功能的完整性。

## 2. 新架构

### 2.1 顶层结构

```
Adaptive-Deep-Networks/
├── adn/                          # 统一Python包（原 src/ 升级）
│   ├── __init__.py               # 包根，导出主要类和函数
│   ├── core/                     # 基础组件（配置、基类、工具）
│   ├── models/                   # 自适应Transformer模型
│   ├── attention/                # 注意力机制（AttnRes等）
│   ├── qttt/                     # qTTT查询时自适应训练
│   ├── quantization/             # KV缓存量化（RaBitQ等）
│   ├── memory/                   # 外部记忆（Engram）
│   ├── gating/                   # 门控与自适应计算
│   ├── qasp/                     # QASP论文模块
│   ├── matdo_e/                  # MATDO-E论文模块
│   ├── experiments/              # 统一实验框架
│   └── utils/                    # 通用工具
├── scripts/                      # 精简后的脚本
├── tests/                        # 测试
├── configs/                      # 配置文件
├── archive/                      # 过时代码归档
├── docs/                         # 文档
├── data/                         # 数据
├── results/                      # 实验结果
├── pyproject.toml                # 项目配置
├── README.md                     # 主文档
└── QASP_paper_cn.md              # 论文（保留在根目录）
└── matdo-e_paper_cn.md           # 论文（保留在根目录）
```

### 2.2 adn/ 包内部结构

#### adn/core/ — 基础组件
- `__init__.py` — 导出 ModelConfig, ADNConfig, BaseModule
- `config.py` — 统一配置系统（合并各模块配置）
- `base.py` — 基础模块：RMSNorm, SwiGLU, 基类
- `types.py` — 共享类型定义

#### adn/models/ — 自适应Transformer
- `__init__.py` — 导出 AdaptiveTransformer, AdaptiveLayer
- `adaptive_transformer.py` — 主模型（从 src/models/ 迁移并清理）
- `configs.py` — 模型配置类
- `tokenizer.py` — 分词器封装
- `generator.py` — 增量生成器（合并 incremental_generator）

#### adn/attention/ — 注意力机制
- `__init__.py` — 导出 BlockAttnRes, TwoPhaseBlockAttnRes
- `block_attnres.py` — AttnRes实现（从 src/attnres/ 迁移）
- `pseudo_query.py` — 伪查询实现
- `polar_pseudo_query.py` — 极坐标伪查询

#### adn/qttt/ — 查询时自适应
- `__init__.py` — 导出 qTTT, PolarQTTT, KVCache
- `adaptation.py` — 核心qTTT适配（从 src/qttt/ 迁移）
- `polar_adaptation.py` — 极坐标qTTT
- `batch_adaptation.py` — 批量适配
- `margin_loss.py` — 边际损失
- `config.py` — qTTT配置

#### adn/quantization/ — KV缓存量化
- `__init__.py` — 导出 RaBitQ, TurboQuant
- `rabitq_api.py` — RaBitQ API（从 src/rabitq/api.py 迁移）
- `rabitq_rotation.py` — 旋转变换
- `rabitq_quantizer.py` — 量化器
- `rabitq_packing.py` — 位打包
- `compressor.py` — 通用压缩接口

#### adn/memory/ — 外部记忆
- `__init__.py` — 导出 Engram, NgramHash
- `engram.py` — Engram模块（从 src/engram/ 迁移）
- `ngram_hash.py` — n-gram哈希
- `embeddings.py` — 嵌入管理

#### adn/gating/ — 门控机制
- `__init__.py` — 导出 PonderGate, DepthPriorityGate
- `ponder_gate.py` — 思考门控
- `depth_priority.py` — 深度优先门控
- `threshold.py` — 阈值机制
- `reconstruction.py` — 重建门控

#### adn/qasp/ — QASP模块
- `__init__.py` — 导出 QASPLayer, QASPTransformer, QualityScore
- `stiefel.py` — Stiefel流形优化（从 QASP/adaptation/stiefel.py 迁移）
- `matrix_qasp.py` — 矩阵级QASP
- `quality_score.py` — 信息质量评分
- `value_weighted_attnres.py` — 质量加权AttnRes
- `value_weighted_engram.py` — 质量加权Engram
- `models.py` — QASP模型层和Transformer

#### adn/matdo_e/ — MATDO-E模块
- `__init__.py` — 导出 MATDOPolicy, MATDOConfig
- `config.py` — MATDO配置
- `policy.py` — 策略决策
- `error_model.py` — 误差模型
- `resource_theory.py` — 资源理论
- `online_estimation.py` — 在线估计

#### adn/experiments/ — 实验框架
- `__init__.py` — 导出 ExperimentRunner
- `runner.py` — 统一实验运行器
- `benchmarks/` — 基准测试
  - `math_eval.py` — 数学评估
  - `needle.py` — Needle-in-Haystack
  - `flop_analysis.py` — FLOP分析

#### adn/utils/ — 通用工具
- `__init__.py` — 导出工具函数
- `paths.py` — 路径管理
- `device.py` — 设备管理
- `logging_config.py` — 日志配置
- `visualization.py` — 可视化

### 2.3 scripts/ 精简结构

```
scripts/
├── train.py              # 统一训练入口（替代 train_model.py + wrappers）
├── evaluate.py           # 统一评估入口
├── benchmark.py          # 统一基准测试入口
├── generate.py           # 生成入口（从 QASP/scripts/ 合并）
└── experiment.py         # 实验入口（从 experiments/ 合并）
```

### 2.4 tests/ 结构

```
tests/
├── conftest.py
├── unit/
│   ├── test_core.py
│   ├── test_attention.py
│   ├── test_qttt.py
│   ├── test_quantization.py
│   ├── test_memory.py
│   ├── test_qasp.py
│   └── test_matdo_e.py
├── integration/
│   └── test_end_to_end.py
└── e2e/
    └── test_paper_reproduction.py
```

### 2.5 archive/ 归档结构

```
archive/
├── src_legacy/           # 旧 src/ 完整备份
├── QASP_legacy/          # 旧 QASP/ 完整备份
├── MATDO_legacy/         # 旧 experiments/matdo/ 备份
├── scripts_legacy/       # 旧 scripts/ 备份
└── README_ARCHIVE.md     # 归档说明
```

## 3. 模块接口规范

### 3.1 核心配置接口

```python
# adn/core/config.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class ModelConfig:
    """统一模型配置"""
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    num_blocks: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    attention_dropout: float = 0.0
    vocab_size: int = 32000
    max_seq_len: int = 2048
    use_qttt: bool = False
    use_attnres: bool = True
    use_engram: bool = False
    qttt_config: Optional[Dict] = None
    engram_config: Optional[Dict] = None
    rabitq_config: Optional[Dict] = None

@dataclass  
class ADNConfig:
    """ADN全局配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    qasp: Optional[Dict] = None
    matdo_e: Optional[Dict] = None
```

### 3.2 模型接口

```python
# adn/models/adaptive_transformer.py
class AdaptiveTransformer(nn.Module):
    """统一自适应Transformer"""
    def __init__(self, config: ModelConfig): ...
    def forward(self, input_ids, attention_mask=None, 
                use_qttt=False, use_engram=False): ...
    def generate(self, input_ids, max_new_tokens, **kwargs): ...
```

### 3.3 QASP接口

```python
# adn/qasp/
class QASPLayer(nn.Module):
    """QASP增强层"""
    def __init__(self, config, base_layer): ...
    def forward(self, hidden_states, compute_quality=True): ...

class QualityScore(nn.Module):
    """信息质量评分"""
    def forward(self, hidden_states): -> torch.Tensor: ...

def stiefel_projection(W: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """Stiefel流形投影"""
```

### 3.4 MATDO-E接口

```python
# adn/matdo_e/
class MATDOPolicy:
    """MATDO资源策略"""
    def solve(self, observation: RuntimeObservation) -> PolicyDecision: ...
    
def estimate_error(r_bits, m_blocks, t_steps, engram_entries, config) -> float: ...
def required_adaptation_steps(...) -> int: ...
```

## 4. 迁移映射

### 4.1 src/ → adn/

| 源文件 | 目标文件 | 操作 |
|--------|---------|------|
| `src/models/adaptive_transformer.py` | `adn/models/adaptive_transformer.py` | 迁移 + 清理 |
| `src/models/configs.py` | `adn/core/config.py` | 合并配置 |
| `src/attnres/*.py` | `adn/attention/*.py` | 迁移 |
| `src/qttt/*.py` | `adn/qttt/*.py` | 迁移 |
| `src/rabitq/*.py` | `adn/quantization/rabitq_*.py` | 迁移 + 重命名 |
| `src/engram/*.py` | `adn/memory/*.py` | 迁移 |
| `src/gating/*.py` | `adn/gating/*.py` | 迁移 |
| `src/models/incremental_*.py` | `adn/models/generator.py` | 合并 |

### 4.2 QASP/ → adn/qasp/

| 源文件 | 目标文件 | 操作 |
|--------|---------|------|
| `QASP/adaptation/stiefel.py` | `adn/qasp/stiefel.py` | 迁移 |
| `QASP/adaptation/matrix_qasp.py` | `adn/qasp/matrix_qasp.py` | 迁移 |
| `QASP/adaptation/quality_score.py` | `adn/qasp/quality_score.py` | 迁移 |
| `QASP/models/*.py` | `adn/qasp/models.py` | 合并 |
| `QASP/scripts/*.py` | `scripts/generate.py, experiment.py` | 合并 |

### 4.3 MATDO-new/ → adn/matdo_e/

| 源文件 | 目标文件 | 操作 |
|--------|---------|------|
| `MATDO-new/matdo_new/core/*.py` | `adn/matdo_e/*.py` | 迁移 |
| `MATDO-new/matdo_new/modeling/*.py` | `adn/matdo_e/` | 合并 |
| `MATDO-new/matdo_new/runtime/*.py` | `adn/matdo_e/` | 合并 |

### 4.4 experiments/ → adn/experiments/

| 源文件 | 目标文件 | 操作 |
|--------|---------|------|
| `experiments/common/*.py` | `adn/utils/*.py` | 合并 |
| `experiments/core/*/` | `adn/experiments/` | 保留引用 |
| `experiments/benchmarks/*.py` | `adn/experiments/benchmarks/` | 迁移 |

### 4.5 scripts/ → scripts/

| 源文件 | 目标文件 | 操作 |
|--------|---------|------|
| `scripts/training/train_model.py` | `scripts/train.py` | 统一入口 |
| `scripts/evaluation/run_benchmarks.py` | `scripts/benchmark.py` | 统一入口 |
| `scripts/legacy/*` | `archive/scripts_legacy/` | 归档 |
| `scripts/training/train_*.py` (wrappers) | `archive/scripts_legacy/` | 归档 |

## 5. 需要移除的内容

### 5.1 过时模块（移入 archive/）
- `src/turboquant/` — 被RaBitQ替代
- `src/rabitq/legacy/` — 旧版RaBitQ实现
- `scripts/legacy/` — 旧脚本
- `scripts/experiments/legacy/` — 旧实验
- `experiments/validation/legacy/` — 旧验证
- `experiments/autoresearch/` — 自动研究（非核心）
- `.workbuddy/` — 编辑器配置

### 5.2 重复实现（保留最优版本）
- `QASP/` 包 — 整合进 `adn/qasp/`
- `MATDO-new/` 包 — 整合进 `adn/matdo_e/`
- `experiments/matdo/` — 由 `adn/matdo_e/` 替代

## 6. 入口点命令映射

重构后的入口命令保持兼容：

```bash
# 训练（保持兼容）
python -m adn.scripts.train --model-size small --output-dir results/small

# 评估
python -m adn.scripts.evaluate --model-size medium --benchmarks all

# 生成（QASP）
python -m adn.scripts.generate --dry-run

# 实验
python -m adn.scripts.experiment --category core --quick

# 基准测试
python -m adn.scripts.benchmark --model-size medium
```

## 7. pyproject.toml 更新

更新包名和入口点：

```toml
[project]
name = "adn"
version = "0.2.0"

[project.scripts]
adn-train = "adn.scripts.train:main"
adn-eval = "adn.scripts.evaluate:main"
adn-generate = "adn.scripts.generate:main"
adn-benchmark = "adn.scripts.benchmark:main"
adn-experiment = "adn.scripts.experiment:main"
```

## 8. 验证清单

重构完成后需验证：

- [ ] `python -c "import adn; print(adn.__version__)"` 成功
- [ ] 所有核心模块可导入
- [ ] AttnRes前向传播正常
- [ ] qTTT适配正常
- [ ] RaBitQ压缩/解压正常
- [ ] Engram记忆读写正常
- [ ] QASP Stiefel投影正常
- [ ] MATDO-E策略求解正常
- [ ] 训练脚本可运行（至少到模型创建）
- [ ] 实验脚本可列出实验
