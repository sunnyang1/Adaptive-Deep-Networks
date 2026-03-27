# AutoDL 平台实验验证指南

**目标**: 在 AutoDL 平台上复现论文中的关键实验数据  
**预算估算**: 约 ¥500-1500 (取决于验证范围)  
**预计时间**: 3-7 天

---

## 一、环境准备

### 1.1 AutoDL 实例选择

| 验证任务 | 推荐实例 | 显存 | 价格(约) | 说明 |
|----------|---------|------|----------|------|
| Small (2.2B) 模型验证 | RTX 4090 24GB | 24GB | ¥1.5/h | 可运行 2.2B 模型推理 |
| Medium (8.7B) 模型推理 | A100 40GB | 40GB | ¥4/h | 8.7B 模型 FP16 推理 |
| Medium (8.7B) 模型微调 | 4×A100 80GB | 320GB | ¥40/h | qTTT adaptation |
| 大规模 FLOP 测量 | A100 80GB | 80GB | ¥8/h | 精确 FLOP 计算 |

**推荐方案**: 先用单卡 A100 40GB 做推理验证（Table 1, 3, 4），确认关键数据后再决定是否做训练验证。

### 1.2 基础环境配置

```bash
# 1. 创建实例后，使用以下镜像
# 推荐镜像: PyTorch 2.1.0 / CUDA 12.1 / Python 3.10

# 2. 安装依赖
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.36.0 datasets==2.16.0
pip install flash-attn==2.3.6
pip install accelerate==0.25.0
pip install evaluate sacrebleu rouge_score
pip install numpy pandas matplotlib
pip install pytest pytest-cov

# 3. 克隆项目
cd /root/autodl-tmp
git clone https://github.com/your-repo/adaptive-deep-networks.git
cd adaptive-deep-networks

# 4. 验证安装
python -m pytest tests/ -v
```

### 1.3 数据集下载

```bash
# 下载所有评估数据集
cd /root/autodl-tmp/adaptive-deep-networks

# MATH 和 GSM8K
python -c "
from datasets import load_dataset
load_dataset('hendrycks/competition_math', split='test')  # MATH test: 5000 题
load_dataset('openai/gsm8k', 'main', split='test')  # GSM8K test: 1319 题
"

# LongBench-v2
python -c "
from datasets import load_dataset
load_dataset('THUDM/LongBench-v2', split='train')  # 503 样本
"

# ZeroScrolls (需手动下载)
# 从 https://zeroscrolls-benchmark.com/ 下载
# 放置到 data/zero_scrolls/ 目录

# 通用推理 benchmark
python -c "
from datasets import load_dataset
load_dataset('Rowan/hellaswag', split='validation')
load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
load_dataset('openai/openai_humaneval', split='test')
load_dataset('lukaemon/bbh', 'boolean_expressions', split='test')
"
```

---

## 二、验证计划（按优先级排序）

### 优先级 P0：核心声明验证（必须）

#### 实验 1: MATH 数据集验证

**目标**: 验证论文 Table 3 中 AttnRes + qTTT 的 MATH 准确率  
**对比基线**: GPT-4 base 已知为 42.5%

```bash
# 运行 MATH 评估
cd /root/autodl-tmp/adaptive-deep-networks

# 使用 Medium (8.7B) 模型
python scripts/run_benchmarks.py \
  --model-size medium \
  --benchmark math \
  --output results/math_medium.json

# 使用 Small (2.2B) 模型
python scripts/run_benchmarks.py \
  --model-size small \
  --benchmark math \
  --output results/math_small.json
```

**预期结果 vs 论文声明**:
| 模型 | 论文声明 | 可接受范围 | 红线（低于此需修改论文） |
|------|---------|-----------|----------------------|
| AttnRes-M + qTTT (gated) | 52.3% | 45-55% | <40% |
| AttnRes-S + qTTT (max) | 56.1% | 45-58% | <35% |
| Transformer baseline | 35.2% | 28-40% | <20% |

**注意**: 如果 MATH 结果 < 40%，则论文中 "超越 GPT-4" 的声明不成立，需要下调数字。

#### 实验 2: GSM8K 数据集验证

```bash
python scripts/run_benchmarks.py \
  --model-size medium \
  --benchmark gsm8k \
  --output results/gsm8k_medium.json
```

**预期 vs 论文**:
| 模型 | 论文声明 | 可接受范围 | 红线 |
|------|---------|-----------|------|
| AttnRes-S + qTTT | 81.5% | 70-85% | <60% |
| AttnRes-M + qTTT | 81.4% | 70-85% | <60% |

**已知参考**: LLaMA-3 8B + CoT ≈ 68.4%

---

### 优先级 P1：长上下文验证

#### 实验 3: Needle-in-Haystack 验证

**目标**: 验证 Table 1 中各上下文长度的检索准确率

```bash
# 运行 NIH 测试（先测试较短上下文以节省时间）
python scripts/run_benchmarks.py \
  --model-size small \
  --benchmark needle_haystack \
  --context-lengths 1000 4000 16000 32000 \
  --output results/nih_small.json

# 如果短上下文结果合理，再测试长上下文
python scripts/run_benchmarks.py \
  --model-size small \
  --benchmark needle_haystack \
  --context-lengths 64000 128000 256000 \
  --output results/nih_small_long.json
```

**预期 vs 论文**:
| 上下文 | 论文 (Small) | 可接受范围 | 红线 |
|--------|-------------|-----------|------|
| 1K | 98.7% | 95-100% | <90% |
| 4K | 99.1% | 95-100% | <90% |
| 16K | 94.1% | 85-98% | <75% |
| 32K | 85.1% | 75-90% | <60% |
| 64K | 84.6% | 70-90% | <50% |
| 128K | 75.1% | 60-85% | <40% |
| Average | 89.4% | 78-92% | <65% |

**关键验证点**: 
- 如果 Average < 78%，论文中 "超越 GPT-4 (82.3%)" 的声明需要修改
- 如果 128K < 40%，长上下文优势声明需要修改

#### 实验 4: LongBench-v2 验证

```bash
python scripts/run_benchmarks.py \
  --model-size medium \
  --benchmark longbench_v2 \
  --output results/longbench_medium.json
```

**预期 vs 论文**:
| 模型 | 论文声明 | 可接受范围 | 红线 | 已知参考 |
|------|---------|-----------|------|---------|
| AttnRes-M + qTTT avg | 56.8% | 45-58% | <35% | Qwen3.5-4B: 50.0% |

---

### 优先级 P2：效率与消融验证

#### 实验 5: FLOP 等价性验证

```bash
# 运行 FLOP 分析
python src/benchmarks/flop_analysis.py \
  --model-size small \
  --output results/flop_small.json

python src/benchmarks/flop_analysis.py \
  --model-size medium \
  --output results/flop_medium.json
```

**验证目标**:
- $T_{think} / (2 \times N_{qTTT} \times k) \approx 1.0$ (论文声称 = 1.000)
- 接受范围: [0.8, 1.2]

#### 实验 6: Gating Oracle Recovery

```bash
# 验证 gating 决策与 oracle 的一致性
python scripts/run_benchmarks.py \
  --model-size medium \
  --benchmark gating_recovery \
  --output results/gating_recovery.json
```

**预期**: Oracle recovery 82-89% (论文 Table 8)

#### 实验 7: 消融实验（可选）

```bash
# 完整消融实验需要较长时间
python scripts/run_benchmarks.py \
  --model-size medium \
  --benchmark ablation \
  --output results/ablation_medium.json
```

---

## 三、结果记录与对比模板

### 3.1 实验记录表

每次实验后，填写以下表格：

```
实验ID: EXP-001
日期: 2026-03-XX
实例: A100 40GB
模型: AttnRes-S (2.2B)
Benchmark: MATH
随机种子: 42

结果:
| Level | 论文声明 | 实验结果 | 偏差 | 状态 |
|-------|---------|---------|------|------|
| 1     | 76.3%   | XX.X%   | ±X.X%| ✅/⚠️/❌ |
| 2     | 66.5%   | XX.X%   | ±X.X%| ✅/⚠️/❌ |
| 3     | 56.6%   | XX.X%   | ±X.X%| ✅/⚠️/❌ |
| 4     | 46.1%   | XX.X%   | ±X.X%| ✅/⚠️/❌ |
| 5     | 34.9%   | XX.X%   | ±X.X%| ✅/⚠️/❌ |
| Overall| 56.1%  | XX.X%   | ±X.X%| ✅/⚠️/❌ |

结论: ____________
```

### 3.2 论文数据更新规则

| 实验结果 | 论文处理方式 |
|----------|------------|
| 与论文声明偏差 <5% | 保持原数据，标注 ✅ |
| 偏差 5-15% | 更新为实验值，标注 ⚠️ |
| 偏差 >15% | 更新为实验值，重写相关分析 |
| 远低于红线 | 删除或大幅修改相关声明 |

---

## 四、常见问题与解决方案

### Q1: 显存不足 (OOM)

```bash
# 方案 1: 使用梯度检查点
accelerate launch --use_fp16 --gradient_checkpointing ...

# 方案 2: 减小 batch size
python scripts/run_benchmarks.py --batch-size 1 ...

# 方案 3: 使用 8-bit 量化
python scripts/run_benchmarks.py --load-in-8bit ...
```

### Q2: NIH 长上下文测试太慢

```bash
# 先测试短上下文 (1K-16K)，确认方向正确后再测长上下文
python scripts/run_benchmarks.py \
  --benchmark needle_haystack \
  --context-lengths 1000 4000 16000 \
  --num-depths 5  # 减少测试深度点 (默认10)
```

### Q3: 数据集下载失败

```bash
# AutoDL 可能无法直接访问 HuggingFace
# 方案 1: 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方案 2: 使用 ModelScope 下载
pip install modelscope
# MATH: modelscope 下载数据后转换格式
```

### Q4: 模型权重加载失败

```bash
# 如果 HuggingFace 模型未发布，需要先本地训练或使用占位权重
# 论文声明有 HuggingFace checkpoint (2.2B, 8.7B, 27B)
# 验证时可能需要使用随机初始化权重或 LLaMA 权重作为基线

# 使用 LLaMA 权重作为基线:
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
```

---

## 五、预算与时间规划

### 方案 A: 最小验证（仅 P0）

| 任务 | 时间 | 实例 | 费用(约) |
|------|------|------|----------|
| MATH (Small + Medium) | 6h | A100 40GB | ¥24 |
| GSM8K (Small + Medium) | 2h | A100 40GB | ¥8 |
| 环境配置 | 1h | — | ¥4 |
| **合计** | **9h** | | **¥36** |

### 方案 B: 标准验证（P0 + P1）

| 任务 | 时间 | 实例 | 费用(约) |
|------|------|------|----------|
| P0 验证 | 9h | A100 40GB | ¥36 |
| NIH (全长度) | 12h | A100 40GB | ¥48 |
| LongBench-v2 | 4h | A100 40GB | ¥16 |
| FLOP 分析 | 2h | A100 40GB | ¥8 |
| **合计** | **27h** | | **¥108** |

### 方案 C: 完整验证（P0 + P1 + P2）

| 任务 | 时间 | 实例 | 费用(约) |
|------|------|------|----------|
| P0 + P1 验证 | 27h | A100 40GB | ¥108 |
| 消融实验 | 24h | A100 40GB | ¥96 |
| Gating 分析 | 8h | A100 40GB | ¥32 |
| 推理延迟测量 | 4h | A100 40GB | ¥16 |
| **合计** | **63h** | | **¥252** |

---

## 六、验证完成后

### 6.1 更新论文

根据验证结果：
1. 用真实数据替换论文中估算数据
2. 在数据后添加 `*` 标注（如仍有未验证数据）
3. 更新 Table X 中的对比表格
4. 调整摘要和结论中的声明

### 6.2 生成验证报告

```bash
# 汇总所有实验结果
python scripts/summarize_results.py \
  --results-dir results/ \
  --output VERIFICATION_RESULTS.md
```

### 6.3 代码仓库更新

```bash
# 提交验证代码和结果
git add results/
git commit -m "Add independent verification results"
git push
```

---

## 七、附录：AutoDL 快速启动脚本

```bash
#!/bin/bash
# autodl_quick_start.sh
# AutoDL 平台一键环境配置

set -e

echo "=== Adaptive Deep Networks 验证环境配置 ==="

# 1. 设置镜像
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 安装依赖
echo "[1/4] 安装 Python 依赖..."
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers==4.36.0 datasets==2.16.0 accelerate==0.25.0 -q
pip install flash-attn==2.3.6 --no-build-isolation -q
pip install evaluate sacrebleu rouge_score -q
pip install pytest pytest-cov numpy pandas matplotlib -q

# 3. 下载评估数据集
echo "[2/4] 下载评估数据集..."
python -c "
from datasets import load_dataset
print('  下载 MATH...')
load_dataset('hendrycks/competition_math', split='test')
print('  下载 GSM8K...')
load_dataset('openai/gsm8k', 'main', split='test')
print('  下载 LongBench-v2...')
load_dataset('THUDM/LongBench-v2', split='train')
print('  下载 HellaSwag...')
load_dataset('Rowan/hellaswag', split='validation')
print('  下载 ARC-Challenge...')
load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
print('  数据集下载完成!')
"

# 4. 运行单元测试
echo "[3/4] 运行单元测试..."
cd /root/autodl-tmp/adaptive-deep-networks
python -m pytest tests/ -v --tb=short

# 5. 开始 P0 验证
echo "[4/4] 开始 P0 核心验证..."
echo "  运行 MATH 验证 (Small 模型)..."
python scripts/run_benchmarks.py \
  --model-size small \
  --benchmark math \
  --output results/math_small.json

echo "=== 环境配置完成 ==="
echo "后续步骤:"
echo "  1. 查看 results/ 目录中的结果"
echo "  2. 运行 ./autodl_run_all.sh 进行完整验证"
```

```bash
#!/bin/bash
# autodl_run_all.sh
# 完整验证脚本（标准方案 B）

set -e

RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

echo "=== 开始完整验证 (方案 B) ==="
echo "结果目录: $RESULTS_DIR"

# P0: MATH
echo "[P0-1] MATH Small..."
python scripts/run_benchmarks.py --model-size small --benchmark math --output $RESULTS_DIR/math_small.json
echo "[P0-2] MATH Medium..."
python scripts/run_benchmarks.py --model-size medium --benchmark math --output $RESULTS_DIR/math_medium.json

# P0: GSM8K
echo "[P0-3] GSM8K Small..."
python scripts/run_benchmarks.py --model-size small --benchmark gsm8k --output $RESULTS_DIR/gsm8k_small.json
echo "[P0-4] GSM8K Medium..."
python scripts/run_benchmarks.py --model-size medium --benchmark gsm8k --output $RESULTS_DIR/gsm8k_medium.json

# P1: NIH (短上下文先跑)
echo "[P1-1] NIH Short (1K-16K)..."
python scripts/run_benchmarks.py --model-size small --benchmark needle_haystack --context-lengths 1000 4000 16000 --output $RESULTS_DIR/nih_short.json

echo "[P1-2] NIH Long (32K-128K)..."
python scripts/run_benchmarks.py --model-size small --benchmark needle_haystack --context-lengths 32000 64000 128000 --output $RESULTS_DIR/nih_long.json

# P1: LongBench-v2
echo "[P1-3] LongBench-v2..."
python scripts/run_benchmarks.py --model-size medium --benchmark longbench_v2 --output $RESULTS_DIR/longbench_medium.json

# P2: FLOP
echo "[P2-1] FLOP Analysis..."
python src/benchmarks/flop_analysis.py --model-size small --output $RESULTS_DIR/flop_small.json
python src/benchmarks/flop_analysis.py --model-size medium --output $RESULTS_DIR/flop_medium.json

echo "=== 所有验证完成 ==="
echo "结果保存在: $RESULTS_DIR/"
```

---

*最后更新: 2026-03-27*
