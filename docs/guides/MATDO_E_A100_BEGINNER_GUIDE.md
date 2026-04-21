# MATDO-E 新手完全指南 (A100 80G)

本指南适合从未接触过大型语言模型训练的新手，手把手教你从零开始在A100 80G机器上完成MATDO-E论文的完整流程。

---

## 第一阶段：环境准备 (30分钟)

### 1.1 确认硬件环境

在你的A100机器上运行以下命令，确认环境正常：

```bash
# 1. 查看GPU信息
nvidia-smi

# 预期输出示例：
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |   0  NVIDIA A100 80GB    On   | 00000000:00:04.0 Off |                    0 |
# | N/A   35C    P0    45W / 300W |      0MiB / 81920MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

```bash
# 2. 检查CUDA版本
nvcc --version

# 预期输出（需要CUDA 11.8或更高）：
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2022 NVIDIA Corporation
# Built on Wed_Sep_21_10:33:58_PDT_2022
# Cuda compilation tools, release 11.8, V11.8.89
```

```bash
# 3. 检查磁盘空间（需要至少100GB）
df -h /

# 预期输出：
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/nvme0n1    500G   50G  450G  10% /
```

### 1.2 安装基础依赖

```bash
# 更新系统包
sudo apt-get update
sudo apt-get install -y git wget vim htop tmux

# 安装conda（如果没有）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda init bash
source ~/.bashrc
```

### 1.3 创建Python环境

```bash
# 创建MATDO-E专用环境
conda create -n matdo python=3.10 -y
conda activate matdo

# 验证Python版本
python --version
# 输出：Python 3.10.x
```

---

## 第二阶段：代码准备 (15分钟)

### 2.1 克隆代码仓库

```bash
# 进入你的工作目录（根据你的机器调整）
cd ~
mkdir -p workspace && cd workspace

# 克隆项目代码
git clone https://github.com/your-org/Adaptive-Deep-Networks.git
cd Adaptive-Deep-Networks

# 查看项目结构
ls -la
```

### 2.2 安装Python依赖

```bash
# 确保在matdo环境中
conda activate matdo

# 安装基础依赖
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt

# 安装可选依赖（用于实验和可视化）
pip install matplotlib seaborn pandas scipy scikit-learn
pip install faiss-gpu  # GPU加速的向量检索
pip install transformers datasets accelerate
```

### 2.3 验证安装

```bash
# 测试PyTorch GPU可用
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'GPU可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 预期输出：
# PyTorch版本: 2.1.0+cu118
# GPU可用: True
# GPU数量: 1
```

---

## 第三阶段：数据准备 (20分钟)

### 3.1 自动下载数据

```bash
# 项目提供了数据下载脚本
cd ~/workspace/Adaptive-Deep-Networks
python scripts/data/download_all.py

# 这会下载：
# - LongBench数据集（用于长上下文测试）
# - Needle-in-Haystack测试数据
# - Wikipedia 2023子集（用于Engram预训练）
```

### 3.2 手动下载（如果自动失败）

```bash
# 创建数据目录
mkdir -p data/raw data/processed data/engram

# 下载Wikipedia数据用于Engram（示例）
wget https://dumps.wikimedia.org/enwiki/2023-04-01/enwiki-20230401-pages-articles-multistream.xml.bz2 \
    -O data/raw/wikipedia_2023.xml.bz2

# 下载LongBench
pip install datasets
python -c "
from datasets import load_dataset
dataset = load_dataset('THUDM/LongBench', 'narrativeqa')
dataset.save_to_disk('data/raw/longbench')
"
```

### 3.3 构建Engram索引（重要）

```bash
# 使用MiniLM构建Engram向量索引
python scripts/engram/build_index.py \
    --data_path data/raw/wikipedia_2023 \
    --output_path data/engram/wikipedia_hnsw.index \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --num_clusters 128000 \
    --device cuda

# 预期输出：
# Loading embedding model...
# Processing documents: 100%|████████| 50000/50000 [05:30<00:00, 151.23it/s]
# Building HNSW index...
# Index saved to data/engram/wikipedia_hnsw.index
# Total entries: 128000
```

---

## 第四阶段：模型训练 (2-4小时)

### 4.1 训练小型模型（推荐新手先跑小模型）

```bash
# 创建训练配置
cat > configs/train_small.yaml << 'EOF'
# MATDO-E Small Model Training Config
model:
  name: "matdo_e_small"
  size: "small"  # 1.1B参数
  d_model: 1408
  n_layers: 32
  n_heads: 8
  n_blocks: 8

training:
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-4
  num_epochs: 3
  warmup_steps: 1000
  max_seq_len: 8192
  
matdo_e:
  enable_arbitrage: true
  zeta: 0.35
  eta: 0.5
  E_max: 128000
  rho_target: 0.99
  
hardware:
  device: cuda
  mixed_precision: fp16
  compile: false  # 新手建议关闭torch.compile

checkpoint:
  save_dir: "checkpoints/small"
  save_every: 1000
EOF

# 开始训练
python scripts/train.py --config configs/train_small.yaml
```

### 4.2 监控训练过程

```bash
# 方法1：使用tmux保持训练会话
tmux new -s matdo_train
conda activate matdo
cd ~/workspace/Adaptive-Deep-Networks
python scripts/train.py --config configs/train_small.yaml

# 按 Ctrl+B 然后按 D  detach会话
# 之后可以通过以下命令恢复：
tmux attach -t matdo_train

# 方法2：使用nvidia-smi监控显存
watch -n 1 nvidia-smi

# 方法3：查看训练日志
tail -f checkpoints/small/training.log
```

### 4.3 训练预期输出

```
[2024-01-15 10:30:45] Epoch 1/3 - Step 100/5000 - Loss: 2.345 - LR: 3.0e-4 - GPU: 42.3GB/80GB
[2024-01-15 10:35:12] Epoch 1/3 - Step 200/5000 - Loss: 1.987 - LR: 2.9e-4 - GPU: 42.3GB/80GB
...
[2024-01-15 12:45:30] Training complete! Model saved to checkpoints/small/final.pt
```

### 4.4 训练中型模型（可选，需要更多时间）

```bash
# 如果小型模型训练成功，可以尝试中型模型（5.7B参数）
cat > configs/train_medium.yaml << 'EOF'
model:
  name: "matdo_e_medium"
  size: "medium"
  d_model: 2496
  n_layers: 56
  n_heads: 16
  n_blocks: 8

training:
  batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 2.5e-4
  num_epochs: 2
  warmup_steps: 500
  max_seq_len: 16384
  
matdo_e:
  enable_arbitrage: true
  zeta: 0.35
  eta: 0.5
  E_max: 128000
  
hardware:
  device: cuda
  mixed_precision: bf16  # A100支持bf16
  
checkpoint:
  save_dir: "checkpoints/medium"
EOF

python scripts/train.py --config configs/train_medium.yaml
```

---

## 第五阶段：推理测试 (30分钟)

### 5.1 基础推理测试

```bash
# 使用训练好的模型进行推理
python scripts/inference.py \
    --checkpoint checkpoints/small/final.pt \
    --prompt "The capital of France is" \
    --max_new_tokens 20 \
    --device cuda

# 预期输出：
# Prompt: The capital of France is
# Generated: Paris, which is known for its iconic Eiffel Tower and rich cultural heritage.
```

### 5.2 Needle-in-Haystack测试（长上下文）

```bash
# 测试模型在长文档中检索信息的能力
python experiments/real_model/needle_haystack_real.py \
    --checkpoint checkpoints/small/final.pt \
    --context_lengths 4096 8192 16384 32768 \
    --num_samples 10 \
    --output_dir results/needle_test

# 查看结果
cat results/needle_test/summary.json
```

### 5.3 使用MATDO-E套利模式推理

```bash
# 测试高显存压力下的推理
python scripts/inference_matdo_e.py \
    --checkpoint checkpoints/small/final.pt \
    --prompt "Explain quantum computing in simple terms" \
    --max_new_tokens 200 \
    --rho 0.99 \
    --enable_arbitrage \
    --E 128000 \
    --device cuda

# 观察显存使用
# 预期：即使rho=0.99，模型仍能正常生成而不OOM
```

---

## 第六阶段：验证实验 (1-2小时)

### 6.1 运行所有MATDO实验

```bash
# 运行完整的实验套件（使用真实模型）
cd experiments/matdo
python run_all_experiments.py \
    --use-real-model \
    --checkpoint ../../checkpoints/small/final.pt \
    --size small \
    --device cuda
```

### 6.2 单独运行关键实验

```bash
# US1: 二阶奇点标度律验证
python singularity/measure_t_opt.py

# US4: SOTA对比实验
python sota_comparison/compare_baselines.py

# US5: 消融实验
python ablation/run_ablation.py
```

### 6.3 vLLM集成实验

```bash
# 运行vLLM集成实验套件
cd vllm_integration
python run_all_vllm_experiments.py

# 查看结果
ls -la results/
cat results/vllm_integration_summary.json
```

---

## 第七阶段：结果可视化 (15分钟)

### 7.1 生成论文图表

```bash
# 生成所有图表
python scripts/visualization/generate_all_figures.py \
    --results_dir results/ \
    --output_dir figures/

# 查看生成的图表
ls figures/
# 预期输出：
# fig_a_throughput.pdf
# fig_b_latency.pdf
# fig_c_accuracy_recovery.pdf
# fig_singularity_scaling.pdf
# ...
```

### 7.2 关键指标检查

```bash
# 运行验证脚本检查是否达到论文指标
python scripts/validate_paper_metrics.py \
    --results_dir results/ \
    --checkpoint checkpoints/small/final.pt

# 预期输出：
# === Paper Metrics Validation ===
# [✓] Throughput at ρ=0.99: 1420 tok/s (target: >1000)
# [✓] Accuracy with E: 97.8% (target: >95%)
# [✓] Wall postponement: ρ=0.93 → ρ=0.99
# [✓] Latency masking efficiency: 85% (target: >50%)
# ================================
# All metrics PASSED! ✅
```

---

## 第八阶段：故障排除

### 问题1：CUDA OOM（显存不足）

```bash
# 解决方案1：减小batch size
# 编辑configs/train_small.yaml
# batch_size: 4 → batch_size: 2

# 解决方案2：减小序列长度
# max_seq_len: 8192 → max_seq_len: 4096

# 解决方案3：启用gradient checkpointing
# 在config中添加：
# training:
#   gradient_checkpointing: true
```

### 问题2：训练速度太慢

```bash
# 解决方案1：使用torch.compile（PyTorch 2.0+）
# 在config中设置：
# hardware:
#   compile: true

# 解决方案2：增加数据加载workers
# training:
#   num_workers: 8

# 解决方案3：使用flash attention
pip install flash-attn --no-build-isolation
```

### 问题3：Engram索引构建失败

```bash
# 如果Faiss安装失败，使用CPU版本
pip uninstall faiss-gpu
pip install faiss-cpu

# 或者跳过Engram构建（仅用于测试3D MATDO）
# 在训练配置中设置：
# matdo_e:
#   enable_arbitrage: false
```

### 问题4：依赖冲突

```bash
# 清理环境重新开始
conda activate matdo
pip freeze | xargs pip uninstall -y
pip install -r requirements.txt
```

---

## 快速启动脚本

为了方便，创建一个一键启动脚本：

```bash
cat > ~/start_matdo_training.sh << 'EOF'
#!/bin/bash
set -e

echo "=== MATDO-E Training Starter ==="

# 激活环境
source ~/miniconda3/bin/activate matdo

# 进入项目目录
cd ~/workspace/Adaptive-Deep-Networks

# 检查GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 开始训练
echo "Starting training..."
python scripts/train.py --config configs/train_small.yaml

echo "Training complete!"
EOF

chmod +x ~/start_matdo_training.sh

# 使用：
# ~/start_matdo_training.sh
```

---

## 下一步建议

1. **理解代码**：阅读`experiments/matdo/MATDO_E_IMPLEMENTATION.md`了解实现细节
2. **修改实验**：尝试调整`zeta`和`eta`参数观察对结果的影响
3. **扩展模型**：在小型模型成功后，尝试中型或大型模型
4. **自定义数据**：使用你自己的数据集训练

---

## 获得帮助

如果遇到问题：

1. 检查日志文件：`checkpoints/*/training.log`
2. 查看GPU状态：`nvidia-smi`
3. 测试基础功能：`python -c "import torch; print(torch.cuda.is_available())"`
4. 参考AGENTS.md了解项目架构

祝你实验顺利！🚀