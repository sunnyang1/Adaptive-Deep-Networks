# A100 80G 新手完全指南

> **目标读者**：完全的新手，第一次使用 A100 训练大模型  
> **硬件环境**：A100 80GB GPU  
> **预计时间**：首次设置约 30 分钟，训练约 6-12 小时

---

## 📋 准备工作

## ⚡ A100 单卡最稳一键流程（建议先跑）

如果你希望先快速跑通，再看后面的详细说明，可以直接按下面执行：

```bash
# 0) 进入项目
cd ~/adaptive-deep-networks

# 1) 激活虚拟环境
source venv/bin/activate

# 2) 安装项目依赖（首次）
pip install -e ".[dev]"

# 3) 验证 GPU 与 PyTorch
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"

# 4) 开一个 tmux 会话避免 SSH 断线中断训练
tmux new-session -s adn-train

# 5) 训练 medium（单卡 A100 推荐）
python3 scripts/training/train_model.py \
    --model-size medium \
    --output-dir results/medium_model \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-4 \
    --seq-len 512 \
    --train-samples 50000 \
    --val-samples 5000

# 6) 训练后检查结果文件
python3 -m json.tool results/medium_model/training_results.json | head -80
```

如果你更关注论文对齐而不是快速跑通，直接用：

```bash
make train-paper-medium OUTPUT_DIR=results/medium_paper
```

### 1.1 你需要什么

| 项目 | 要求 | 说明 |
|------|------|------|
| 机器 | A100 80GB x 1 或多卡 | 本指南以单卡为例 |
| 系统 | Ubuntu 20.04/22.04 | 或其他 Linux 发行版 |
| 网络 | 能访问 GitHub 和 PyPI | 用于下载代码和依赖 |
| 存储 | 至少 100GB 可用空间 | 模型和数据需要空间 |
| 时间 | 首次设置 ~30 分钟 | 后续训练 6-12 小时 |

### 1.2 获取服务器访问权限

通常你会收到以下信息：
- IP 地址（如 `192.168.1.100`）
- 用户名（如 `ubuntu` 或 `root`）
- 密码或 SSH 密钥

**连接到服务器：**
```bash
# 使用密码连接
ssh ubuntu@192.168.1.100

# 或使用密钥连接
ssh -i ~/.ssh/your_key.pem ubuntu@192.168.1.100
```

连接成功后，你会看到类似这样的提示：
```
ubuntu@a100-server:~$
```

---

## 🚀 第一步：环境设置（逐行复制粘贴）

### 2.1 更新系统

复制以下命令，逐行粘贴执行：

```bash
# 更新软件包列表
sudo apt-get update

# 安装基础工具（可能需要输入密码）
sudo apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    python3.10 \
    python3.10-venv \
    python3.10-dev
```

**解释：**
- `build-essential`：编译工具
- `git`：版本控制
- `htop`：系统监控（类似于 Windows 任务管理器）
- `tmux`：终端复用器（训练时保持会话不中断）
- `python3.10`：Python 环境

### 2.2 创建项目目录

```bash
# 创建项目目录（~ 表示用户主目录）
mkdir -p ~/adaptive-deep-networks
cd ~/adaptive-deep-networks

# 查看当前路径（确认你在正确的位置）
pwd
# 应该输出: /home/ubuntu/adaptive-deep-networks
```

### 2.3 创建 Python 虚拟环境

```bash
# 创建虚拟环境（名为 venv）
python3.10 -m venv venv

# 激活虚拟环境
# 注意：每次新开终端都需要执行这个命令！
source venv/bin/activate

# 确认激活成功（看到前面的 (venv) 表示成功）
# (venv) ubuntu@a100-server:~/adaptive-deep-networks$
```

**⚠️ 重要提示：**
- 虚拟环境激活后，命令行前面会显示 `(venv)`
- 每次新开终端都要执行 `source venv/bin/activate`
- 如果忘记激活，会安装到系统环境，可能导致冲突

### 2.4 安装 PyTorch（GPU 版本）

```bash
# 先升级 pip
pip install --upgrade pip

# 安装 PyTorch 2.1.0 + CUDA 12.1（专门为 A100 优化）
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

**验证 PyTorch 安装：**
```bash
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'CUDA 版本: {torch.version.cuda}')
print(f'GPU 数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

**预期输出：**
```
PyTorch 版本: 2.1.0+cu121
CUDA 可用: True
CUDA 版本: 12.1
GPU 数量: 1
  GPU 0: NVIDIA A100-SXM4-80GB
```

如果看到 `CUDA 可用: False`，说明驱动有问题，需要联系服务器提供商。

### 2.5 安装其他依赖

```bash
# 安装 Transformers、数据集等 ML 库
pip install \
    transformers==4.35.0 \
    datasets==2.14.0 \
    accelerate==0.24.0 \
    numpy==1.24.0 \
    scipy==1.11.0 \
    matplotlib==3.8.0 \
    seaborn==0.13.0 \
    pandas==2.0.0 \
    tqdm==4.66.0 \
    wandb==0.15.0 \
    pytest==7.4.0

# 安装 Flash Attention（A100 专用加速库）
# 这个安装可能需要 5-10 分钟
pip install flash-attn==2.3.0 --no-build-isolation
```

### 2.6 克隆代码仓库

```bash
# 确保你在项目目录
pwd
# 应该输出: /home/ubuntu/adaptive-deep-networks

# 克隆代码
git clone https://github.com/sunnyang1/Adaptive-Deep-Networks.git .

# 查看文件
ls -la
```

你应该看到类似这样的目录结构：
```
AGENTS.md                      docs/
Adaptive_Deep_Networks_...     experiments/
LICENSE                        matdo_paper.tex
Makefile                       pyproject.toml
PROJECT_ORGANIZATION.md        requirements.txt
README.md                      results/
archive/                       scripts/
configs/                       src/
data/                          tasks/
```

---

## 🏋️ 第二步：训练模型（已更新为统一入口）

### 3.1 了解模型大小

| 模型 | 参数量 | 显存需求 | 训练时间 (A100 80G) | 适用场景 |
|------|--------|----------|---------------------|----------|
| Small | 1.1B | ~16GB | 2-4 小时 | 测试/开发 |
| Medium | 5.7B | ~40GB | 6-12 小时 | **推荐** |
| Large | 23B | ~80GB+ | 24-48 小时 | 需要多卡 |

对于单张 A100 80GB，**推荐训练 Medium 模型**（5.7B 参数）。

### 3.2 使用 tmux 保持会话（重要！）

训练过程中如果 SSH 断开，训练会中断。**tmux** 可以让训练在后台继续运行。

```bash
# 创建一个新的 tmux 会话（命名为 training）
tmux new-session -s training

# 现在你在 tmux 会话中，底部会显示绿色状态栏
```

**tmux 常用命令：**
- `Ctrl+b 然后按 d`：暂时离开会话（训练继续在后台运行）
- `tmux attach -t training`：重新连接到会话
- `tmux ls`：列出所有会话
- `Ctrl+b 然后按 %`：垂直分屏
- `Ctrl+b 然后按 "`：水平分屏

### 3.3 开始训练 Medium 模型（推荐）

在 tmux 会话中执行：

```bash
# 1. 进入项目目录
cd ~/adaptive-deep-networks

# 2. 激活虚拟环境（必须！）
source venv/bin/activate

# 3. 创建结果目录
mkdir -p results/medium_model

# 4. 开始训练（统一入口）
python3 scripts/training/train_model.py \
    --model-size medium \
    --output-dir results/medium_model \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-4 \
    --seq-len 512 \
    --train-samples 50000 \
    --val-samples 5000
```

**参数解释：**
| 参数 | 值 | 说明 |
|------|-----|------|
| `--model-size medium` | 中等模型 | 单卡 A100 80GB 推荐 |
| `--epochs 3` | 训练 3 轮 | 完整遍历数据集 3 次 |
| `--batch-size 4` | 批次大小 4 | 根据显存调整，A100 80G 可设 4-8 |
| `--lr 2e-4` | 学习率 0.0002 | 控制参数更新速度 |
| `--seq-len 512` | 序列长度 512 | 每次处理的 token 数 |
| `--train-samples 50000` | 训练样本 5 万 | 用于快速验证 |
| `--val-samples 5000` | 验证样本 5 千 | 评估模型性能 |

**训练过程中的输出示例：**
```
======================================================================
MEDIUM MODEL (AttnRes-M) - ~5.7B Parameters
======================================================================
Architecture:
  Layers: 56
  Hidden dim: 2496
  Attention heads: 16
  Head dim: 156
  AttnRes blocks: 8
  Layers per block: 7

Optimal Ratios (Paper §5.4.1):
  d_model/L_b = 44.6 (optimal: ~45)
  H/L_b = 0.29 (optimal: ~0.3)

Hardware Requirements:
  GPU Memory: ~24GB (BF16) or ~48GB (FP32)
  Recommended: 4x A100 80GB or 8x H20 96GB
======================================================================

Epoch 1/3:   5%|███▌| 2500/50000 [15:32<5:12:34, loss=2.3456]
```

### 3.4 监控训练进度

**方法 1：在 tmux 中查看**
```bash
# 如果已断开，重新连接
tmux attach -t training

# 实时查看 GPU 使用情况（需要另开一个终端）
watch -n 1 nvidia-smi
```

**方法 2：新开终端查看 GPU 状态**
```bash
# 在新终端中
ssh ubuntu@你的IP
nvidia-smi
```

**预期 GPU 使用率：**
- GPU 利用率：90-100%
- 显存使用：约 40-60GB
- 温度：正常 60-80°C

**方法 3：查看训练日志**
```bash
# 训练日志保存在输出目录
cat results/medium_model/training.log

# 实时查看
tail -f results/medium_model/training.log
```

### 3.5 训练完成后的文件（当前版本）

训练完成后，`results/medium_model/` 目录下会有（以当前脚本实现为准）：

```
results/medium_model/
├── checkpoints/                # 按 epoch/step 保存的检查点
│   ├── checkpoint_epoch_*.pt
│   └── best.pt (如果产生)
└── training_results.json       # 训练历史 + paper_alignment 检查结果
```

---

## 🧠 第三步：模型推理（使用训练好的模型）

### 4.1 快速测试模型（前向推理 smoke test）

创建一个测试脚本：

```bash
# 创建测试文件
cat > test_model.py << 'EOF'
import torch
import sys
sys.path.insert(0, '.')

from src.models.adaptive_transformer import AdaptiveTransformer
from src.models.configs import AttnResMediumConfig

print("加载模型...")
config = AttnResMediumConfig()
model = AdaptiveTransformer(config)

# 可选：加载训练好的检查点（按你的实际文件名修改）
# ckpt = torch.load("results/medium_model/checkpoints/checkpoint_epoch_1.pt", map_location="cpu")
# model.load_state_dict(ckpt["model_state_dict"], strict=False)

model = model.cuda().eval()
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"模型已加载到 GPU: {next(model.parameters()).device}")

# 测试前向传播
print("\n测试推理...")
dummy_input = torch.randint(0, 50000, (1, 100)).cuda()
with torch.no_grad():
    output = model(dummy_input)
print(f"输入形状: {dummy_input.shape}")
print(f"输出形状: {output.shape}")
print("✅ 推理测试成功！")
EOF

# 运行测试
python test_model.py
```

### 4.2 运行官方基准测试

```bash
# 创建结果目录
mkdir -p results/benchmarks

# 运行所有基准测试
python3 scripts/evaluation/run_benchmarks.py \
    --model-size medium \
    --benchmarks all \
    --output-dir results/benchmarks \
    --device cuda
```

**可用的基准测试：**
| 测试 | 说明 | 预计时间 |
|------|------|----------|
| `needle` | 针在干草堆（长上下文检索） | 10-20 分钟 |
| `math` | MATH 数学推理 | 30-60 分钟 |
| `flop` | FLOP 效率分析 | 5 分钟 |
| `ablation` | 消融研究 | 20-30 分钟 |
| `all` | 全部运行 | 1-2 小时 |

### 4.3 运行特定实验

**Needle-in-Haystack 测试（长文本检索能力）：**
```bash
# 单独运行 needle 测试
python3 scripts/evaluation/run_benchmarks.py \
    --model-size medium \
    --benchmarks needle \
    --output-dir results/needle_test \
    --device cuda
```

**预期结果：**
- 256K 上下文：准确率 > 85%
- 512K 上下文：准确率 > 80%

### 4.4 使用模型生成文本

创建交互式生成脚本：

```bash
cat > generate_text.py << 'EOF'
import torch
import sys
sys.path.insert(0, 'src')

from models.adaptive_transformer import create_adaptive_transformer
from src.models.configs import AttnResMediumConfig
from transformers import AutoTokenizer

# 加载模型
print("加载模型...")
config = AttnResMediumConfig()
model = create_adaptive_transformer(config)

# 尝试加载训练好的权重
try:
    checkpoint = torch.load('results/medium_model/checkpoint-final/pytorch_model.bin')
    model.load_state_dict(checkpoint)
    print("✅ 已加载训练好的权重")
except:
    print("⚠️ 使用随机初始化的权重（请先用 train_model.py 训练）")

model = model.cuda().eval()

# 简单的 tokenizer（实际应使用对应 tokenizer）
# 这里使用 GPT2 tokenizer 作为示例
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 生成文本
prompt = "The future of artificial intelligence is"
print(f"\n提示: {prompt}")
print("="*50)

input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

with torch.no_grad():
    for _ in range(50):  # 生成 50 个 token
        outputs = model(input_ids)
        next_token_logits = outputs[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

generated = tokenizer.decode(input_ids[0])
print(f"生成: {generated}")
EOF

python generate_text.py
```

---

## 🔧 第四步：高级用法

### 5.1 使用 DeepSpeed 加速训练（推荐用于大模型）

DeepSpeed 可以更高效地利用显存，支持更大的 batch size。

```bash
# 安装 DeepSpeed
pip install deepspeed==0.12.0

# 使用 DeepSpeed 训练
deepspeed --num_gpus=1 scripts/training/train_model.py \
    --model-size medium \
    --output-dir results/medium_ds \
    --epochs 3 \
    --batch-size 8 \
    --deepspeed configs/ds_config_h20.json
```

### 5.2 多卡训练（如果你有多张 A100）

```bash
# 4 卡并行训练
torchrun --nproc_per_node=4 scripts/training/train_model.py \
    --model-size medium \
    --output-dir results/medium_4gpu \
    --epochs 3 \
    --batch-size 2 \
    --distributed
```

### 5.3 Paper 对齐训练（推荐）

```bash
# 小模型（严格 paper_alignment 校验）
make train-paper-small OUTPUT_DIR=results/small_paper

# 中模型
make train-paper-medium OUTPUT_DIR=results/medium_paper

# 大模型
make train-paper-large OUTPUT_DIR=results/large_paper
```

### 5.4 恢复中断的训练

如果训练中断，可以从检查点恢复：

```bash
# 找到最新的检查点
ls -lt results/medium_model/checkpoints/ | head -20

# 通过统一入口继续训练
python3 scripts/training/train_model.py \
    --model-size medium \
    --output-dir results/medium_model \
    --epochs 3 \
    --resume results/medium_model/checkpoints/checkpoint_epoch_1.pt
```

---

## 🐛 故障排除

### 问题 1：CUDA Out of Memory（显存不足）

**症状：**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**解决：**
```bash
# 减小 batch size
python3 scripts/training/train_model.py --model-size medium --batch-size 2  # 甚至 1

# 或使用梯度累积（保持有效 batch size 不变）
python3 scripts/training/train_model.py \
    --model-size medium \
    --batch-size 1 \
    --grad-accum 4  # 实际 batch = 1 * 4 = 4
```

### 问题 2：训练速度很慢

**检查 GPU 利用率：**
```bash
# 持续监控
watch -n 1 nvidia-smi

# 如果 GPU 利用率 < 50%，可能是数据加载瓶颈
# 数据加载相关参数在当前训练入口未暴露 num_workers，可先降低 seq_len/batch-size 稳定吞吐
python3 scripts/training/train_model.py --model-size medium --batch-size 2 --seq-len 384
```

### 问题 3：SSH 断开，训练中断

**解决：** 使用 tmux 或 screen
```bash
# 创建 tmux 会话
tmux new -s my_training

# 在会话中运行训练
python3 scripts/training/train_model.py --model-size medium ...

# 按 Ctrl+b，然后按 d 分离会话

# 之后重新连接
tmux attach -t my_training
```

### 问题 4：导入错误（ModuleNotFoundError）

**症状：**
```
ModuleNotFoundError: No module named 'src'
```

**解决：**
```bash
# 确保在项目根目录
cd ~/adaptive-deep-networks
pwd  # 确认路径

# 确保激活了虚拟环境
source venv/bin/activate

# 确保 PYTHONPATH 包含 src
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### 问题 5：Git 连接超时

**症状：**
```
fatal: unable to access 'https://github.com/...': Connection timed out
```

**解决：**
```bash
# 使用 SSH 代替 HTTPS
git clone git@github.com:sunnyang1/Adaptive-Deep-Networks.git

# 或配置代理
git config --global http.proxy http://your-proxy:port
git config --global https.proxy https://your-proxy:port
```

---

## 📊 第五步：监控和日志

### 6.1 使用 Weights & Biases 监控（可选）

```bash
# 登录 wandb
wandb login

# 训练时会自动记录指标
# 在 https://wandb.ai 查看实时图表
```

### 6.2 查看训练曲线

训练脚本会自动保存 `loss_plot.png`：

```bash
# 查看损失曲线
ls results/medium_model/*.png

# 如果安装了 matplotlib，可以实时查看
python -c "
import matplotlib.pyplot as plt
import json

with open('results/medium_model/metrics.json') as f:
    metrics = json.load(f)

plt.plot(metrics['train_loss'])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('loss_curve.png')
print('图表已保存到 loss_curve.png')
"
```

### 6.3 系统监控

```bash
# 查看 GPU 状态（每 1 秒刷新）
watch -n 1 nvidia-smi

# 查看 CPU 和内存
htop

# 查看磁盘空间
df -h

# 查看当前目录大小
du -sh .
du -sh results/*
```

---

## ✅ 检查清单

在开始训练前，确认以下事项：

- [ ] 已连接到 A100 服务器
- [ ] 已创建并激活虚拟环境 `(venv)`
- [ ] PyTorch CUDA 可用（`torch.cuda.is_available()` 返回 True）
- [ ] 已克隆代码仓库
- [ ] 有足够的磁盘空间（`df -h` 显示至少 50GB 可用）
- [ ] 已创建 tmux 会话（防止 SSH 断开）
- [ ] 知道如何查看训练日志
- [ ] 知道如何监控 GPU 状态

---

## 📚 下一步

完成训练和推理后，你可以：

1. **调优超参数**：尝试不同的学习率、batch size
2. **训练 Large 模型**：使用多卡训练 23B 模型
3. **运行完整实验**：`python experiments/run_experiments_unified.py --category paper`
4. **部署模型**：导出为 ONNX 或 TensorRT 格式
5. **阅读论文**：`Adaptive_Deep_Networks_Query_Optimization_REVISED.md`

---

## 🆘 获取帮助

如果遇到问题：

1. 查看训练结果：`cat results/medium_model/training_results.json`
2. 检查 GPU 状态：`nvidia-smi`
3. 查看项目文档：`docs/`
4. 在 GitHub 上提交 Issue

---

**祝训练顺利！🎉**

*最后更新: 2026-04-07*
