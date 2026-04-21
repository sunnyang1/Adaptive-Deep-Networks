# MATDO-E 新手快速入门指南

欢迎来到MATDO-E项目！如果你是新手且拥有一台A100 80G机器，本指南将帮助你从零开始。

## 🚀 最快开始方式（5分钟）

在A100机器上依次执行以下命令：

```bash
# 1. 克隆代码仓库
git clone https://github.com/your-org/Adaptive-Deep-Networks.git
cd Adaptive-Deep-Networks

# 2. 运行一键启动脚本
bash scripts/setup/QUICKSTART.sh
```

如果一切正常，你会看到：
```
==========================================
Quick Start Complete!
==========================================

You can now:
1. Run validation experiments...
```

## 📋 详细步骤

如果一键脚本遇到问题，请按以下步骤手动操作：

### 步骤1: 环境检查

```bash
# 检查GPU
nvidia-smi

# 应该看到类似：
# NVIDIA A100 80GB
# CUDA Version: 12.0
```

### 步骤2: 运行环境检查脚本

```bash
python scripts/setup/check_env.py
```

这个脚本会检查：
- ✅ Python版本 (需要>=3.10)
- ✅ GPU可用性 (需要A100)
- ✅ CUDA版本 (需要>=11.8)
- ✅ 磁盘空间 (需要>=100GB)
- ✅ 关键依赖
- ✅ MATDO-E模块

### 步骤3: 安装依赖（如果检查失败）

```bash
# 运行自动安装脚本
bash scripts/setup/a100_setup.sh

# 或者手动安装
conda create -n matdo python=3.10 -y
conda activate matdo
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 步骤4: 验证安装

```bash
python -c "
from experiments.matdo.matdo_e.solver import MATDOESolver
solver = MATDOESolver()
opt = solver.solve(0.95)
print(f'MATDO-E working! rho=0.95 -> arbitrage={opt.is_arbitrage}')
"
```

## 🧪 运行实验

### 实验1: 快速验证（无需训练）

```bash
# 运行MATDO-E核心实验（使用模拟）
python experiments/matdo/run_all_experiments.py

# 运行vLLM集成实验
python experiments/matdo/vllm_integration/run_all_vllm_experiments.py
```

### 实验2: 训练小型模型（需要1-2小时）

```bash
# 确保有数据（如果没有会自动下载示例数据）
python scripts/data/download_all.py

# 开始训练
python scripts/train.py --config configs/train_small_example.yaml
```

训练时会显示：
```
Epoch 1/3 - Step 100/5000 - Loss: 2.345 - GPU: 42.3GB/80GB
```

### 实验3: 推理测试

```bash
# 基础推理
python scripts/inference.py \
    --checkpoint checkpoints/small/final.pt \
    --prompt "Explain quantum computing" \
    --max_new_tokens 100

# 高显存压力推理（MATDO-E特色）
python scripts/inference_matdo_e.py \
    --checkpoint checkpoints/small/final.pt \
    --prompt "Explain quantum computing" \
    --rho 0.99 \
    --enable_arbitrage
```

## 📚 学习路径

### 第1天：理解概念
1. 阅读论文 `docs/papers/matdo-e_revised_paper.md`
2. 理解四个维度：R(量化), M(上下文), T(TTA步数), E(Engram)
3. 运行求解器看配置变化：
   ```bash
   python -c "from experiments.matdo.matdo_e.solver import MATDOESolver; \
   s=MATDOESolver(); \
   [print(f'rho={r}: {s.solve(r)}') for r in [0.8,0.9,0.95,0.99]]"
   ```

### 第2天：运行实验
1. 完成所有验证实验
2. 查看生成的图表 `figures/`
3. 理解实验结果

### 第3天：深入代码
1. 阅读 `MATDO_E_IMPLEMENTATION.md`
2. 修改参数观察效果（如改变zeta, eta）
3. 尝试训练自己的模型

## ❓ 常见问题

### Q: 遇到CUDA OOM错误
**A:** 减小batch size或序列长度：
```yaml
# 编辑configs/train_small_example.yaml
training:
  batch_size: 2  # 从4改为2
  max_seq_len: 4096  # 从8192改为4096
```

### Q: 训练速度太慢
**A:** 
- 使用tmux保持会话：`tmux new -s train`
- 监控GPU：`watch -n 1 nvidia-smi`
- 启用混合精度（如果还没启用）

### Q: 没有A100能用这个项目吗？
**A:** 可以，但需要调整：
- 使用更小的模型配置
- 减小batch size和序列长度
- 部分功能可能受限

### Q: Engram索引构建失败
**A:** 项目包含模拟Engram数据，可以直接使用：
```python
# 代码会自动使用mock数据
from experiments.matdo.matdo_e.engram_manager import MockFaissIndex
```

## 📖 更多资源

- **完整指南**: `docs/guides/MATDO_E_A100_BEGINNER_GUIDE.md`（详细教程）
- **实现文档**: `experiments/matdo/MATDO_E_IMPLEMENTATION.md`（技术细节）
- **项目架构**: `AGENTS.md`（代码结构）
- **论文**: `docs/papers/matdo-e_revised_paper.md`

## 🆘 获得帮助

如果遇到问题：
1. 查看错误日志：`checkpoints/*/training.log`
2. 运行检查脚本：`python scripts/setup/check_env.py`
3. 检查GPU状态：`nvidia-smi`
4. 查看详细指南：`cat docs/guides/MATDO_E_A100_BEGINNER_GUIDE.md`

## ✅ 成功标志

当你看到以下输出，说明一切正常：

```
MATDO-E Environment Checker
...
Total: 6/6 checks passed

✓ Environment is ready for MATDO-E!
```

祝你实验顺利！🎉
