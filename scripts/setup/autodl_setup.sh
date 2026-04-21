#!/bin/bash
# AutoDL 服务器设置脚本
# 适用于 Adaptive Deep Networks 训练

set -e

echo "======================================"
echo "AutoDL Setup for ADN Training"
echo "======================================"

# AutoDL 环境配置
PROJECT_DIR="/root/autodl-tmp/adaptive-deep-networks"
DATA_DIR="/root/autodl-tmp/data"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints"

echo ""
echo "Step 1: Create Directories"
echo "---------------------------"
mkdir -p $PROJECT_DIR
mkdir -p $DATA_DIR
mkdir -p $CHECKPOINT_DIR
echo "Project: $PROJECT_DIR"
echo "Data: $DATA_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"

echo ""
echo "Step 2: Check GPU"
echo "-----------------"
nvidia-smi
echo ""

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /root/miniconda3
    export PATH="/root/miniconda3/bin:$PATH"
    echo 'export PATH="/root/miniconda3/bin:$PATH"' >> ~/.bashrc
fi

echo ""
echo "Step 3: Create Conda Environment"
echo "---------------------------------"
ENV_NAME="adn"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists."
else
    conda create -n $ENV_NAME python=3.10 -y
fi

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo ""
echo "Step 4: Install PyTorch with CUDA"
echo "----------------------------------"
pip install --upgrade pip

# 根据 CUDA 版本安装对应 PyTorch
CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
echo "Detected CUDA version: $CUDA_VERSION"

if [[ "$CUDA_VERSION" == "12.1" ]]; then
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch with CUDA 12.1 (default)..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
fi

echo ""
echo "Step 5: Install Dependencies"
echo "-----------------------------"
pip install transformers==4.35.0 datasets==2.14.0 accelerate==0.24.0
pip install triton==2.1.0 numpy==1.24.0 scipy==1.11.0
pip install matplotlib==3.8.0 seaborn==0.13.0 pandas==2.0.0 tqdm==4.66.0
pip install wandb==0.15.0 pytest==7.4.0 pytest-cov==4.1.0

# 可选：安装 flash-attn（编译时间较长，如有需要可取消注释）
# echo "Installing flash-attn (this may take a while)..."
# pip install flash-attn==2.3.0 --no-build-isolation

echo ""
echo "Step 6: Verify Installation"
echo "----------------------------"
python -c "
import torch
import transformers
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "使用指南:"
echo "1. 激活环境: conda activate adn"
echo "2. 进入目录: cd $PROJECT_DIR"
echo "3. 上传代码到: $PROJECT_DIR"
echo "4. 数据存放于: $DATA_DIR"
echo "5. 检查点存放于: $CHECKPOINT_DIR"
echo ""
echo "训练命令示例:"
echo "  python scripts/training/train_model.py --model-size medium --epochs 3 --batch-size 2"
echo ""
