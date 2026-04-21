#!/bin/bash
# AutoDL H20-NVLink 服务器专用设置脚本
# GPU: 4x H20 96GB (NVLink)
# CPU: 16核 AMD EPYC 9K84
# 内存: 150 GB
# CUDA: ≤ 13.0, 驱动: 580.65.06

set -e

echo "======================================"
echo "AutoDL H20-NVLink Setup for ADN"
echo "======================================"
echo ""
echo "硬件配置:"
echo "  GPU: H20-NVLink 96GB × 4"
echo "  CPU: AMD EPYC 9K84 16核"
echo "  内存: 150 GB"
echo "  CUDA: ≤ 13.0"
echo "======================================"

# AutoDL H20 环境配置
PROJECT_DIR="/root/autodl-tmp/adaptive-deep-networks"
DATA_DIR="/root/autodl-tmp/data"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints"
CACHE_DIR="/root/autodl-tmp/cache"

echo ""
echo "Step 1: Create Directories"
echo "---------------------------"
mkdir -p $PROJECT_DIR
mkdir -p $DATA_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p $CACHE_DIR
# 创建 HuggingFace 缓存目录
mkdir -p $CACHE_DIR/huggingface
echo "Project: $PROJECT_DIR"
echo "Data: $DATA_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Cache: $CACHE_DIR"

echo ""
echo "Step 2: Check H20 GPUs"
echo "----------------------"
nvidia-smi
echo ""
nvidia-smi topo -m
echo ""

# 检查 CUDA 版本
CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
echo "CUDA Version: $CUDA_VERSION"
echo "Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo ""

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /root/miniconda3
    export PATH="/root/miniconda3/bin:$PATH"
    echo 'export PATH="/root/minoconda3/bin:$PATH"' >> ~/.bashrc
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
echo "Step 4: Install PyTorch for H20 (CUDA 12.1+)"
echo "---------------------------------------------"
pip install --upgrade pip

# H20 需要 CUDA 12.1+ 和 PyTorch 2.1+
# 由于 CUDA ≤ 13.0，使用 CUDA 12.1 版本
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Step 5: Install Core Dependencies"
echo "----------------------------------"
pip install transformers==4.35.0 datasets==2.14.0 accelerate==0.24.0
pip install triton==2.1.0 numpy==1.24.0 scipy==1.11.0
pip install matplotlib==3.8.0 seaborn==0.13.0 pandas==2.0.0 tqdm==4.66.0
pip install wandb==0.15.0 pytest==7.4.0 pytest-cov==4.1.0

echo ""
echo "Step 6: Install Distributed Training Tools"
echo "-------------------------------------------"
# H20 8卡配置，需要分布式训练工具
pip install deepspeed==0.12.0
pip install ninja

echo ""
echo "Step 7: Optional - Flash Attention 2"
echo "-------------------------------------"
echo "H20 supports Flash Attention 2, but installation may fail due to network."
echo "This is OPTIONAL - training works fine without it."
echo ""
read -p "Try to install Flash Attention? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing Flash Attention (may take 10-20 minutes)..."
    pip install flash-attn==2.3.0 --no-build-isolation || {
        echo ""
        echo "⚠️ Flash Attention installation failed (likely network issue)."
        echo "Training will work without it. You can retry later with:"
        echo "  pip install flash-attn==2.3.0 --no-build-isolation"
    }
else
    echo "Skipping Flash Attention installation."
fi

echo ""
echo "Step 8: Verify H20 Installation"
echo "--------------------------------"
python -c "
import torch
import transformers
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
print('')
print('GPU Details:')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'    Total Memory: {props.total_memory / 1e9:.1f} GB')
    print(f'    Compute Capability: {props.major}.{props.minor}')
    print(f'    Multi Processor Count: {props.multi_processor_count}')
"

echo ""
echo "Step 9: Set Environment Variables for H20"
echo "------------------------------------------"
# 添加到 .bashrc
cat >> ~/.bashrc << 'EOF'

# H20 Multi-GPU Settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export PYTHONPATH=/root/autodl-tmp/adaptive-deep-networks:$PYTHONPATH

# HuggingFace Cache
export HF_HOME=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface

# PyTorch Settings for H20
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
EOF

echo ""
echo "======================================"
echo "H20 Setup Complete!"
echo "======================================"
echo ""
echo "硬件配置摘要:"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  单卡显存: 96 GB"
echo "  总显存: $((96 * 4)) GB (384 GB)"
echo ""
echo "使用指南:"
echo "1. 激活环境: conda activate adn"
echo "2. 进入目录: cd $PROJECT_DIR"
echo "3. 上传代码到: $PROJECT_DIR"
echo ""
echo "传统训练 (需下载数据集到本地):"
echo "  python scripts/training/train_model.py --model-size medium --batch-size 2 --output-dir $CHECKPOINT_DIR"
echo ""
echo "流式加载训练 (推荐! 零本地存储):"
echo "  python scripts/training/train_streaming.py --model-size medium --max-steps 10000"
echo ""
echo "单机多卡训练 (4x H20) + 流式加载:"
echo "  torchrun --nproc_per_node=4 scripts/training/train_streaming.py --model-size medium --max-steps 100000"
echo ""
echo "DeepSpeed + 流式加载 (Large 27B):"
echo "  deepspeed --num_gpus=4 scripts/training/train_streaming.py --model-size large --use-deepspeed"
echo ""
echo "数据盘: /root/autodl-tmp (50GB)"
echo "注意: 使用流式加载避免存储大数据集"
echo "======================================"
