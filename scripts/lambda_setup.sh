#!/bin/bash
# Lambda AI Setup Script for Adaptive Deep Networks Validation

set -e

echo "======================================"
echo "Lambda AI Setup for ADN Validation"
echo "======================================"

# Configuration
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"
PROJECT_DIR="~/adaptive-deep-networks"
VENV_DIR="$PROJECT_DIR/venv"

echo ""
echo "Step 1: System Updates"
echo "----------------------"
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux

echo ""
echo "Step 2: Python Environment"
echo "--------------------------"
# Install Python if needed
if ! command -v python$PYTHON_VERSION &> /dev/null; then
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python$PYTHON_VERSION python$PYTHON_VERSION-venv python$PYTHON_VERSION-dev
fi

# Create project directory
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    python$PYTHON_VERSION -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

echo ""
echo "Step 3: Install PyTorch with CUDA $CUDA_VERSION"
echo "------------------------------------------------"
pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Step 4: Install Dependencies"
echo "-----------------------------"
pip install \
    transformers==4.35.0 \
    datasets==2.14.0 \
    accelerate==0.24.0 \
    flash-attn==2.3.0 \
    triton==2.1.0 \
    numpy==1.24.0 \
    scipy==1.11.0 \
    matplotlib==3.8.0 \
    seaborn==0.13.0 \
    pandas==2.0.0 \
    tqdm==4.66.0 \
    wandb==0.15.0 \
    pytest==7.4.0 \
    pytest-cov==4.1.0 \
    black==23.0.0 \
    flake8==6.1.0 \
    mypy==1.6.0

echo ""
echo "Step 5: Verify Installation"
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
echo "Step 6: Clone Repository (if needed)"
echo "------------------------------------"
if [ ! -d "$PROJECT_DIR/.git" ]; then
    # Copy current code to project directory
    echo "Please ensure code is copied to $PROJECT_DIR"
fi

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To activate environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run validation:"
echo "  cd $PROJECT_DIR"
echo "  python scripts/run_benchmarks.py"
echo ""
