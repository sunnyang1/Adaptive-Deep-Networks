#!/bin/bash
# A100环境一键设置脚本
# 使用方法: bash scripts/setup/a100_setup.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "MATDO-E A100 80G Environment Setup"
echo "=========================================="

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查函数
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 found"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

# 1. 检查GPU
echo ""
echo "Step 1: Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    echo -e "${GREEN}✓${NC} GPU found: $GPU_INFO"
    
    # 检查是否是A100
    if nvidia-smi | grep -q "A100"; then
        echo -e "${GREEN}✓${NC} A100 detected"
    else
        echo -e "${YELLOW}⚠${NC} Warning: Not A100, but will try to continue"
    fi
else
    echo -e "${RED}✗${NC} nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

# 2. 检查CUDA
echo ""
echo "Step 2: Checking CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "${GREEN}✓${NC} CUDA found: $CUDA_VERSION"
    
    # 检查CUDA版本是否>=11.8
    if [ "$(printf '%s\n' "11.8" "$CUDA_VERSION" | sort -V | head -n1)" = "11.8" ]; then
        echo -e "${GREEN}✓${NC} CUDA version is sufficient (>=11.8)"
    else
        echo -e "${YELLOW}⚠${NC} CUDA version might be too old (<11.8)"
    fi
else
    echo -e "${RED}✗${NC} CUDA not found. Please install CUDA toolkit."
    exit 1
fi

# 3. 安装conda（如果没有）
echo ""
echo "Step 3: Checking/Installing Conda..."
if command -v conda &> /dev/null; then
    echo -e "${GREEN}✓${NC} Conda found: $(conda --version)"
else
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    source $HOME/miniconda3/bin/activate
    conda init bash
    echo -e "${GREEN}✓${NC} Miniconda installed"
fi

# 4. 创建Python环境
echo ""
echo "Step 4: Creating Python environment..."
source ~/miniconda3/bin/activate 2>/dev/null || true

if conda env list | grep -q "matdo"; then
    echo -e "${GREEN}✓${NC} Environment 'matdo' already exists"
else
    echo "Creating new environment 'matdo'..."
    conda create -n matdo python=3.10 -y
    echo -e "${GREEN}✓${NC} Environment 'matdo' created"
fi

# 激活环境
source ~/miniconda3/bin/activate matdo

# 5. 安装PyTorch
echo ""
echo "Step 5: Installing PyTorch (CUDA 11.8)..."
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "${GREEN}✓${NC} PyTorch already installed: $PYTORCH_VERSION"
else
    echo "Installing PyTorch 2.1.0 with CUDA 11.8..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu118
    echo -e "${GREEN}✓${NC} PyTorch installed"
fi

# 验证PyTorch GPU
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo -e "${GREEN}✓${NC} PyTorch GPU working: $GPU_NAME"
else
    echo -e "${RED}✗${NC} PyTorch GPU not working!"
    exit 1
fi

# 6. 安装项目依赖
echo ""
echo "Step 6: Installing project dependencies..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname $(dirname $SCRIPT_DIR))"

cd $PROJECT_ROOT

if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓${NC} Project dependencies installed"
else
    echo -e "${YELLOW}⚠${NC} requirements.txt not found, skipping"
fi

# 7. 安装额外依赖
echo ""
echo "Step 7: Installing additional dependencies..."
pip install -q matplotlib seaborn pandas scipy scikit-learn tqdm tensorboard 2>/dev/null || true
echo -e "${GREEN}✓${NC} Additional dependencies installed"

# 8. 创建必要目录
echo ""
echo "Step 8: Creating directory structure..."
mkdir -p data/raw data/processed data/engram
mkdir -p checkpoints/small checkpoints/medium checkpoints/large
mkdir -p results figures logs
echo -e "${GREEN}✓${NC} Directories created"

# 9. 测试MATDO-E核心模块
echo ""
echo "Step 9: Testing MATDO-E core modules..."
python -c "
from experiments.matdo.matdo_e.solver import MATDOESolver
from experiments.matdo.common.config import config
solver = MATDOESolver()
opt = solver.solve(0.95)
print(f'MATDO-E Solver: OK (rho=0.95 -> arbitrage={opt.is_arbitrage})')
" 2>/dev/null && echo -e "${GREEN}✓${NC} MATDO-E solver working" || echo -e "${YELLOW}⚠${NC} MATDO-E solver test failed"

# 10. 生成示例配置
echo ""
echo "Step 10: Generating example configs..."
cat > configs/train_small_example.yaml << 'EOF'
# MATDO-E Small Model Training Config
model:
  name: "matdo_e_small"
  size: "small"
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
  
hardware:
  device: cuda
  mixed_precision: fp16
  
checkpoint:
  save_dir: "checkpoints/small"
  save_every: 1000
EOF
echo -e "${GREEN}✓${NC} Example config created: configs/train_small_example.yaml"

# 完成
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: conda activate matdo"
echo "2. Start training: python scripts/train.py --config configs/train_small_example.yaml"
echo "3. Or run tests: python experiments/matdo/run_all_experiments.py"
echo ""
echo "For detailed guide, see: docs/guides/MATDO_E_A100_BEGINNER_GUIDE.md"
echo ""
