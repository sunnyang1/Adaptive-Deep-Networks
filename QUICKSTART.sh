#!/bin/bash
# MATDO-E A100 快速启动脚本
# 使用方法: bash QUICKSTART.sh

set -e

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "MATDO-E Quick Start for A100 80G"
echo "=========================================="

# 1. 运行环境检查
echo ""
echo -e "${YELLOW}Step 1: Checking environment...${NC}"
if python scripts/setup/check_env.py; then
    echo -e "${GREEN}Environment check passed!${NC}"
else
    echo -e "${YELLOW}Setting up environment...${NC}"
    bash scripts/setup/a100_setup.sh
fi

# 2. 激活环境
source ~/miniconda3/bin/activate matdo 2>/dev/null || conda activate matdo

# 3. 检查数据
echo ""
echo -e "${YELLOW}Step 2: Checking data...${NC}"
if [ ! -d "data/raw" ]; then
    echo "Creating data directories..."
    mkdir -p data/raw data/processed data/engram
fi

if [ ! -f "data/engram/wikipedia_hnsw.index" ]; then
    echo -e "${YELLOW}Engram index not found. Building with mock data...${NC}"
    python -c "
import numpy as np
import os
os.makedirs('data/engram', exist_ok=True)
# 创建模拟索引文件
np.random.seed(42)
embeddings = np.random.randn(128000, 384).astype('float32')
np.save('data/engram/mock_embeddings.npy', embeddings)
print('Mock Engram data created')
"
fi

# 4. 检查配置文件
echo ""
echo -e "${YELLOW}Step 3: Checking configs...${NC}"
if [ ! -f "configs/train_small_example.yaml" ]; then
    echo "Creating example config..."
    cat > configs/train_small_example.yaml << 'EOF'
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
  num_epochs: 1
  warmup_steps: 100
  max_seq_len: 4096
  num_workers: 4
  
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
  save_every: 500
EOF
fi

# 5. 运行测试
echo ""
echo -e "${YELLOW}Step 4: Running quick test...${NC}"
python -c "
from experiments.matdo.matdo_e.solver import MATDOESolver
from experiments.matdo.common.config import config

print('Testing MATDO-E solver...')
solver = MATDOESolver()

for rho in [0.8, 0.9, 0.95, 0.99]:
    opt = solver.solve(rho)
    status = 'ARBITRAGE' if opt.is_arbitrage else 'NORMAL'
    print(f'  rho={rho:.2f}: R={opt.R}, M={opt.M}, T={opt.T}, E={opt.E}, {status}')

print('\nArbitrage Inequality: {} > {} = {}'.format(
    config.zeta, 
    config.eta/(config.E_max*config.E_target),
    config.check_arbitrage_inequality()
))

print('\n✓ MATDO-E is ready!')
"

# 完成
echo ""
echo "=========================================="
echo -e "${GREEN}Quick Start Complete!${NC}"
echo "=========================================="
echo ""
echo "You can now:"
echo ""
echo "1. Run validation experiments:"
echo "   python experiments/matdo/run_all_experiments.py"
echo ""
echo "2. Run vLLM integration tests:"
echo "   python experiments/matdo/vllm_integration/run_all_vllm_experiments.py"
echo ""
echo "3. Start training (if you have data):"
echo "   python scripts/train.py --config configs/train_small_example.yaml"
echo ""
echo "4. Read the full guide:"
echo "   cat MATDO_E_A100_BEGINNER_GUIDE.md"
echo ""
