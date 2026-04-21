#!/bin/bash
# H20 快速启动脚本

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Adaptive Deep Networks - H20 (4卡) 快速启动${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# 检查环境
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda，请先运行: bash scripts/autodl_h20_setup.sh"
    exit 1
fi

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate adn 2>/dev/null || {
    echo "错误: 未找到 adn 环境，请先运行: bash scripts/autodl_h20_setup.sh"
    exit 1
}

# 检查 GPU
echo -e "${GREEN}GPU 状态:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu \
    --format=table

echo ""
echo -e "${GREEN}可用训练选项:${NC}"
echo ""
echo "1) Small 模型 (2.2B) - 单卡流式训练"
echo "   显存需求: ~24GB"
echo "   命令: python scripts/training/train_streaming.py --model-size small --max-steps 10000"
echo ""
echo "2) Medium 模型 (8.7B) - 单卡流式训练"
echo "   显存需求: ~92GB"
echo "   命令: python scripts/training/train_streaming.py --model-size medium --max-steps 10000"
echo ""
echo "3) Medium 模型 (8.7B) - 4卡流式训练 (推荐!)"
echo "   显存需求: ~92GB per GPU"
echo "   命令: torchrun --nproc_per_node=4 scripts/training/train_streaming.py --model-size medium --max-steps 100000"
echo ""
echo "4) Large 模型 (27B) - DeepSpeed + 流式 (4卡)"
echo "   显存需求: 4x 96GB + CPU Offload"
echo "   命令: deepspeed --num_gpus=4 scripts/training/train_streaming.py --model-size large --use-deepspeed"
echo ""
echo "5) FineWeb 数据集流式训练示例"
echo "   命令: python scripts/training/train_streaming.py --model-size small --dataset fineweb --max-steps 50000"
echo ""
echo "6) 运行基准测试"
echo "   命令: python scripts/evaluation/run_benchmarks.py --model-size medium --benchmarks all"
echo ""
echo "7) 验证配置"
echo "   命令: python scripts/training/train_h20.py --model-size medium"
echo ""
echo -e "${BLUE}提示: 使用 --dataset fineweb 启用 FineWeb-Edu 数据集流式加载${NC}"
echo -e "${BLUE}======================================${NC}"
echo "数据盘位置: /root/autodl-tmp (50GB)"
echo "检查点位置: /root/autodl-tmp/checkpoints"
echo -e "${BLUE}======================================${NC}"

# 运行验证
read -p "是否运行配置验证? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/training/train_h20.py --model-size medium
fi
