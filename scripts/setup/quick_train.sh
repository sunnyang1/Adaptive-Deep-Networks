#!/bin/bash
# =============================================================================
# A100 一键训练启动脚本
# 使用方法: bash scripts/setup/quick_train.sh [small|medium|large]
# =============================================================================

set -e

# 默认训练 medium 模型
MODEL_SIZE="${1:-medium}"

echo "============================================================"
echo "  ADN 一键训练启动器"
echo "============================================================"
echo "模型大小: $MODEL_SIZE"
echo ""

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}!${NC} 未检测到虚拟环境，尝试自动激活..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo -e "${GREEN}✓${NC} 虚拟环境已激活"
    else
        echo -e "${RED}✗${NC} 未找到虚拟环境，请先运行: bash scripts/setup/a100_setup.sh"
        exit 1
    fi
fi

# 检查 GPU
echo "检查 GPU 状态..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo ""

# 创建 tmux 会话名
SESSION_NAME="adn_${MODEL_SIZE}_$(date +%m%d_%H%M)"

# 根据模型大小设置参数
case $MODEL_SIZE in
    small)
        BATCH_SIZE=8
        EPOCHS=3
        SEQ_LEN=512
        TRAIN_SAMPLES=20000
        SCRIPT="scripts/training/train_small.py"
        MEM_REQ=16000  # 16GB
        ;;
    medium)
        BATCH_SIZE=4
        EPOCHS=3
        SEQ_LEN=512
        TRAIN_SAMPLES=50000
        SCRIPT="scripts/training/train_medium.py"
        MEM_REQ=40000  # 40GB
        ;;
    large)
        BATCH_SIZE=1
        EPOCHS=2
        SEQ_LEN=512
        TRAIN_SAMPLES=100000
        SCRIPT="scripts/training/train_large.py"
        MEM_REQ=80000  # 80GB
        ;;
    *)
        echo "错误: 未知的模型大小 '$MODEL_SIZE'"
        echo "用法: bash scripts/setup/quick_train.sh [small|medium|large]"
        exit 1
        ;;
esac

# 检查显存是否足够
if [ "$GPU_MEM" -lt "$MEM_REQ" ]; then
    echo -e "${YELLOW}!${NC} 警告: 可用显存 ${GPU_MEM}MB 可能不足以训练 $MODEL_SIZE 模型"
    echo "建议显存: ${MEM_REQ}MB"
    read -p "是否继续? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 创建输出目录
OUTPUT_DIR="results/${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 构建训练命令
TRAIN_CMD="python $SCRIPT \\
    --output-dir $OUTPUT_DIR \\
    --epochs $EPOCHS \\
    --batch-size $BATCH_SIZE \\
    --seq-len $SEQ_LEN \\
    --train-samples $TRAIN_SAMPLES \\
    --lr 2e-4"

echo "训练配置:"
echo "------------------------------------------------------------"
echo "模型大小:    $MODEL_SIZE"
echo "批次大小:    $BATCH_SIZE"
echo "训练轮数:    $EPOCHS"
echo "序列长度:    $SEQ_LEN"
echo "训练样本:    $TRAIN_SAMPLES"
echo "输出目录:    $OUTPUT_DIR"
echo "------------------------------------------------------------"
echo ""

# 创建 tmux 会话
echo "创建 tmux 会话: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME"

# 在 tmux 会话中执行训练命令
tmux send-keys -t "$SESSION_NAME" "
cd $(pwd)
source venv/bin/activate
echo '============================================================'
echo '  训练开始: $MODEL_SIZE 模型'
echo '============================================================'
echo ''
$TRAIN_CMD
echo ''
echo '训练完成！按 Enter 退出'
read
" C-m

echo -e "${GREEN}✓${NC} 训练已在后台启动"
echo ""
echo "会话名称: $SESSION_NAME"
echo ""
echo "操作命令:"
echo "------------------------------------------------------------"
echo "查看训练进度:  tmux attach -t $SESSION_NAME"
echo "分离会话:      按 Ctrl+b 然后按 d"
echo "查看所有会话:  tmux ls"
echo "查看 GPU:      nvidia-smi"
echo "查看日志:      tail -f $OUTPUT_DIR/training.log"
echo "------------------------------------------------------------"
echo ""
echo -e "${GREEN}训练正在进行中...${NC}"
echo ""

# 提示用户是否立即查看
read -p "是否立即查看训练进度? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    tmux attach -t "$SESSION_NAME"
fi
