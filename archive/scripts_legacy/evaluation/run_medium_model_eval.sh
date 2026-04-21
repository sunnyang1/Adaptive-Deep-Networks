#!/bin/bash
# Medium Model (7B) Section 5.2 Evaluation Script
# Run this on a GPU server with sufficient memory

set -e

echo "========================================"
echo "Adaptive Deep Networks - Medium Model Evaluation"
echo "========================================"
echo ""
echo "Model: AttnRes-M (7B parameters)"
echo "Requirements: 40GB+ GPU memory or 64GB+ CPU memory"
echo ""

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
    DEVICE="cuda"
else
    echo "WARNING: No GPU detected. Will use CPU (very slow)."
    DEVICE="cpu"
fi

# Create results directory
mkdir -p results

# Run evaluation
echo "Starting Section 5.2 Evaluation..."
echo ""

python scripts/evaluation/eval_5_2.py \
    --model-size medium \
    --device $DEVICE

echo ""
echo "Evaluation complete! Results saved to results/"
