#!/bin/bash
#
# Run Real Model Validation
#
# Usage:
#   bash scripts/run_real_validation.sh [checkpoint_path]
#
# Example:
#   bash scripts/run_real_validation.sh checkpoints/adb_medium.pt
#   bash scripts/run_real_validation.sh  # Run with random initialization

set -e

CHECKPOINT=${1:-""}
MODEL_SIZE="medium"
DEVICE="cuda"

if [ ! -d "results/real_model" ]; then
    mkdir -p results/real_model
fi

echo "========================================"
echo "Real Model Validation"
echo "========================================"

if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
    CHECKPOINT_ARG="--checkpoint $CHECKPOINT"
else
    echo "No checkpoint provided, using random initialization"
    echo "Model size: $MODEL_SIZE"
    CHECKPOINT_ARG="--size $MODEL_SIZE"
fi

echo "Device: $DEVICE"
echo ""

# Run validation
echo "Running all tests..."
python experiments/real_model/validator.py \
    $CHECKPOINT_ARG \
    --device $DEVICE \
    --output-dir results/real_model \
    --all

echo ""
echo "========================================"
echo "Validation complete!"
echo "Results: results/real_model/validation_results.json"
echo "========================================"
