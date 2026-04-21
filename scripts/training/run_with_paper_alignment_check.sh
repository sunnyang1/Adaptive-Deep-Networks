#!/usr/bin/env bash
set -euo pipefail

# Run training with paper preset, then enforce paper_alignment check.
#
# Example:
#   bash scripts/training/run_with_paper_alignment_check.sh \
#     --model-size medium \
#     --output-dir results/medium_paper_train

MODEL_SIZE="medium"
OUTPUT_DIR=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/training/run_with_paper_alignment_check.sh [options] [-- <extra training args>]

Options:
  --model-size SIZE   t4|small|medium|large (default: medium)
                      (t4 uses --paper-preset-t4: same paper hyperparams + T4 VRAM caps)
  --output-dir PATH   required output directory
  -h, --help          show this help

Everything after "--" is forwarded to the training script.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-size)
      MODEL_SIZE="${2:-}"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"; shift 2 ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: --output-dir is required" >&2
  usage
  exit 1
fi

TRAIN_SCRIPT="scripts/training/train_${MODEL_SIZE}.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "ERROR: training script not found: $TRAIN_SCRIPT" >&2
  exit 1
fi

PRESET_FLAG=(--paper-preset)
if [[ "$MODEL_SIZE" == "t4" ]]; then
  PRESET_FLAG=(--paper-preset-t4)
fi

echo "Running $TRAIN_SCRIPT with paper preset (${PRESET_FLAG[*]})..."
python3 "$TRAIN_SCRIPT" \
  --output-dir "$OUTPUT_DIR" \
  "${PRESET_FLAG[@]}" \
  --deterministic \
  "${EXTRA_ARGS[@]}"

RESULTS_JSON="$OUTPUT_DIR/training_results.json"
echo "Checking paper alignment in $RESULTS_JSON ..."
python3 scripts/training/check_paper_alignment.py \
  --results "$RESULTS_JSON" \
  --strict

echo "PASS: training run is paper-aligned."

