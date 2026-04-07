#!/usr/bin/env bash
set -euo pipefail

# ADN paper verification runner for A100 80G.
# Focuses on REAL measurement paths and avoids target-replay scripts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT=""
MODEL_SIZE="medium"
DEVICE="cuda"
MAX_LENGTH=131072
NUM_SAMPLES=10
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR=""
STRICT=0
NEEDLE_TOL_PP=5.0
THROUGHPUT_TARGET=115.0
INCLUDE_256K=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/evaluation/verify_adn_a100.sh [options]

Options:
  --checkpoint PATH      Model checkpoint path (recommended)
  --size SIZE            Model size: small|medium|large (default: medium)
  --device DEVICE        cuda|cpu (default: cuda)
  --max-length N         Max context length for needle test (default: 131072)
  --num-samples N        Samples per context in needle test (default: 10)
  --include-256k         Also run 256K Needle-in-Haystack (may OOM; off by default)
  --needle-tol-pp X      Needle target tolerance (percentage points, default: 5.0)
  --throughput-target X  Throughput target in tok/s (default: 115)
  --strict               Exit non-zero if P0 report overall is FLAG
  --out-dir PATH         Output directory (default: results/paper_verify_<timestamp>)
  -h, --help             Show help

Examples:
  bash scripts/evaluation/verify_adn_a100.sh --checkpoint checkpoints/adb_medium.pt
  bash scripts/evaluation/verify_adn_a100.sh --size medium --num-samples 20
  bash scripts/evaluation/verify_adn_a100.sh --checkpoint checkpoints/adb_medium.pt --strict --needle-tol-pp 3
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="${2:-}"; shift 2 ;;
    --size)
      MODEL_SIZE="${2:-}"; shift 2 ;;
    --device)
      DEVICE="${2:-}"; shift 2 ;;
    --max-length)
      MAX_LENGTH="${2:-}"; shift 2 ;;
    --num-samples)
      NUM_SAMPLES="${2:-}"; shift 2 ;;
    --include-256k)
      INCLUDE_256K=1; shift 1 ;;
    --needle-tol-pp)
      NEEDLE_TOL_PP="${2:-}"; shift 2 ;;
    --throughput-target)
      THROUGHPUT_TARGET="${2:-}"; shift 2 ;;
    --strict)
      STRICT=1; shift 1 ;;
    --out-dir)
      OUT_DIR="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="results/paper_verify_${RUN_ID}"
fi

mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/commands.log"
MANIFEST="$OUT_DIR/verification_manifest.md"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "ADN A100 Verification Run"
echo "============================================================"
echo "Root:        $ROOT_DIR"
echo "Output:      $OUT_DIR"
echo "Model size:  $MODEL_SIZE"
echo "Device:      $DEVICE"
echo "Checkpoint:  ${CHECKPOINT:-<none; random init>}"
echo "Max length:  $MAX_LENGTH"
echo "Num samples: $NUM_SAMPLES"
echo "Needle tol:  ${NEEDLE_TOL_PP} pp"
echo "Tput target: ${THROUGHPUT_TARGET} tok/s"
echo "Include256K: ${INCLUDE_256K}"
echo "Strict:      ${STRICT}"
echo

echo "[0] Environment checks"
python3 --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found. GPU checks skipped."
fi
python3 - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_version={torch.version.cuda}")
    print(f"gpu_count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"gpu_{i}={torch.cuda.get_device_name(i)}")
PY

CHECKPOINT_ARGS=(--size "$MODEL_SIZE")
if [[ -n "$CHECKPOINT" ]]; then
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 1
  fi
  CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
fi

echo
echo "[1] Static arithmetic checks (paper consistency)"
python3 - <<'PY' > "$OUT_DIR/static_consistency.json"
import json

out = {}
kv_per_token_values = 2 * 8 * 128
bytes_per_layer = kv_per_token_values * 131072 * 2
gb_per_layer = bytes_per_layer / (1024**3)
total_gb_80_layers = gb_per_layer * 80
out["kv_cache"] = {
    "kv_values_per_token": kv_per_token_values,
    "bytes_per_layer_128k": bytes_per_layer,
    "gb_per_layer_128k": gb_per_layer,
    "gb_total_80_layers": total_gb_80_layers
}
base = 40.0
out["compression_table4"] = {
    "fp16_gb": base,
    "3bit_gb": round(base / 5.3, 2),
    "2bit_gb": round(base / 8.0, 2),
    "1bit_gb": round(base / 16.0, 2)
}
out["ponder_amortized"] = {
    "trigger_30pct_step2_overhead3p6x": round(0.30 * 3.6, 3),
    "trigger_30pct_step10_overhead12p8x": round(0.30 * 12.8, 3)
}
print(json.dumps(out, indent=2))
PY
cat "$OUT_DIR/static_consistency.json"

echo
echo "[2] REAL Needle-in-Haystack (P0)"
LENGTHS=(4096 32768)
if [[ "$MAX_LENGTH" -ge 131072 ]]; then
  LENGTHS+=(131072)
fi
if [[ "$INCLUDE_256K" -eq 1 ]]; then
  LENGTHS+=(262144)
fi
python3 experiments/real_model/needle_haystack_real.py \
  "${CHECKPOINT_ARGS[@]}" \
  --device "$DEVICE" \
  --lengths "${LENGTHS[@]}" \
  --num-samples "$NUM_SAMPLES" \
  --output "$OUT_DIR/needle_haystack_real.json"

echo
echo "[3] REAL memory profiling (P0)"
python3 experiments/real_model/memory_profiler.py \
  "${CHECKPOINT_ARGS[@]}" \
  --device "$DEVICE" \
  --context-lengths 4096 8192 16384 32768 65536 131072 \
  --output "$OUT_DIR/memory_profile.json"

echo
echo "[3b] Table 4 bitwidth sweep (REAL prefill throughput)"
python3 scripts/evaluation/benchmark_table4_bitsweep.py \
  "${CHECKPOINT_ARGS[@]}" \
  --device "$DEVICE" \
  --context-len 131072 \
  --repeats 3 \
  --use-attnres \
  --output "$OUT_DIR/table4_bitsweep.json"

echo
echo "[3c] Table 5 component sweep (REAL NIH ablations)"
python3 scripts/evaluation/benchmark_table5_components.py \
  "${CHECKPOINT_ARGS[@]}" \
  --device "$DEVICE" \
  --lengths 4096 32768 131072 \
  --num-samples "$NUM_SAMPLES" \
  --rabitq-bits 1 \
  --qttt-steps 10 \
  --output "$OUT_DIR/table5_components.json"

echo
echo "[4] Throughput probe (P0 support)"
python3 - <<PY
import json
from pathlib import Path
from experiments.real_model.validator import ModelValidator

checkpoint = "${CHECKPOINT}"
kwargs = dict(model_size="${MODEL_SIZE}", device="${DEVICE}", output_dir="${OUT_DIR}/validator_throughput")
if checkpoint:
    kwargs["checkpoint_path"] = checkpoint
validator = ModelValidator(**kwargs)
validator.load_model()
res = validator.run_throughput_test()
out = Path("${OUT_DIR}") / "throughput_result.json"
out.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(f"Wrote {out}")
PY

echo
echo "[5] FLOP equivalence analysis (theory support)"
python3 - <<PY
import json
from src.benchmarks.flop_analysis import run_flop_analysis
res = run_flop_analysis(output_path="${OUT_DIR}/flop_analysis.json")
print(json.dumps(res, indent=2))
PY

echo
echo "[6] Build consolidated P0 report"
python3 scripts/evaluation/collect_adn_p0_report.py \
  --run-dir "$OUT_DIR" \
  --needle-tol-pp "$NEEDLE_TOL_PP" \
  --throughput-target "$THROUGHPUT_TARGET"

if [[ "$STRICT" -eq 1 ]]; then
  python3 - <<PY
import json, sys
from pathlib import Path
p = Path("${OUT_DIR}") / "p0_summary.json"
if not p.exists():
    print("STRICT: missing p0_summary.json", file=sys.stderr)
    sys.exit(2)
obj = json.loads(p.read_text(encoding="utf-8"))
ok = bool(obj.get("overall_passed"))
print(f"STRICT overall_passed={ok}")
sys.exit(0 if ok else 1)
PY
fi

cat > "$MANIFEST" <<EOF
# ADN Verification Manifest (${RUN_ID})

## Run Configuration
- Model size: \`${MODEL_SIZE}\`
- Device: \`${DEVICE}\`
- Checkpoint: \`${CHECKPOINT:-random_init}\`
- Max context length: \`${MAX_LENGTH}\`
- Samples per context: \`${NUM_SAMPLES}\`

## Executed (REAL measurement path)
- \`experiments/real_model/validator.py --test needle\`
- \`experiments/real_model/memory_profiler.py\`
- \`experiments/real_model/validator.py --test throughput\`
- \`src/benchmarks/flop_analysis.py\`
- Static arithmetic consistency checks

## Outputs
- \`static_consistency.json\`
- \`needle_haystack_real.json\`
- \`memory_profile.json\`
- \`throughput_result.json\`
- \`flop_analysis.json\`
- \`commands.log\`

## Not used as evidence
The following scripts are target-replay/simulated and are **not** treated as primary evidence:
- \`scripts/evaluation/eval_5_2.py\`
- \`experiments/validation/table2_gradient_flow.py\`
- \`experiments/validation/table3_rabitq_space_accuracy.py\`
- \`experiments/validation/table6_math.py\`
- \`experiments/validation/table7_synergy.py\`
EOF

echo
echo "============================================================"
echo "Verification completed."
echo "Manifest: $MANIFEST"
echo "Log:      $LOG_FILE"
echo "Report:   $OUT_DIR/P0_verification_report.md"
echo "============================================================"
