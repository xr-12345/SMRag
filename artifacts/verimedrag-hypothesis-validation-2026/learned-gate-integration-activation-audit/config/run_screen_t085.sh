#!/bin/bash
set -u

PY="/Users/xr-12345/miniforge3/bin/python3"
ROOT="/Users/xr-12345/Desktop/SafeMedRAG"
OUT="$ROOT/artifacts/verimedrag-hypothesis-validation-2026/learned-gate-integration-activation-audit/screen-t085"
GATE="$ROOT/artifacts/verimedrag-hypothesis-validation-2026/gate-evaluation/model/learned_gate.json"

mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

pids=()
for seed in 2026 2027 2028; do
  run_dir="$OUT/seed-$seed"
  log="$OUT/logs/seed-$seed.log"
  (
    cd "$ROOT" || exit 1
    PYTHONPATH=src "$PY" -m powerful_medrag benchmark-reliability-ddxplus \
      --model data/ddxplus/model-full.json \
      --patients data/ddxplus/release_validate_patients.zip \
      --evidences data/ddxplus/release_evidences.json \
      --output-dir "$run_dir" \
      --cases-per-disease 20 --sample-seed 2026 --seeds "$seed" \
      --noise-rates 0.0 0.1 0.2 0.3 --max-total-turns 15 \
      --strategies joint_new_verify_stop joint_learned_gate \
      --learned-verification-gate "$GATE" \
      --policy-posterior-threshold 0.85 --policy-margin-threshold 0.70 \
      --minimum-action-utility 0.08 --verification-cost 0.03 \
      --minimum-history-cues 1 > "$log" 2>&1
    status=$?
    echo "DONE seed-$seed status=$status" >> "$log"
    exit "$status"
  ) &
  pids+=("$!")
done

failure=0
for pid in "${pids[@]}"; do
  wait "$pid" || failure=1
done
echo "ALL_DONE failure=$failure"
exit "$failure"
