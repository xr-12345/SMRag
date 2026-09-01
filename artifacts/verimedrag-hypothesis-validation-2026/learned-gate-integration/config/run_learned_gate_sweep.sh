#!/bin/bash
# Prompt #8 formal sweep: joint_new_verify_stop + joint_learned_gate,
# 6 thresholds x 3 seeds = 18 processes. One process per (threshold, seed).
set -u

PY="/Users/xr-12345/miniforge3/bin/python3"
ROOT="/Users/xr-12345/Desktop/SafeMedRAG"
GATE="$ROOT/artifacts/verimedrag-hypothesis-validation-2026/gate-evaluation/model/learned_gate.json"
OUT="$ROOT/artifacts/verimedrag-hypothesis-validation-2026/learned-gate-integration/reliability-sweep"
LOG="$OUT/logs"
mkdir -p "$LOG"

declare -a LABELS=(t060 t070 t080 t085 t090 t095)
declare -a VALUES=(0.60 0.70 0.80 0.85 0.90 0.95)
declare -a SEEDS=(2026 2027 2028)

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

pids=()
for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  thresh="${VALUES[$i]}"
  for seed in "${SEEDS[@]}"; do
    dir="$OUT/$label/seed-$seed"
    log="$LOG/${label}_seed-${seed}.log"
    (
      cd "$ROOT" || exit 1
      PYTHONPATH=src "$PY" -m powerful_medrag benchmark-reliability-ddxplus \
        --model data/ddxplus/model-full.json \
        --patients data/ddxplus/release_validate_patients.zip \
        --evidences data/ddxplus/release_evidences.json \
        --output-dir "$dir" \
        --cases-per-disease 20 --sample-seed 2026 --seeds "$seed" \
        --noise-rates 0.0 0.1 0.2 0.3 --max-total-turns 15 \
        --strategies joint_new_verify_stop joint_learned_gate \
        --learned-verification-gate "$GATE" \
        --policy-posterior-threshold "$thresh" \
        > "$log" 2>&1
      echo "DONE $label seed-$seed" >> "$log"
    ) &
    pids+=($!)
  done
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "ALL_LEARNED_GATE_DONE fail=$fail"
