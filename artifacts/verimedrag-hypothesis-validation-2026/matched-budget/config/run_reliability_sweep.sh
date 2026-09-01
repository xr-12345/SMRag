#!/bin/bash
# Formal reliability sweep: 4 strategies x 6 thresholds x 3 seeds = 18 processes.
# One process per (threshold, seed); each writes its own case-level CSV.
set -u

PY="/Users/xr-12345/miniforge3/bin/python3"
ROOT="/Users/xr-12345/Desktop/SafeMedRAG"
OUT="$ROOT/artifacts/verimedrag-hypothesis-validation-2026/matched-budget/reliability-sweep"
LOG="$OUT/logs"
mkdir -p "$LOG"

# thresholds: label -> value
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
        --strategies full_two_layer joint_new_verify_stop oracle_verify oracle_select_same_channel \
        --policy-posterior-threshold "$thresh" \
        > "$log" 2>&1
      echo "DONE $label seed-$seed" >> "$log"
    ) &
    pids+=($!)
  done
done

# Wait for all children; report any non-zero exit.
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "ALL_RELIABILITY_DONE fail=$fail"
