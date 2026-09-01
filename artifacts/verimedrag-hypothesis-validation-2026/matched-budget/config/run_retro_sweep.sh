#!/bin/bash
# Retro multi-threshold curve: retro_utility_u050_b1 x 6 thresholds x 3 seeds.
# One process per seed (internally parallel via --workers).
set -u

PY="/Users/xr-12345/miniforge3/bin/python3"
ROOT="/Users/xr-12345/Desktop/SafeMedRAG"
OUT="$ROOT/artifacts/verimedrag-hypothesis-validation-2026/matched-budget/retro-sweep"
LOG="$OUT/logs"
mkdir -p "$LOG"

SEEDS=(2026 2027 2028)
THRESHOLDS="0.60 0.70 0.80 0.85 0.90 0.95"

pids=()
for seed in "${SEEDS[@]}"; do
  dir="$OUT/seed-$seed"
  log="$LOG/seed-$seed.log"
  (
    cd "$ROOT" || exit 1
    PYTHONPATH=src "$PY" -m powerful_medrag clarify-ddxplus \
      --model data/ddxplus/model-full.json \
      --patients data/ddxplus/release_validate_patients.zip \
      --evidences data/ddxplus/release_evidences.json \
      --output-dir "$dir" \
      --cases-per-disease 20 --sample-seed 2026 --max-questions 15 \
      --noise-rates 0.0 0.1 0.2 0.3 \
      --thresholds $THRESHOLDS \
      --variants retro_utility_u050_b1 \
      --seed "$seed" --workers 6 --executor process \
      > "$log" 2>&1
    echo "DONE seed-$seed" >> "$log"
  ) &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "ALL_RETRO_DONE fail=$fail"
