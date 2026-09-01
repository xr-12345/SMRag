#!/bin/bash
# Prompt #7 train gate-event collection: 3 seeds x 980 train cases x 4 noise.
# One process per seed (internally parallel via --workers 12 --executor process).
set -u

PY="/Users/xr-12345/miniforge3/bin/python3"
ROOT="/Users/xr-12345/Desktop/SafeMedRAG"
OUT="$ROOT/artifacts/verimedrag-hypothesis-validation-2026/gate-evaluation/train-events"
LOG="$OUT/logs"
mkdir -p "$LOG"

SEEDS=(101 102 103)

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

pids=()
for seed in "${SEEDS[@]}"; do
  dir="$OUT/seed-$seed"
  log="$LOG/seed-$seed.log"
  (
    cd "$ROOT" || exit 1
    PYTHONPATH=src "$PY" -m powerful_medrag.cli analyze-gate-ddxplus \
      --model data/ddxplus/model-full.json \
      --patients data/ddxplus/release_train_patients.zip \
      --evidences data/ddxplus/release_evidences.json \
      --output-dir "$dir" \
      --cases-per-disease 20 \
      --max-questions 15 \
      --noise-rates 0 0.1 0.2 0.3 \
      --seed "$seed" \
      --sample-seed 101 \
      --workers 12 \
      --executor process \
      > "$log" 2>&1
    echo "DONE seed-$seed" >> "$log"
  ) &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "ALL_TRAIN_GATE_DONE fail=$fail"
