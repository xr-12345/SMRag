#!/bin/bash
# Phase 4 -- wait for validation, then train (single-threaded HGB) -> evaluate -> report.
set -u
cd "$(dirname "$0")"
ROOT=/Users/xr-12345/Desktop/SafeMedRAG
PY=/Users/xr-12345/miniforge3/bin/python3
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=1   # HGB single-thread: avoids OpenMP oversubscription thrash

echo "[pipeline] waiting for validation to finish ..."
until grep -q "verify rows" validation_full.log; do sleep 15; done
echo "[pipeline] validation done: $(tail -1 validation_full.log)"

echo "[pipeline] training (OMP_NUM_THREADS=1) ..."
"$PY" train_model.py > train_full.log 2>&1
ec=$?
echo "[pipeline] train exit $ec"
[ $ec -eq 0 ] || { echo "[pipeline] TRAIN FAILED"; cat train_full.log; exit $ec; }

echo "[pipeline] evaluating ..."
"$PY" evaluate.py > evaluate_full.log 2>&1
ec=$?
echo "[pipeline] evaluate exit $ec"
[ $ec -eq 0 ] || { echo "[pipeline] EVALUATE FAILED"; cat evaluate_full.log; exit $ec; }

echo "[pipeline] writing reports ..."
"$PY" write_reports.py > write_reports_full.log 2>&1
ec=$?
echo "[pipeline] report exit $ec"
[ $ec -eq 0 ] || { echo "[pipeline] REPORT FAILED"; cat write_reports_full.log; exit $ec; }

echo "[pipeline] COMPLETE"
ls -la models/ 2>/dev/null
