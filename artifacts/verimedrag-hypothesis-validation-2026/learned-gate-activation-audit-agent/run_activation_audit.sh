#!/bin/bash
# Learned-gate activation audit — frozen primary point (threshold 0.85, 3 seeds).
# 3 seeds run in parallel; each writes to seed-<SEED>/ and a per-seed log.
set -u
cd /Users/xr-12345/Desktop/SafeMedRAG
PY=/Users/xr-12345/miniforge3/bin/python3
BASE=artifacts/verimedrag-hypothesis-validation-2026/learned-gate-activation-audit-agent
GATE=artifacts/verimedrag-hypothesis-validation-2026/gate-evaluation/model/learned_gate.json
mkdir -p "$BASE"

for seed in 2026 2027 2028; do
  (
    PYTHONPATH=src "$PY" -m powerful_medrag benchmark-reliability-ddxplus \
      --model data/ddxplus/model-full.json \
      --patients data/ddxplus/release_validate_patients.zip \
      --evidences data/ddxplus/release_evidences.json \
      --output-dir "$BASE/seed-$seed" \
      --cases-per-disease 20 \
      --sample-seed 2026 \
      --seeds "$seed" \
      --noise-rates 0 0.1 0.2 0.3 \
      --max-total-turns 15 \
      --strategies joint_new_verify_stop joint_learned_gate \
      --learned-verification-gate "$GATE" \
      --policy-posterior-threshold 0.85 \
      --policy-margin-threshold 0.70 \
      --minimum-action-utility 0.08 \
      --verification-cost 0.03 \
      --minimum-history-cues 1 \
      > "$BASE/seed-$seed.log" 2>&1
    echo "seed-$seed exit=$?" >> "$BASE/run_status.txt"
  ) &
done
wait
echo "ALL_ACTIVATION_AUDIT_DONE"
