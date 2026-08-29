# Frozen experiment protocol and commands

The machine-readable success criteria are in
`configs/preregistered_success_criteria.json`. Do not change them after test evaluation.

## Current availability

- Unit tests and the toy pilot are runnable without external dependencies.
- DDXPlus aggregate artifacts are committed, but raw patient ZIPs and `model-full.json` are absent.
- Therefore the commands below are prepared but have not produced new DDXPlus results in this
  revision.

## Tests and synthetic pilot

```bash
PYTHONPATH=src python -m unittest discover -s tests -v

PYTHONPATH=src python -m powerful_medrag pilot-reliability \
  --output-dir artifacts/toy-reliability-pilot \
  --training-seed 101 --sample-seed 2026 \
  --seeds 2026 2027 2028 \
  --noise-rates 0 0.1 0.2 0.3 \
  --minimum-action-utility 0.08
```

The toy training and evaluation generators use different seeds. This is a code/trend check, not a
clinical or DDXPlus result.

## Learned gate: validation-only fitting

First collect observable events on the validation split:

```bash
PYTHONPATH=src python -m powerful_medrag analyze-gate-ddxplus \
  --model data/ddxplus/model-full.json \
  --patients data/ddxplus/release_validate_patients.zip \
  --evidences data/ddxplus/release_evidences.json \
  --output-dir artifacts/learned-gate-validation-events \
  --cases-per-disease 20 --seed 2026
```

Then use a case-grouped held-out portion for Platt calibration:

```bash
PYTHONPATH=src python -m powerful_medrag fit-gate \
  --events artifacts/learned-gate-validation-events/ddxplus_gate_events.csv \
  --output artifacts/learned-gate-validation-events/learned_gate.json \
  --target harmful_misreport \
  --calibration-fraction 0.2 --split-seed 2026
```

The fitted gate may be evaluated beside the oracle ceiling with `ablate-ddxplus --variants
adaptive_learned oracle_gate ... --learned-gate learned_gate.json`. The oracle must never be listed
as a deployable method.

## Joint-policy DDXPlus benchmark

After restoring the validation archive and model, tune only on validation:

```bash
PYTHONPATH=src python -m powerful_medrag benchmark-reliability-ddxplus \
  --model data/ddxplus/model-full.json \
  --patients data/ddxplus/release_validate_patients.zip \
  --evidences data/ddxplus/release_evidences.json \
  --output-dir artifacts/reliability-policy-validation-n20 \
  --cases-per-disease 20 \
  --seeds 2026 2027 2028 \
  --noise-rates 0 0.1 0.2 0.3 \
  --max-total-turns 15 \
  --minimum-action-utility 0.08
```

Freeze the configuration before replacing `release_validate_patients.zip` with a genuinely untouched
holdout or external set. The existing DDXPlus test split is not pristine over the full project
lifecycle and should not support a confirmatory novelty claim.

## Required reporting

Retain case-level outcomes. Report Top-1/Top-3, new questions, verification questions, total atomic
questions, interaction turns, Brier, ECE, unnecessary verification, mitigation, premature stopping,
multi-seed intervals, paired case statistics, and full accuracy--question curves. Red-flag and
under-triage metrics are permitted only after a source-backed label mapping is frozen.

