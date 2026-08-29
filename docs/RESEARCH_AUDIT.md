# Reliability-Aware Active Medical Interviewing: repository audit

Audit date: 2026-08-29

## Scope and reproducibility status

The repository is a working structured-dialogue research prototype, not a clinical system.  The
audit was performed against commit `e7b49bd` with a clean worktree.  All 35 unit tests pass with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The dependency-free toy experiment is reproducible with:

```bash
PYTHONPATH=src python -m powerful_medrag demo --seed 2026
```

For seed 2026 it predicts `influenza` after five additional questions in the displayed example;
the 90-case comparison reports EIG and the prevalence proxy at 0.844 accuracy, with 4.47 versus
4.96 average questions.  This is only a smoke test, not a paper result.

The committed DDXPlus aggregate CSVs, figures, and reports can be inspected.  End-to-end DDXPlus
reproduction is currently blocked because the ignored patient archives
`release_{train,validate,test}_patients.zip` and the reproducible fitted model
`data/ddxplus/model-full.json` are absent.  No new DDXPlus result should be claimed until those
inputs are restored and case-level outputs are regenerated.

## What is implemented

| Requirement | Evidence in the repository | Status |
|---|---|---|
| Explicit unknown distinct from negative | `schema.UNKNOWN`, channel likelihoods, belief and estimation tests | Implemented |
| Context-sensitive feature identity | `FeatureKey` includes sorted context pairs; conflict check requires equal keys | Implemented at structured-variable level |
| Latent clinical state and report mode | `DiseaseStateModel`, `AnswerChannel`, `BeliefTracker` | Implemented |
| Soft update over report modes | `AnswerChannel.marginal_probability` and `BeliefTracker.update` | Implemented |
| Repeated reports share one latent state | Per-disease feature-state posterior in `BeliefTracker` | Implemented and tested |
| Disease EIG question selection | reference and NumPy selectors in `questioning.py` | Implemented and tested |
| MedRAG reciprocal-degree comparator | `MedRAGReciprocalDegreeSelector` | Formula-level structured adaptation only |
| Feature-indexed paired answer noise | `StructuredPatientSimulator.answer` | Implemented and tested |
| Fixed report-layer ablations | `ablation.py` | Implemented |
| Heuristic, sparse, and history reliability gates | `gating.py` | Implemented |
| Gate diagnostics | `gate_analysis.py` computes AUROC/AP and activation metrics | Implemented, but lacks ECE/Brier |
| PaMis-style online clarification | `SurprisalClarificationProtocol` | Mechanism proxy only |
| Retrospective selective clarification | risk, risk-by-influence, and hybrid variants in `clarification.py` | Implemented as a budgeted audit at stopping/budget boundaries |
| Paired confidence intervals and tests | Wilson, case-cluster bootstrap, exact McNemar | Implemented |
| Equal-turn comparison and Pareto checks | `curve_analysis.py` | Implemented |

## What is only partial or absent

| Requested capability | Audit finding |
|---|---|
| Rich natural-language observation record | `Observation` has only key, value, certainty, and raw text. Extraction confidence, source, duration, severity, evidence span, and explicit conflict metadata are absent. Time/activity/location can be encoded only through `FeatureKey.context`. |
| Joint belief over all `D, Z, M_1:t` | Disease and per-feature latent-state beliefs are maintained; report-mode posteriors are returned per update but no full persistent joint trajectory is represented. |
| Learned and calibrated gate | Absent. There is no logistic/MLP gate, fit/validation split, serialization, ECE, or calibration procedure. |
| Oracle gate | Absent. |
| Independent-evidence gate input | Absent as an explicit feature. Current-history gating deliberately avoids current-answer compatibility to reduce confirmation bias. |
| Unified `new / verify / stop` action utility | Absent. New-question EIG and clarification protocols are separate control flows; verification is not ranked directly against new questions. |
| Reliability-aware stopping | Partial. Existing stopping uses posterior, entropy, minimum question utility, budget, or no questions. It does not test dependence on suspicious evidence. |
| Red flags, triage, and unsafe-stop loss | Absent by explicit project decision. No clinical safety claim is valid. |
| Random-question baseline | Absent. |
| Top-k, ECE, early-stop, under-triage metrics | Absent. Brier, top-1, turns, clarification diagnostics, and AUROC/AP are present. |
| Atomic question/new-versus-verify accounting | Clarification results separate primary and clarification counts, but the general benchmark schema does not expose all requested cost measures. |
| Fine-grained contradiction/missing/context stress tests | Only generic latent answer modes and structured contextual keys exist. DDXPlus does not establish these as real patient phenomena. |
| Non-accusatory natural-language clarification generation | Absent; current simulator repeats the same structured evidence question. |

## Existing results that may be cited with qualifications

- The committed reports support EIG over this repository's MedRAG-RDC formula adaptation, not
  over a complete MedRAG multi-turn implementation.
- The fixed report-layer ablation supports treating unknown/uncertain separately from negative.
- The history-only gate improves the committed high-noise test Pareto position, but the test split
  was used earlier in the project and is not a pristine final holdout across the project lifecycle.
- The PaMis-style result is a surprisal-triggered structured proxy.  It does not implement PaMis's
  dialogue entity graph, structural entropy, or natural-language generator.
- Retrospective utility clarification was selected on the DDXPlus validation split and has not been
  run as a frozen method on a new untouched test set.

## Immediate engineering order

1. Preserve the current behavior as named baselines and add missing metric/accounting tests.
2. Add a configurable observable-feature record and gate diagnostics without changing old JSON.
3. Implement learned and oracle gates behind explicit experimental names.
4. Introduce a common action-score interface that compares new questions, one-time verification,
   and stopping; retain current clarification implementations as baselines.
5. Add reliability-aware stopping before adding any red-flag claim.  Safety evaluation requires an
   explicit external label source and must not be fabricated from DDXPlus.
6. Restore DDXPlus archives/model, freeze configuration, run a small validation pilot, then run
   multiple seeds while retaining case-level outputs.

