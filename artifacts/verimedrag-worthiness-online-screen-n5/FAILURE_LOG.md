# Phase 5 — Failure / Issue Log

Issues encountered while wiring the frozen learned worthiness model into the
online dialogue and running the N=5 screen.  No red line was crossed and no
retraining / model change / tau_harm change occurred.

## 1. Runner reimplemented `_pilot_seed` locally (fixed)

The first draft of `run_screen.py` re-implemented `_pilot_seed` with a local
`hashlib.blake2b` call.  Although byte-identical to the reference, it duplicated
the frozen-seeding logic and risked drift.  Replaced with a direct import from
`powerful_medrag.reliability_experiment` so the patient seeds are guaranteed to
match the Phase 2B / Phase 2C runs.

## 2. Brier-unit strategies stop earlier than the heuristic (finding, not a bug)

The smoke run revealed that `model_based_vbayes_verify` and the two `learned_*`
strategies stop after 0 questions on some cases (e.g. case 80212), while the
heuristic asks ~10.  Root cause: the unified-Brier comparison values AskNew by
the deployable `V_Bayes_new = R(b_t) - E[R(b_{t+1})] - C_new`; on a diffuse
prior the best question's expected Brier-risk reduction is below `C_new = 0.03`,
so `V_Bayes_new < 0` and `choose_action` correctly stops (no positive-utility
action).  This is the intended, spec-mandated behaviour (EIG nats vs Brier value
must not be compared directly), and it is reported as a primary finding: the
Brier-unit integration asks far fewer questions than the EIG-based heuristic.

## 3. Missing realized correct→wrong capture (fixed, run relaunched)

The first full-run launch did not record per-verification top-1 flips, so report
item #3 (does the harm gate reduce correct→wrong) could not be computed online.
Stopped that run, added per-turn `turn_flip` tracking (pre/post top-1 vs the true
disease from `result.turns[].belief`), added `correct_to_wrong_flips` /
`wrong_to_correct_flips` to `case_outcomes.csv` and `realized_correct_to_wrong` /
`realized_wrong_to_correct` to `verification_candidates.csv`, then relaunched.

## 4. Git state surprise (no action needed)

The working tree already contained uncommitted Phase 3A / Phase 4 files
(`action_value.py`, `verification_worthiness.py` and their tests/artifacts) plus
the Phase 5 files; Phase 1–2C had been committed as `91268d7`.  Phase 5 adds
only two new files (`worthiness_policy.py`, `test_worthiness_policy.py`) and this
artifact directory.  Nothing was committed by this phase (red line honoured).
