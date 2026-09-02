# Phase 6 — Failure Log

No test failures and no trajectory-level failures.  Two smoke-check *harness*
bugs were found and fixed during the 49-case smoke (they were false-FAILs in the
checking code, not in the drop-in policy or the data).

## 1. `heuristic_regression` false-FAIL (fixed)

The first smoke run reported `[FAIL] heuristic_regression: 0 mismatches of 49
cases (phase5 overlap 245)`.  The comparison itself found **0 mismatches** — the
baseline was byte-identical to Phase 5's `heuristic_verify` on all 49 cases.
The FAIL came from a wrong arity check: `len(p5) == len(base)` compared the full
245-case Phase 5 table against the 49-case smoke subset.  Fixed to
`0 mismatches AND no missing case_ids` (now PASS).

## 2. `no_new_premature_stop` misinterpretation (fixed)

The first smoke run also FAILed a check that treated the *downstream realized*
premature-stop rate (`predicted != diagnosis and confident stop`) as an
isolation invariant.  It is not: re-ranking the VerifyOld target legitimately
changes the belief and thus the final diagnosis, so the premature-stop rate can
differ on a 49-case sample without any isolation violation.  The true isolation
invariant is `extra_stop == 0` (the learned model never opens a Stop the
controller would not) — captured by `no_extra_stop`, which PASSed.  The
premature-stop rate is now reported as an informational line only.

## 3. Status of the N=5 screen

Completed cleanly: 5,880 trajectories in 1,845 s (mean wall 4.96 s/dialogue,
16 workers, peak RSS 2.1 GB).  No trajectory-level failures.  Verdict: **No-Go**
(4/6 pre-registered criteria pass) — see `GO_NO_GO.md` and `DROPIN_REPORT.md`.

## Red-line compliance

No test split touched; no retraining; no tau_harm / feature / model change; no
AskNew EIG or Stop threshold change; learned never opens a VerifyOld; no
oracle/true-state in any prediction path; default strategy stays
`heuristic_baseline`; no git commit.
