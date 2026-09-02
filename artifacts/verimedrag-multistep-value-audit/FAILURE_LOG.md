# Phase 7 — Failure Log

No test failures and no trajectory-level failures across the smoke (10 cases,
20 dialogues) and the full audit (49 cases, 98 dialogues).  All ten smoke
integrity gates passed on the first run (see `smoke_checks.txt`).

## Harness notes (not failures)

1. **Coverage 45/49.**  Four of the 49 sampled cases never produced a
   qualifying state (no turn with both >=1 AskNew and >=2 non-UNKNOWN
   VerifyOld candidates) before the frozen heuristic stopped.  This is honest
   coverage, not a bug — recorded in `analysis.json` (`counts.n_cases`).

2. **Runtime.**  Full audit = 98 dialogues, 4,888 continuations, 1,598 s wall
   (single continuation 2.97 s, 12 workers, peak RSS 1.9 GB).  Consistent with
   the 10-case smoke estimate (single rollout 1.45 s).

## Red-line compliance

No test split touched (train only); no retraining; no online integration; no
oracle correction; no true state / latent state in the continuation policy
(the continuation is the exact frozen `ReliabilityAwareActionPolicy`); AskNew /
Stop thresholds unchanged; no LLM / dense retriever / SFT / RL; no git commit;
no prior artifacts overwritten (new directory).  `git status` confirms only new
untracked files; `git diff` (tracked) is empty.
