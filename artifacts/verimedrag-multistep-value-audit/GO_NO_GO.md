# Phase 7 — Multi-Step Verification Value: Go / No-Go

## A. Is multi-step value more reasonable than one-step Brier value?

- wrong vs correct V_multi: +0.1394 vs -0.0676 (right sign).
- one-step V_Bayes top pick realized V_multi = -0.0775 (< 0 = counterproductive); heuristic = +0.0103; oracle best-wrong = +0.1285.
- precision@1: V_multi 0.331 vs one-step 0.174 vs heuristic 0.281.
- one-step vs multi-step: Pearson 0.033, sign agreement 0.586.
**A verdict: YES** — multi-step value is more reasonable than one-step Brier value.

## B. Is it stable and learnable enough to deploy (train)?

- label stability: 17% significant (SE 0.159 vs |V| 0.148), paired sign-flip 18.5%.
- learnability: OLS R²(full features) = 0.034 (heuristic-only 0.027).
- RAG adds ΔR² = 0.004.
**B verdict: NO** — the label is not stable and learnable from label-free features.

## Overall: **NO-GO** (A=YES, B=NO)

Coverage: 45/49 cases reached a qualifying state (121 states, 490 verify labels, 64 truly-wrong).
