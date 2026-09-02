# Multi-Step Verification Value — Feasibility Audit (Phase 7)

Scope: DDXPlus **train** split, 45/49 cases reached a qualifying state (1/disease, balanced; 4 cases never accumulated >=2 non-UNKNOWN reports), noise ['0.2', '0.3'], 121 qualifying states, 490 verify-candidate labels (64 truly-wrong).

Objective: `J(a|H_t) = E[Brier(b_T, D*) + 0.03·N_future | H_t, a, π_heuristic]`, `Q_multi = -J`, `V_multi(i) = J(AskNew_best) - J(VerifyOld(i))`.

## 1. Label stability

- n actions = 611; mean J SE = 0.1105 (median 0.1056, p90 0.2336).
- |V_multi| mean = 0.1476 vs SE mean 0.1588; 17.3% candidates are >2 SE from zero.
- paired (common-random-number) sign-flip rate = 18.5% over 3920 rollouts.

## 2. One-step vs multi-step misalignment

- Pearson(V_Bayes_Δ, V_multi) = 0.033; sign agreement = 58.6%.
- top-1 verify target agreement = 47.9% over 121 states.

## 3. Heuristic predictive power

- Pearson(heuristic_util, V_multi) = 0.164; Pearson(error_prob, V_multi) = 0.109.
- top-1 agreement (heuristic #1 vs V_multi #1) = 33.1% over 121 states.

## 4. Oracle upper bound

- states with >=1 wrong report: 55.
- mean V_multi of best-wrong (oracle) = 0.1285 vs heuristic-#1 = 0.0103 vs one-step-#1 = -0.0775.
- oracle − heuristic gap = 0.1182.
- wrong candidates V_multi mean = 0.1394 vs correct candidates = -0.0676.

## 5. Questions vs Brier decomposition

- pooled V_multi = BrierΔ -0.0356 + 0.03·futureΔ -0.0049.
- wrong candidates: BrierΔ 0.1447 + 0.03·futureΔ -0.0053 (n=64).
- correct candidates: BrierΔ -0.0627 + 0.03·futureΔ -0.0049 (n=426).

## 6. RAG signal

- Pearson(retrieval_impact, V_multi) = -0.034 vs Pearson(error_prob, V_multi) = 0.109.
- OLS R²: error_prob only = 0.012, + retrieval_impact = 0.015 (Δ = 0.004, n=490).

## 7. Leakage

True disease D* and sampled answers enter ONLY the evaluation-side terminal metrics (terminal_brier / nll / top1 / wrong_to_correct). The continuation policy is the exact frozen ReliabilityAwareActionPolicy; no oracle_states, no learned gate, no true-state read.  Unit tests test_10 (deployable policy reads no true state) and test_11 (oracle labels do not feed policy) enforce this.

## 8. Go / No-Go

- precision@1 (wrong-report selection): V_multi = 33.1%, one-step = 17.4%, heuristic = 28.1%.
- V_multi − one-step = 15.7%; V_multi − heuristic = 5.0%.
- V_multi wrong mean = 0.1394 vs correct = -0.0676; fraction positive = 40.0%.
- NOTE: V_multi's selection gain is the *oracle* value (it uses true D* in the terminal rollouts); it is not deployable. The deployable question is item 9 (learnability). See GO_NO_GO.md for the two-part verdict.

## 9. Worth training

- OLS R²(heuristic_util only) = 0.027; R²([error_prob, retrieval_impact, heuristic_util]) = 0.034 (n=490).
