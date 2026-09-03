# Phase 8A — Unified Brier Action Value + Pre-stop Verification Audit

Implements a new, independent strategy `UnifiedBrierReliabilityAuditPolicy` that
fixes the two theoretical gaps in the existing `_UnifiedBrierPolicy`.  No formal
N=20 experiment, no DDXPlus test split, no retraining, no commit.

## What changed

| File | Change |
| --- | --- |
| `src/powerful_medrag/decision.py` | `ReliabilityAwarePolicyConfig` gains `verification_audit_threshold: float = 0.03` (validated `>= 0`). Old policies never read it, so default behaviour is unchanged. |
| `src/powerful_medrag/worthiness_policy.py` | New `UnifiedBrierReliabilityAuditPolicy` (subclass of `ReliabilityAwareActionPolicy`), a new `WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT`, and a `build_policy` branch. |
| `tests/test_unified_brier_audit.py` | 16 new tests (see below). |

## The two fixes

### 1. AskNew on one Brier scale

Old behaviour (`_UnifiedBrierPolicy.rank_actions`) selected a single question by
EIG, then valued only that one question in Brier units.  New behaviour enumerates
every still-unasked question and values each by

```
V_new(j) = R_B(b_t) - E_y[R_B(b_{t+1}^{j,y})] - C_new,j
C_new,j  = new_question_cost_weight * spec.cost_j
j_B*     = argmax_j V_new(j)
```

`R_B(b) = 1 - sum_d b(d)^2` (the shared `brier_risk`).  The EIG-best and
Brier-best keys are both recorded (`last_eig_best_key` / `last_brier_best_key`
and per-question `is_eig_best` / `is_brier_best` in `last_asknew_log`) so their
agreement is observable.  Note the old `_UnifiedBrierPolicy` used a fixed
`C_NEW = 0.03`; the new policy uses the per-question cost as specified.

### 2. Explicit VerifyOld audit before Stop

Old behaviour stopped on the inherited `choose_action` confidence/utility/
reliability/safety checks with no VerifyOld gross-value check.  New behaviour
values every unverified, non-UNKNOWN report by `verify_value` (net `V_verify`)
and its gross risk reduction

```
G_verify(i) = V_verify(i) + C_verify
G_t^verify  = max_i G_verify(i)
```

Before returning Stop, if `G_t^verify >= verification_audit_threshold` the policy
returns `VerifyOld(argmax_i G_verify)` instead (bounded by `maximum_verifications`).
This is the confirmed stop rule from §四.

## Red lines honoured

- Default policy remains `ReliabilityAwareActionPolicy` (heuristic) — new strategy
  is opt-in via `unified_brier_reliability_audit`.
- Action value reads only the belief, the fitted model, and the frozen answer
  channel.  No `patient.latent_states`, true disease, noise label, or true
  wrongness enters the prediction path (`rank_actions`/`choose_action` `del
  oracle_states`; source check test_12 + existing test_08 both pass).
- Oracle path untouched; no oracle code added.
- RAG is kept (retriever/retrieval_mode still drive retrieval logging and the
  `suspicious` reliability check), but retrieval Jaccard never enters the Brier
  value (`verify_value` is retrieval-free).
- No old strategy / test / artifact deleted or overwritten; new artifact dir only.
- No git commit.

## Tests

`tests/test_unified_brier_audit.py` — 16 tests covering: AskNew full enumeration
(01), per-question cost + Brier-best selection (02), EIG/Brier-best recording
(03), VerifyOld enumeration incl. UNKNOWN skip (04/05), gross-vs-net + audit state
(06), audit threshold semantics (07/08), pre-stop audit blocks/does-not-block Stop
(09/10), `maximum_verifications` bound (11), no-true-state (12), RAG-free Brier
value (13), default-stays-heuristic + explicit enable (14), config validation (15),
and a full dialogue smoke (16).

Full suite: **163 tests, OK** (147 prior + 16 new).  `TEST_LOG.txt` in this dir.
