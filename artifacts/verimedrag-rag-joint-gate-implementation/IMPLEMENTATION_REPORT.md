# VeriMedRAG Phase 2A — Retrieval Joint Gate: Implementation Report

## 1. Is the implementation complete?

**Yes.** The `retrieval_joint_gate` mode is implemented end-to-end in
`src/powerful_medrag/decision.py` and `src/powerful_medrag/clarification.py`:

- New `RetrievalGateMode` enum — `RANK_ONLY` (default) / `JOINT_GATE`.
- New config fields `retrieval_gate_mode`, plus the pre-existing
  `retrieval_impact_weight`, `minimum_action_utility`, `verification_cost`.
- `_verification_scores` now fills `RetrospectiveScore.retrieval_impact`
  (the Phase-1 bug of folding impact into `score.score` is removed).
- New `_retrieval_activation_value(error_probability, normalized_impact)`.
- New `_verification_utility(score) -> (existing, final)`.
- `rank_actions` gains the joint-gate branch: only when the history gate is
  closed, `retrieval_gate_mode is JOINT_GATE`, retriever non-None, and mode
  `!= NO_RAG`, it computes the activation values and — only if
  `max(activation_value) >= minimum_action_utility` — sets
  `verification_gate_open=True` and `retrieval_opened_gate=True`.
- Opened VerifyOld candidates are tagged `retrieval_triggered=True` and still
  compete against the best AskNew / Stop via the unified `choose_action`
  comparison; verification is never forced.
- Per-candidate logging (`self.last_verification_log`, 12 label-free fields),
  and `PolicyTurn.verification_triggered_by_retrieval` is propagated through
  `run_reliability_aware_dialogue`.

## 2. Test pass count

**81 / 81 tests pass** (`Ran 81 tests in 0.115s — OK`).

- 10 new gate tests in `tests/test_retrieval_gate.py`.
- 1 updated assertion in `tests/test_retrieval.py`
  (`test_retrieval_impact_enters_verification_scores` now asserts the impact
  lives in `retrieval_impact` and re-ranks the VerifyOld utility, not `score.score`).
- All pre-existing tests (71) continue to pass.

## 3. Is the old path byte-identical (no regression)?

**Yes.**

- `retrieval_gate_mode` defaults to `RANK_ONLY`, and `retrieval_impact_weight`
  defaults to `0.0`, so the default path matches pre-change behaviour.
- `retrieval_mode=NO_RAG` (the default) is exercised by
  `test_retrieval_gate.py::test_no_rag_degenerates_to_baseline`, which asserts
  the joint-gate config with `retriever=None` produces **exactly** the same
  utilities as the plain `ReliabilityAwareActionPolicy()` baseline.
- `test_retrieval_gate.py::test_rank_only_keeps_gate_closed_and_reranks`
  verifies that in `RANK_ONLY` the gate is never opened by retrieval
  (`opened_by_retrieval` all False) and that retrieval only re-ranks an
  already-history-allowed VerifyOld candidate.
- Fixing the Phase-1 bug intentionally *removes* a side effect: impact no
  longer inflates the `suspicious` ceiling in `choose_action`; with the default
  weight `0.0` the STOP reliability logic is unchanged from `no_rag`.

## 4. Does the joint gate open as expected?

**Yes.** In the constructed "high-risk + high retrieval-impact" toy case
(`runny_nose` + `itchy_eyes` present, then `fever=present`):

- `fever` error probability = `0.268718`, normalized retrieval impact = `1.0`
  (deleting `fever` wipes the corpus hits).
- activation value = `0.268718 * 1.0 - 0.03 = 0.238718 >= 0.03`, so the gate
  opens and `fever` is offered as `VerifyOld` with `retrieval_triggered=True`
  (best action `verify`, `best_report_index=2`).
- The converse cases are also covered: high impact + low error probability
  stays below the bar; high error probability + low impact does not open;
  activation below `minimum_action_utility` does not open; and when AskNew
  utility (`0.202897`) exceeds the opened VerifyOld utility, the system still
  chooses `AskNew` — the gate opens but never forces verification.

## 5. Is there any leakage?

**No.**

- The activation value uses only the label-free retrospective error
  probability (`clarification._report_error_probability`) and
  `normalized_retrieval_impact` (a `1 - jaccard` counterfactual retrieval
  difference). `true_wrongness`, `latent_state`, `noise_type`, and the true
  disease never enter the prediction path.
- `test_prediction_path_reads_no_latent_state` runs `rank_actions` with no
  oracle state and still opens the gate, and asserts the per-candidate log
  carries exactly the 12 label-free fields (no latent/trueness keys).
- The logging schema is: `report_index`, `evidence_code`, `error_probability`,
  `raw_retrieval_impact`, `normalized_retrieval_impact`,
  `retrieval_activation_value`, `existing_verification_utility`,
  `final_verification_utility`, `old_history_gate_open`, `opened_by_retrieval`,
  `chosen_action`, `best_asknew_utility`.
- `raw_retrieval_impact = jaccard` (retrieval overlap) and
  `normalized_retrieval_impact = 1 - jaccard` (difference / impact) are
  complements; only `normalized_retrieval_impact` feeds the gate and utility.

## 6. Is it ready for Phase 2B screening?

**Yes.** The implementation is complete, the full suite is green, the old path
is regression-free, the joint gate opens only on the intended signal and never
forces verification, and there is no label leakage. The remaining Phase 2B work
(e.g. per-turn cost of the per-report delete counterfactual retrieval, top-k
pre-filtering, and the DDXPlus N=5/N=20 screen) can proceed. Per the red
lines, **no DDXPlus N=5/N=20 was run, no test split was touched, no old
artifact was overwritten, and nothing was committed** — this phase is
implementation + unit tests + toy smoke only.

---

### Products (this directory)

| file | contents |
|------|----------|
| `IMPLEMENTATION_REPORT.md` | this report (answers the six questions) |
| `CONFIG.json` | config surface + formulas + defaults |
| `TEST_LOG.txt` | full `unittest` output (81/81 OK) |
| `smoke_metrics.json` | toy smoke results (all checks passed) |
| `FAILURE_LOG.md` | empty (no failures) |
| `git_diff.txt` | working-tree diff (tracked + new files) |
