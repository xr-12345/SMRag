# Phase 4 — Failure Log

Per the Phase 4 spec, any *discovery that changes formal interpretation* of a
result must be recorded here (rather than silently fixed), with a regression
test and regeneration in a new directory.

## Summary

**No failures requiring remediation were recorded.**

### Step 1 (heuristic_verify_utility audit)
- Outcome: the negative Spearman correlation (−0.223 overall; −0.153 within-case
  demeaned) is **real**, not a bookkeeping artifact. No sign-convention, sort,
  raw-vs-rank, double-deduction, alignment, truncation, or realized-sign bug was
  found. Conclusion documented in `HEURISTIC_UTILITY_AUDIT.md`. Nothing to fix,
  therefore no regression test and no regeneration.

### Implementation-phase issues (resolved, non-interpretation-changing)
These were ordinary development fixes and do not change any formal interpretation:

1. `write_csv` hardcoded `FEATURE_CSV_COLUMNS` → added a `fieldnames=` parameter
   so the manifest (different schema) can be written. No data semantics changed.
2. `generate_validation.run()` referenced `model`/`retriever` not in scope →
   added them as explicit parameters. No data semantics changed.
3. Test 7 and Test 11 were initially over-strict (matched docstrings rather than
   real code references). Relaxed to target actual code references
   (`validation_features.csv`, `asknew_value`, `expected_information_gain`,
   `from .action_value import`). Test intent unchanged.

4. **Shuffled-RAG ablation was a no-op duplicate of `real`** (found before any
   final numbers were produced). `build_feature_matrix(rows, rag_mode="shuffled")`
   fell through to `return X` (the real matrix) instead of shuffling; combined
   with `evaluate.py`'s `"real" if rag_mode == "shuffled"` workaround, the
   shuffled model was trained *and* evaluated on the same real-RAG features as
   the real model — the ablation could not detect any RAG-specific signal.
   Fix: `build_feature_matrix` now returns `shuffle_rag_features(rows, seed=3031)`
   for `"shuffled"`, and `evaluate.py` passes the true `rag_mode` through.
   Regression test `test_08b_build_feature_matrix_shuffled_matches_shuffle` added.
   This changes the RAG-ablation *interpretation* (previously vacuous), so it is
   recorded here per the failure-log convention; it does not touch the audit
   conclusion, schema, labels, or split.

None of the above altered the audit conclusion, the feature schema, the label
definitions, or the split.
