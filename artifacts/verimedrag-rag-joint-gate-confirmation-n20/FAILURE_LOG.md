# Failure Log — VeriMedRAG Phase 2C (Frozen-Config N=20 Confirmation)

**Status: run completed; three analysis bugs found and fixed; verdict No-Go (a
legitimate negative result, not a runtime failure).**

The full 35,280-dialogue run completed without error (see `full_run.log`,
exit 0, 9615 s). Three correctness bugs surfaced during analysis; all were
found before finalization, fixed, and the affected numbers recomputed. The
final verdict (No-Go) is a substantive scientific result, not a software
failure, and is reported in `GO_NO_GO.md` / `CONFIRMATION_REPORT.md`.

---

## Bug 1 — screening-overlap check compared against the wrong set

- **Symptom**: `screening_overlap_check.csv` reported overlap = 245 (i.e. the
  entire screen subset), which would have invalidated the "overlap = 0"
  requirement on the 735-case primary confirmation subset.
- **Cause**: `setup_manifests` computed `screen_ids & {case_id for c in full}`
  instead of `screen_ids & {case_id for c in primary}`. All 245 screen cases
  are (by construction) a subset of the full 980, so the check trivially
  returned 245.
- **Fix**: changed to intersect against the primary subset.
- **Result**: `screening_overlap_check.csv` now reports `245,735,980,0,1`
  (overlap = 0). No trajectories were re-run — the bug was in the *audit* path
  only.

## Bug 2 — candidate base-rate conflation of "correct" vs "unknown"

- **Symptom**: `is_actually_wrong = 0` conflated "the report was correct" with
  "the true state is UNKNOWN (not comparable)", which would have biased the
  candidate-misreport base rate and the triggered-precision enrichment.
- **Cause**: the candidate event logger only emitted `is_actually_wrong`, so
  non-comparable candidates (UNKNOWN true state) were silently treated as
  "not wrong".
- **Fix**: added an `is_comparable` field to candidate events; the trigger
  analysis now conditions base rate and triggered precision on
  `is_comparable == 1`. The first run (`bbs5atwsn`) was killed and relaunched
  (`bsv2u7n5b`) with the corrected logger.
- **Result**: `retrieval_trigger_analysis.csv` reports
  `n_comparable_candidates = 62446`, base rate = 7.7%.

## Bug 3 — pooled-level key collision in paired analysis (CRITICAL)

- **Symptom**: `summary_primary.csv` showed `joint_gate_w1.0` Top-1 *higher*
  than baseline at noise 0.2 (82.90% vs 82.31%), yet the pooled bootstrap
  reported Top-1 Δ = −0.00454 (negative). The two outputs contradicted each
  other, signalling a bug.
- **Cause**: `pair_rows` keyed paired rows by `(case_id, seed)` only. For the
  pooled level (0.2 + 0.3), the same `(case_id, seed)` appears once per noise
  level, so the later noise level (0.3) silently overwrote the earlier (0.2),
  keeping only noise-0.3 data in every pooled statistic.
- **Fix**: changed the analysis-unit key to the 3-tuple `(case_id, seed,
  noise_rate)` throughout `pair_rows`, `_case_sums`, `_case_obs`,
  `_disease_groups`, `mean_delta_bootstrap`, `paired_delta_rows`, and
  `trajectory_changed`.
- **Result**: pooled statistics are now internally consistent. Pooled Top-1 Δ
  = +0.00317 matches the summary-derived average of (+0.005896 @ 0.2,
  +0.000454 @ 0.3); pooled Brier Δ = −0.00306 matches (−0.006573 @ 0.2,
  +0.000452 @ 0.3).

---

## Substantive finding (not a bug): Phase-2B Brier improvement did not replicate

The Phase-2B screening (N=5) showed a Brier improvement for the joint gate.
At N=20 confirmation scale, the pooled Brier Δ is −0.00306 with 95% CI
[−0.00990, +0.00395] — **not statistically significant** (CI crosses 0). The
per-seed breakdown reveals the cause:

| seed | Brier Δ (total, primary, pooled 0.2/0.3) |
|------|-------------------------------------------|
| 2026 | −0.01282 |
| 2027 | −0.00431 |
| 2028 | +0.00795 |

The Phase-2B improvement was seed-2026-specific. This is the preregistered
No-Go condition ("Brier advantage gone / not significant"), not a code fault.
