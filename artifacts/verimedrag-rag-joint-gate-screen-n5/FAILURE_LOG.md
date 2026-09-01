# Failure Log — VeriMedRAG Phase 2B (N=5 Joint Gate Screening)

**Status: no failures.**

This log records any run-time failure, unexpected crash, or assertion during the
Phase 2B screening and analysis. It is populated only if something actually
fails; the convention for a clean run is to keep it empty.

- Cost audit: completed without error (see `retrieval_cost_profile.csv`).
- Full screening: 3430 dialogues over 12 workers, completed without error
  (see `case_outcomes.csv`, `turn_observations.csv`).
- Analysis: completed without error (see `summary_by_config_noise.csv`,
  `paired_deltas.csv`, `screening_metrics.json`).
- Unit tests: 81/81 OK (see `TEST_LOG.txt`).

If a subsequent run surfaces a failure, record it here with: the failing
command, the full traceback / assertion, the expected vs actual value, and
whether it was resolved.
