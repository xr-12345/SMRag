"""Activation-audit analysis (frozen primary point: threshold 0.85, 3 seeds).

Reads:
  - the 3 new CSVs under learned-gate-activation-audit-agent/seed-<SEED>/
    (strategies: joint_new_verify_stop, joint_learned_gate)
  - the frozen matched-budget t085 CSVs (for the baseline byte-identical check)

Produces (under learned-gate-activation-audit-agent/):
  case_outcomes.csv               pooled 3-seed outcomes (23,520 rows)
  summary_by_noise.csv            pooled per (strategy, noise) summary
  paired_deltas.csv               learned - baseline deltas by noise
  baseline_regression_check.txt   new joint_new_verify_stop == frozen t085 joint

Deterministic. Does not access the test split.
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict
from pathlib import Path
from statistics import fmean

REPO = Path("/Users/xr-12345/Desktop/SafeMedRAG")
AUDIT = REPO / "artifacts/verimedrag-hypothesis-validation-2026/learned-gate-activation-audit-agent"
FROZEN = REPO / "artifacts/verimedrag-hypothesis-validation-2026/matched-budget"

NOISE_RATES = (0.0, 0.1, 0.2, 0.3)
SEEDS = (2026, 2027, 2028)
BASELINE_STRATEGY = "joint_new_verify_stop"
LEARNED_STRATEGY = "joint_learned_gate"

# All outcome columns in order (for byte-identical regression and case_outcomes).
OUTCOME_COLUMNS = [
    "seed",
    "case_id",
    "strategy",
    "noise_rate",
    "true_diagnosis",
    "predicted_diagnosis",
    "correct_top1",
    "correct_top3",
    "confidence",
    "brier_score",
    "new_questions",
    "verification_questions",
    "total_atomic_questions",
    "interaction_turns",
    "unnecessary_verifications",
    "resolved_wrong_reports",
    "premature_stop",
    "uncertain_output",
    "stop_reason",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_new() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed in SEEDS:
        path = AUDIT / f"seed-{seed}" / "ddxplus_reliability_outcomes.csv"
        rows.extend(read_rows(path))
    return rows


def load_frozen_t085() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed in SEEDS:
        path = FROZEN / "reliability-sweep" / "t085" / f"seed-{seed}" / "ddxplus_reliability_outcomes.csv"
        rows.extend(read_rows(path))
    return rows


def _f(v: str) -> float:
    return float(v)


def _i(v: str) -> int:
    return int(v)


def merge_and_write_case_outcomes(rows: list[dict[str, str]]) -> None:
    out = AUDIT / "case_outcomes.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUTCOME_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote case_outcomes.csv ({len(rows)} rows)")


def completeness(rows: list[dict[str, str]]) -> None:
    expected = len(SEEDS) * 980  # 2940
    groups: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        groups[(r["strategy"], r["noise_rate"])] += 1
    bad = {k: v for k, v in sorted(groups.items()) if v != expected}
    n_strategy = len({r["strategy"] for r in rows})
    print(f"rows={len(rows)} expected={expected * 2 * len(NOISE_RATES)}")
    print(f"strategies seen: {sorted({r['strategy'] for r in rows})} (n={n_strategy})")
    print(f"groups != {expected}: {bad or 'none'}")


def pairing(rows: list[dict[str, str]]) -> None:
    """Baseline and learned must share the exact (case_id, seed, noise) set."""
    base_keys = {
        (r["case_id"], r["seed"], r["noise_rate"])
        for r in rows
        if r["strategy"] == BASELINE_STRATEGY
    }
    learned_keys = {
        (r["case_id"], r["seed"], r["noise_rate"])
        for r in rows
        if r["strategy"] == LEARNED_STRATEGY
    }
    print(f"baseline keys: {len(base_keys)}  learned keys: {len(learned_keys)}")
    print(f"only in baseline: {len(base_keys - learned_keys)}  only in learned: {len(learned_keys - base_keys)}")


def baseline_regression_check(new_rows: list[dict[str, str]]) -> None:
    frozen = {
        (r["case_id"], r["seed"], r["noise_rate"]): r
        for r in load_frozen_t085()
        if r["strategy"] == BASELINE_STRATEGY
    }
    mine = {
        (r["case_id"], r["seed"], r["noise_rate"]): r
        for r in new_rows
        if r["strategy"] == BASELINE_STRATEGY
    }
    keys = set(frozen) & set(mine)
    mismatched = 0
    first_mismatch = None
    for key in sorted(keys):
        f, m = frozen[key], mine[key]
        for col in OUTCOME_COLUMNS:
            if f[col] != m[col]:
                mismatched += 1
                first_mismatch = first_mismatch or (key, col, f[col], m[col])
                break
    lines = [
        f"frozen t085 joint rows: {len(frozen)}",
        f"new joint rows: {len(mine)}",
        f"shared (case_id, seed, noise) keys: {len(keys)}",
        f"byte-identical rows: {len(keys) - mismatched}/{len(keys)}",
        (
            "REGRESSION PASS: new joint_new_verify_stop is byte-identical to frozen t085."
            if mismatched == 0 and len(keys) == len(frozen) == len(mine)
            else f"REGRESSION MISMATCH: {mismatched} rows differ (first: {first_mismatch})."
        ),
    ]
    (AUDIT / "baseline_regression_check.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


def summary_by_noise(rows: list[dict[str, str]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        groups[(r["strategy"], r["noise_rate"])].append(r)
    out_rows: list[dict] = []
    for (strategy, noise), recs in sorted(groups.items()):
        verify = sum(_i(r["verification_questions"]) for r in recs)
        resolved = sum(_i(r["resolved_wrong_reports"]) for r in recs)
        unnecessary = sum(_i(r["unnecessary_verifications"]) for r in recs)
        premature = sum(_i(r["premature_stop"]) for r in recs)
        out_rows.append(
            {
                "strategy": strategy,
                "noise_rate": f"{float(noise):.1f}",
                "cases": len(recs),
                "top1_accuracy": fmean(_i(r["correct_top1"]) for r in recs),
                "brier_score": fmean(_f(r["brier_score"]) for r in recs),
                "avg_total_atomic_questions": fmean(
                    _i(r["total_atomic_questions"]) for r in recs
                ),
                "avg_new_questions": fmean(_i(r["new_questions"]) for r in recs),
                "avg_verification_questions": fmean(
                    _i(r["verification_questions"]) for r in recs
                ),
                "unnecessary_verification_rate": (
                    unnecessary / verify if verify else 0.0
                ),
                "conflict_resolution_rate": (resolved / verify if verify else 0.0),
                "premature_stop_rate": premature / len(recs),
            }
        )
    fieldnames = [
        "strategy",
        "noise_rate",
        "cases",
        "top1_accuracy",
        "brier_score",
        "avg_total_atomic_questions",
        "avg_new_questions",
        "avg_verification_questions",
        "unnecessary_verification_rate",
        "conflict_resolution_rate",
        "premature_stop_rate",
    ]
    out = AUDIT / "summary_by_noise.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: f"{r[k]:.6f}" if isinstance(r[k], float) else r[k] for k in fieldnames})
    print(f"wrote summary_by_noise.csv ({len(out_rows)} rows)")


def paired_deltas(rows: list[dict[str, str]]) -> None:
    base = {
        r["noise_rate"]: r
        for r in _summaries(rows)
        if r["strategy"] == BASELINE_STRATEGY
    }
    out_rows: list[dict] = []
    for r in _summaries(rows):
        if r["strategy"] != LEARNED_STRATEGY:
            continue
        b = base[r["noise_rate"]]
        out_rows.append(
            {
                "noise_rate": r["noise_rate"],
                "delta_top1": r["top1_accuracy"] - b["top1_accuracy"],
                "delta_brier": r["brier_score"] - b["brier_score"],
                "delta_total_questions": r["avg_total_atomic_questions"]
                - b["avg_total_atomic_questions"],
                "delta_verification_questions": r["avg_verification_questions"]
                - b["avg_verification_questions"],
                "delta_unnecessary_rate": r["unnecessary_verification_rate"]
                - b["unnecessary_verification_rate"],
                "delta_conflict_resolution_rate": r["conflict_resolution_rate"]
                - b["conflict_resolution_rate"],
                "delta_premature_stop_rate": r["premature_stop_rate"]
                - b["premature_stop_rate"],
            }
        )
    fieldnames = list(out_rows[0].keys())
    out = AUDIT / "paired_deltas.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(
                {k: (r[k] if isinstance(r[k], str) else f"{r[k]:.6f}") for k in fieldnames}
            )
    print(f"wrote paired_deltas.csv ({len(out_rows)} rows)")


def _summaries(rows: list[dict[str, str]]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        groups[(r["strategy"], r["noise_rate"])].append(r)
    out: list[dict] = []
    for (strategy, noise), recs in sorted(groups.items()):
        verify = sum(_i(r["verification_questions"]) for r in recs)
        resolved = sum(_i(r["resolved_wrong_reports"]) for r in recs)
        unnecessary = sum(_i(r["unnecessary_verifications"]) for r in recs)
        premature = sum(_i(r["premature_stop"]) for r in recs)
        out.append(
            {
                "strategy": strategy,
                "noise_rate": noise,
                "top1_accuracy": fmean(_i(r["correct_top1"]) for r in recs),
                "brier_score": fmean(_f(r["brier_score"]) for r in recs),
                "avg_total_atomic_questions": fmean(
                    _i(r["total_atomic_questions"]) for r in recs
                ),
                "avg_verification_questions": fmean(
                    _i(r["verification_questions"]) for r in recs
                ),
                "unnecessary_verification_rate": (
                    unnecessary / verify if verify else 0.0
                ),
                "conflict_resolution_rate": (resolved / verify if verify else 0.0),
                "premature_stop_rate": premature / len(recs),
            }
        )
    return out


def main() -> None:
    rows = load_new()
    completeness(rows)
    pairing(rows)
    baseline_regression_check(rows)
    merge_and_write_case_outcomes(rows)
    summary_by_noise(rows)
    paired_deltas(rows)
    print("ACTIVATION_AUDIT_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
