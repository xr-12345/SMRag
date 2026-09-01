"""Prompt #8 learned-gate integration analysis.

Reads:
  - the 18 new CSVs (joint_new_verify_stop + joint_learned_gate, 6 thresholds x
    3 seeds) under learned-gate-integration/reliability-sweep/
  - the frozen matched-budget 18 CSVs (4 strategies) under
    matched-budget/reliability-sweep/

Produces (under learned-gate-integration/analysis/):
  regression_check.txt                        re-run joint == frozen joint
  summary_by_strategy_noise_threshold.csv     pooled per-group summary
  learned_vs_baseline_deltas.csv              Q1/Q2/Q3 headline deltas
  curve_points.csv                            case-clustered accuracy vs questions
  pareto_frontier_combined.csv                frozen 4 strategies + learned gate
  budget_matches_learned_vs_full_two_layer.csv  Q4 matched-budget
  accuracy_questions_curves.png

Deterministic; bootstrap seeded at 2026. Does not access the test split.
"""

from __future__ import annotations

import csv
import glob
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np

REPO = Path("/Users/xr-12345/Desktop/SafeMedRAG")
FROZEN = REPO / "artifacts/verimedrag-hypothesis-validation-2026/matched-budget"
NEW = REPO / "artifacts/verimedrag-hypothesis-validation-2026/learned-gate-integration"
ANALYSIS = NEW / "analysis"

THRESHOLD_LABELS = {
    "t060": 0.60,
    "t070": 0.70,
    "t080": 0.80,
    "t085": 0.85,
    "t090": 0.90,
    "t095": 0.95,
}
NOISE_RATES = (0.0, 0.1, 0.2, 0.3)
SEEDS = (2026, 2027, 2028)
FROZEN_STRATEGIES = (
    "full_two_layer",
    "joint_new_verify_stop",
    "oracle_verify",
    "oracle_select_same_channel",
)
LEARNED_STRATEGY = "joint_learned_gate"
BASELINE_STRATEGY = "joint_new_verify_stop"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 2026


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def threshold_from_path(path: Path) -> float:
    for part in path.parts:
        if part in THRESHOLD_LABELS:
            return THRESHOLD_LABELS[part]
    raise ValueError(f"no threshold label in path: {path}")


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means: list[float] = []
    remaining = samples
    while remaining:
        batch = min(200, remaining)
        idx = rng.integers(0, len(data), size=(batch, len(data)), endpoint=False)
        means.extend(data[idx].mean(axis=1).tolist())
        remaining -= batch
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load(glob_pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(Path(p) for p in glob.glob(glob_pattern)):
        threshold = threshold_from_path(path)
        for r in read_rows(path):
            rows.append(
                {
                    "strategy": r["strategy"],
                    "case_id": r["case_id"],
                    "seed": int(r["seed"]),
                    "noise_rate": float(r["noise_rate"]),
                    "posterior_threshold": threshold,
                    "correct": int(r["correct_top1"]),
                    "brier": float(r["brier_score"]),
                    "new_questions": int(r["new_questions"]),
                    "verification_questions": int(r["verification_questions"]),
                    "total_atomic_questions": int(r["total_atomic_questions"]),
                    "unnecessary_verifications": int(r["unnecessary_verifications"]),
                    "resolved_wrong_reports": int(r["resolved_wrong_reports"]),
                }
            )
    return rows


def completeness(rows: list[dict], label: str, strategies: tuple[str, ...]) -> None:
    expected = len(SEEDS) * 980
    groups: dict[tuple, int] = defaultdict(int)
    for r in rows:
        groups[(r["strategy"], r["noise_rate"], r["posterior_threshold"])] += 1
    bad = {k: v for k, v in sorted(groups.items()) if v != expected}
    print(f"[{label}] rows={len(rows)} expected={expected*len(strategies)*4*6}")
    print(f"[{label}] groups != {expected}: {bad or 'none'}")


def regression_check(my_rows: list[dict], frozen_rows: list[dict]) -> None:
    frozen = {
        (r["case_id"], r["seed"], r["noise_rate"], r["posterior_threshold"]): r
        for r in frozen_rows
        if r["strategy"] == BASELINE_STRATEGY
    }
    mine = {
        (r["case_id"], r["seed"], r["noise_rate"], r["posterior_threshold"]): r
        for r in my_rows
        if r["strategy"] == BASELINE_STRATEGY
    }
    keys = set(frozen) & set(mine)
    mismatched = 0
    for key in keys:
        f, m = frozen[key], mine[key]
        for field in (
            "correct",
            "new_questions",
            "verification_questions",
            "total_atomic_questions",
            "unnecessary_verifications",
            "resolved_wrong_reports",
        ):
            if f[field] != m[field]:
                mismatched += 1
                break
    lines = [
        f"frozen joint rows: {len(frozen)}",
        f"re-run joint rows: {len(mine)}",
        f"shared (case,seed,noise,threshold) keys: {len(keys)}",
        f"exact-match ratio: {(len(keys) - mismatched)}/{len(keys)}",
        (
            "REGRESSION PASS: re-run joint_new_verify_stop is byte-identical to frozen."
            if mismatched == 0 and len(keys) == len(frozen) == len(mine)
            else f"REGRESSION MISMATCH: {mismatched} rows differ."
        ),
    ]
    (ANALYSIS / "regression_check.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


def summary_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["strategy"], r["noise_rate"], r["posterior_threshold"])].append(r)
    out: list[dict] = []
    for (strategy, noise, threshold), recs in sorted(groups.items()):
        verify = sum(r["verification_questions"] for r in recs)
        resolved = sum(r["resolved_wrong_reports"] for r in recs)
        unnecessary = sum(r["unnecessary_verifications"] for r in recs)
        out.append(
            {
                "strategy": strategy,
                "noise_rate": f"{noise:.1f}",
                "posterior_threshold": f"{threshold:.2f}",
                "cases": len(recs),
                "top1_accuracy": fmean(r["correct"] for r in recs),
                "brier_score": fmean(r["brier"] for r in recs),
                "avg_total_atomic_questions": fmean(
                    r["total_atomic_questions"] for r in recs
                ),
                "avg_new_questions": fmean(r["new_questions"] for r in recs),
                "avg_verification_questions": fmean(
                    r["verification_questions"] for r in recs
                ),
                "unnecessary_verification_rate": (
                    unnecessary / verify if verify else None
                ),
                "conflict_resolution_rate": (resolved / verify if verify else None),
            }
        )
    return out


def _csv_fmt(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return f"{v:.6f}"


def write_summary(rows: list[dict]) -> None:
    fieldnames = [
        "strategy",
        "noise_rate",
        "posterior_threshold",
        "cases",
        "top1_accuracy",
        "brier_score",
        "avg_total_atomic_questions",
        "avg_new_questions",
        "avg_verification_questions",
        "unnecessary_verification_rate",
        "conflict_resolution_rate",
    ]
    with (ANALYSIS / "summary_by_strategy_noise_threshold.csv").open(
        "w", newline=""
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_fmt(r[k]) for k in fieldnames})
    print(f"wrote summary_by_strategy_noise_threshold.csv ({len(rows)} rows)")


def curve_points(rows: list[dict]) -> list[dict]:
    index: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["strategy"], r["noise_rate"], r["posterior_threshold"])
        index[key][r["case_id"]].append(r)
    points: list[dict] = []
    for (strategy, noise, threshold), case_map in sorted(index.items()):
        per_case_acc = [fmean([rr["correct"] for rr in recs]) for recs in case_map.values()]
        per_case_q = [
            fmean([rr["total_atomic_questions"] for rr in recs])
            for recs in case_map.values()
        ]
        per_case_brier = [fmean([rr["brier"] for rr in recs]) for recs in case_map.values()]
        ci_low, ci_high = bootstrap_ci(per_case_acc, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
        points.append(
            {
                "strategy": strategy,
                "noise_rate": noise,
                "posterior_threshold": threshold,
                "cases": len(per_case_acc),
                "accuracy": fmean(per_case_acc),
                "accuracy_ci_low": ci_low,
                "accuracy_ci_high": ci_high,
                "average_questions": fmean(per_case_q),
                "questions_standard_error": pstdev(per_case_q) / math.sqrt(len(per_case_q)),
                "brier_score": fmean(per_case_brier),
            }
        )
    return points


def write_curve_points(points: list[dict]) -> None:
    fieldnames = [
        "strategy",
        "noise_rate",
        "posterior_threshold",
        "cases",
        "accuracy",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "average_questions",
        "questions_standard_error",
        "brier_score",
    ]
    with (ANALYSIS / "curve_points.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for p in points:
            w.writerow({k: p[k] for k in fieldnames})
    print(f"wrote curve_points.csv ({len(points)} rows)")


def _delta(a, b):
    if a is None or b is None:
        return ""
    return f"{a - b:.6f}"


def learned_deltas(my_summary: list[dict]) -> None:
    base = {
        (r["noise_rate"], r["posterior_threshold"]): r
        for r in my_summary
        if r["strategy"] == BASELINE_STRATEGY
    }
    out: list[dict] = []
    for r in my_summary:
        if r["strategy"] != LEARNED_STRATEGY:
            continue
        b = base[(r["noise_rate"], r["posterior_threshold"])]
        out.append(
            {
                "noise_rate": r["noise_rate"],
                "posterior_threshold": r["posterior_threshold"],
                "learned_top1_accuracy": f"{r['top1_accuracy']:.6f}",
                "baseline_top1_accuracy": f"{b['top1_accuracy']:.6f}",
                "delta_accuracy": f"{r['top1_accuracy'] - b['top1_accuracy']:.6f}",
                "learned_avg_questions": f"{r['avg_total_atomic_questions']:.6f}",
                "baseline_avg_questions": f"{b['avg_total_atomic_questions']:.6f}",
                "delta_questions": f"{r['avg_total_atomic_questions'] - b['avg_total_atomic_questions']:.6f}",
                "learned_unnecessary_rate": _csv_fmt(r['unnecessary_verification_rate']),
                "baseline_unnecessary_rate": _csv_fmt(b['unnecessary_verification_rate']),
                "delta_unnecessary_rate": _delta(
                    r['unnecessary_verification_rate'], b['unnecessary_verification_rate']
                ),
                "learned_conflict_resolution_rate": _csv_fmt(r['conflict_resolution_rate']),
                "baseline_conflict_resolution_rate": _csv_fmt(b['conflict_resolution_rate']),
                "delta_conflict_resolution_rate": _delta(
                    r['conflict_resolution_rate'], b['conflict_resolution_rate']
                ),
            }
        )
    fieldnames = list(out[0].keys())
    with (ANALYSIS / "learned_vs_baseline_deltas.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out)
    print(f"wrote learned_vs_baseline_deltas.csv ({len(out)} rows)")


def combined_pareto(all_points: list[dict]) -> None:
    out: list[dict] = []
    for noise in NOISE_RATES:
        subset = [p for p in all_points if p["noise_rate"] == noise]
        for p in subset:
            dominated = any(
                q["accuracy"] >= p["accuracy"]
                and q["average_questions"] <= p["average_questions"]
                and (
                    q["accuracy"] > p["accuracy"]
                    or q["average_questions"] < p["average_questions"]
                )
                for q in subset
            )
            out.append(
                {
                    "noise_rate": f"{noise:.1f}",
                    "strategy": p["strategy"],
                    "posterior_threshold": f"{p['posterior_threshold']:.2f}",
                    "accuracy": f"{p['accuracy']:.6f}",
                    "average_questions": f"{p['average_questions']:.6f}",
                    "on_pareto_frontier": "1" if not dominated else "0",
                }
            )
    fieldnames = list(out[0].keys())
    with (ANALYSIS / "pareto_frontier_combined.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out)
    print(f"wrote pareto_frontier_combined.csv ({len(out)} rows)")


def budget_match_learned_vs_full(all_points: list[dict]) -> None:
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from powerful_medrag.benchmark import CurvePoint
    from powerful_medrag.curve_analysis import (
        compare_at_equal_budget,
        save_budget_matches,
    )

    def to_curve(rows: list[dict]) -> list[CurvePoint]:
        return [
            CurvePoint(
                strategy=r["strategy"],
                noise_rate=r["noise_rate"],
                posterior_threshold=r["posterior_threshold"],
                cases=r["cases"],
                accuracy=r["accuracy"],
                accuracy_ci_low=r["accuracy_ci_low"],
                accuracy_ci_high=r["accuracy_ci_high"],
                average_questions=r["average_questions"],
                questions_standard_error=r["questions_standard_error"],
                brier_score=r["brier_score"],
            )
            for r in rows
        ]

    curve = to_curve(all_points)
    for candidate, reference in (
        (LEARNED_STRATEGY, "full_two_layer"),
        (LEARNED_STRATEGY, BASELINE_STRATEGY),
    ):
        matches = compare_at_equal_budget(
            curve,
            curve,
            candidate_strategy=candidate,
            reference_strategy=reference,
        )
        name = f"budget_matches_{candidate}_vs_{reference}.csv"
        save_budget_matches(matches, ANALYSIS / name)
        print(f"wrote {name} ({len(matches)} rows)")
        for noise in NOISE_RATES:
            subset = [m for m in matches if m.noise_rate == noise]
            if not subset:
                continue
            mean_adv = fmean(m.reference_minus_candidate_accuracy for m in subset)
            dominated = sum(m.reference_discretely_dominates for m in subset)
            print(
                f"  {candidate} vs {reference} noise={noise:.0%}: "
                f"reference advantage={mean_adv:+.4f} dominated={dominated}/{len(subset)}"
            )


def plot_curves(points: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = sorted({p["strategy"] for p in points})
    markers = {
        "full_two_layer": "o",
        "joint_new_verify_stop": "^",
        "joint_learned_gate": "*",
        "oracle_verify": "s",
        "oracle_select_same_channel": "D",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for axis, noise in zip(axes.flat, NOISE_RATES):
        for strategy in strategies:
            subset = sorted(
                (p for p in points if p["strategy"] == strategy and p["noise_rate"] == noise),
                key=lambda p: p["average_questions"],
            )
            if not subset:
                continue
            axis.plot(
                [p["average_questions"] for p in subset],
                [p["accuracy"] for p in subset],
                marker=markers.get(strategy, "x"),
                markersize=5,
                label=strategy,
            )
        axis.set_title(f"Answer noise = {noise:.0%}")
        axis.set_xlabel("Average total atomic questions")
        axis.set_ylabel("Top-1 accuracy")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6)
    fig.suptitle("Learned-gate joint policy vs. frozen baselines")
    fig.tight_layout()
    fig.savefig(ANALYSIS / "accuracy_questions_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote accuracy_questions_curves.png")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    my_rows = load(str(NEW / "reliability-sweep" / "t*" / "seed-*" / "ddxplus_reliability_outcomes.csv"))
    frozen_rows = load(str(FROZEN / "reliability-sweep" / "t*" / "seed-*" / "ddxplus_reliability_outcomes.csv"))
    completeness(my_rows, "new", (BASELINE_STRATEGY, LEARNED_STRATEGY))
    completeness(frozen_rows, "frozen", FROZEN_STRATEGIES)

    regression_check(my_rows, frozen_rows)

    my_summary = summary_rows(my_rows)
    write_summary(my_summary)
    learned_deltas(my_summary)

    # Case-clustered curve points: frozen 4 strategies + the learned strategy
    # (use the frozen joint baseline, not the re-run, to keep the frontier on
    # the frozen reference; the re-run is proven identical by regression_check).
    frozen_points = curve_points(frozen_rows)
    learned_points = curve_points([r for r in my_rows if r["strategy"] == LEARNED_STRATEGY])
    all_points = frozen_points + learned_points
    write_curve_points(all_points)
    combined_pareto(all_points)
    budget_match_learned_vs_full(all_points)
    plot_curves(all_points)
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
