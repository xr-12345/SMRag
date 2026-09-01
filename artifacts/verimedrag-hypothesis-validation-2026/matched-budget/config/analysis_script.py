"""Prompt #6 matched-budget analysis.

Reads the 18 reliability partial CSVs (4 strategies x 6 thresholds x 3 seeds)
and the 3 retro seed CSVs (1 variant x 6 thresholds x 3 seeds), then produces:

  analysis/case_outcomes.csv            unified per-case-per-seed outcomes
  analysis/summary_by_strategy_noise_threshold.csv
  analysis/curve_points.csv             CurvePoint schema (accuracy vs questions)
  analysis/budget_matches_*.csv         matched-budget interpolated comparisons
  analysis/pareto_frontier.csv          per-noise Pareto frontier
  analysis/oracle_gap_decomposition.csv identification vs correction split
  analysis/accuracy_questions_curves.png

Deterministic; bootstrap seeded at 2026.
"""

from __future__ import annotations

import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np

REPO = Path("/Users/xr-12345/Desktop/SafeMedRAG")
ROOT = REPO / "artifacts/verimedrag-hypothesis-validation-2026/matched-budget"
REL_GLOB = str(ROOT / "reliability-sweep" / "t*" / "seed-*" / "ddxplus_reliability_outcomes.csv")
RETRO_GLOB = str(ROOT / "retro-sweep" / "seed-*" / "ddxplus_clarification_case_outcomes.csv")
ANALYSIS = ROOT / "analysis"

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
RELIABILITY_STRATEGIES = (
    "full_two_layer",
    "joint_new_verify_stop",
    "oracle_verify",
    "oracle_select_same_channel",
)
RETRO_STRATEGY = "retro_utility_u050_b1"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 2026

NA = ""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
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


def load_unified() -> tuple[list[dict], list[str]]:
    """Return normalized case-level rows (one per case-seed-threshold-strategy)."""
    rows: list[dict] = []
    reliability_files = sorted(Path(p) for p in glob.glob(REL_GLOB))
    for path in reliability_files:
        threshold = threshold_from_path(path)
        for r in read_csv_rows(path):
            rows.append(
                {
                    "source": "reliability",
                    "strategy": r["strategy"],
                    "case_id": r["case_id"],
                    "seed": int(r["seed"]),
                    "noise_rate": float(r["noise_rate"]),
                    "posterior_threshold": threshold,
                    "true_diagnosis": r["true_diagnosis"],
                    "predicted_diagnosis": r["predicted_diagnosis"],
                    "correct": int(r["correct_top1"]),
                    "correct_top3": int(r["correct_top3"]),
                    "brier": float(r["brier_score"]),
                    "new_questions": int(r["new_questions"]),
                    "verification_questions": int(r["verification_questions"]),
                    "total_atomic_questions": int(r["total_atomic_questions"]),
                    "unnecessary_verifications": int(r["unnecessary_verifications"]),
                    "resolved_wrong_reports": int(r["resolved_wrong_reports"]),
                    "premature_stop": int(r["premature_stop"]),
                    "uncertain_output": int(r["uncertain_output"]),
                    "stop_reason": r["stop_reason"],
                }
            )
    retro_files = sorted(Path(p) for p in glob.glob(RETRO_GLOB))
    for path in retro_files:
        for r in read_csv_rows(path):
            rows.append(
                {
                    "source": "retro",
                    "strategy": r["strategy"],
                    "case_id": r["case_id"],
                    "seed": int(r["seed"]),
                    "noise_rate": float(r["noise_rate"]),
                    "posterior_threshold": float(r["posterior_threshold"]),
                    "true_diagnosis": r["true_diagnosis"],
                    "predicted_diagnosis": r["predicted_diagnosis"],
                    "correct": int(r["correct"]),
                    "correct_top3": NA,
                    "brier": float(r["brier_score"]),
                    "new_questions": int(r["questions"]) - int(r["clarifications"]),
                    "verification_questions": int(r["clarifications"]),
                    "total_atomic_questions": int(r["questions"]),
                    "unnecessary_verifications": NA,
                    "resolved_wrong_reports": NA,
                    "premature_stop": NA,
                    "uncertain_output": NA,
                    "stop_reason": NA,
                }
            )
    fieldnames = list(rows[0].keys())
    return rows, fieldnames


def completeness_check(rows: list[dict]) -> None:
    print("=" * 70)
    print("COMPLETENESS CHECK")
    print("=" * 70)
    expected_per_rel_group = len(SEEDS) * 980  # reliability: 980 cases x 3 seeds
    rel = [r for r in rows if r["source"] == "reliability"]
    retro = [r for r in rows if r["source"] == "retro"]

    rel_groups: dict[tuple, int] = defaultdict(int)
    for r in rel:
        rel_groups[(r["strategy"], r["noise_rate"], r["posterior_threshold"])] += 1
    print(f"reliability total rows: {len(rel)} (expected 4x4x6x3x980 = {4*4*6*3*980})")
    bad = {k: v for k, v in sorted(rel_groups.items()) if v != 2940}
    print(f"reliability (strategy,noise,threshold) groups != 2940: {bad or 'none'}")

    retro_groups: dict[tuple, int] = defaultdict(int)
    for r in retro:
        retro_groups[(r["noise_rate"], r["posterior_threshold"])] += 1
    print(f"retro total rows: {len(retro)} (expected 4x6x3x980 = {4*6*3*980})")
    retro_bad = {k: v for k, v in sorted(retro_groups.items()) if v != 2940}
    print(f"retro (noise,threshold) groups != 2940: {retro_bad or 'none'}")

    rel_strategies = sorted({r["strategy"] for r in rel})
    retro_strategies = sorted({r["strategy"] for r in retro})
    print(f"reliability strategies: {rel_strategies}")
    print(f"retro strategies: {retro_strategies}")
    print(f"reliability case_ids unique: {len({r['case_id'] for r in rel})}")
    print(f"retro case_ids unique: {len({r['case_id'] for r in retro})}")
    print(f"reliability == retro case_ids: "
          f"{ {r['case_id'] for r in rel} == {r['case_id'] for r in retro} }")
    print("test split accessed: False (inputs are release_validate_patients.zip)")
    print("=" * 70)


def save_case_outcomes(rows: list[dict], fieldnames: list[str]) -> None:
    with (ANALYSIS / "case_outcomes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote case_outcomes.csv ({len(rows)} rows)")


def curve_points(rows: list[dict]) -> list[dict]:
    """Aggregate case-level rows into CurvePoint-schema records.

    Accuracy and questions are case-clustered: mean over the 3 seeds per case,
    then a mean (and SE / bootstrap CI) across the 980 cases.
    """
    index: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        key = (r["strategy"], r["noise_rate"], r["posterior_threshold"])
        index[key][r["case_id"]].append(r)

    points: list[dict] = []
    for (strategy, noise, threshold), case_map in sorted(index.items()):
        per_case_acc = [
            fmean([rr["correct"] for rr in recs]) for recs in case_map.values()
        ]
        per_case_q = [
            fmean([rr["total_atomic_questions"] for rr in recs])
            for recs in case_map.values()
        ]
        per_case_brier = [
            fmean([rr["brier"] for rr in recs]) for recs in case_map.values()
        ]
        accuracy = fmean(per_case_acc)
        questions = fmean(per_case_q)
        brier = fmean(per_case_brier)
        ci_low, ci_high = bootstrap_ci(
            per_case_acc, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED
        )
        q_se = pstdev(per_case_q) / math.sqrt(len(per_case_q))
        points.append(
            {
                "strategy": strategy,
                "noise_rate": noise,
                "posterior_threshold": threshold,
                "cases": len(per_case_acc),
                "accuracy": accuracy,
                "accuracy_ci_low": ci_low,
                "accuracy_ci_high": ci_high,
                "average_questions": questions,
                "questions_standard_error": q_se,
                "brier_score": brier,
            }
        )
    return points


def save_curve_points(points: list[dict]) -> None:
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


def save_summary(rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["strategy"], r["noise_rate"], r["posterior_threshold"])].append(r)
    out: list[dict] = []
    for (strategy, noise, threshold), recs in sorted(groups.items()):
        n = len(recs)

        def _sum_field(key: str) -> tuple[int, bool]:
            present = [r for r in recs if r[key] not in ("", "NA")]
            return sum(int(r[key]) for r in present), len(present) == len(recs)

        verify, _ = _sum_field("verification_questions")
        resolved, resolved_ok = _sum_field("resolved_wrong_reports")
        unnecessary, unnecessary_ok = _sum_field("unnecessary_verifications")
        out.append(
            {
                "strategy": strategy,
                "noise_rate": f"{noise:.1f}",
                "posterior_threshold": f"{threshold:.2f}",
                "cases": str(n),
                "top1_accuracy": f"{fmean(r['correct'] for r in recs):.6f}",
                "brier_score": f"{fmean(r['brier'] for r in recs):.6f}",
                "avg_total_atomic_questions": f"{fmean(r['total_atomic_questions'] for r in recs):.6f}",
                "avg_new_questions": f"{fmean(r['new_questions'] for r in recs):.6f}",
                "avg_verification_questions": f"{fmean(r['verification_questions'] for r in recs):.6f}",
                "unnecessary_verification_rate": (
                    f"{unnecessary / verify:.6f}" if verify and unnecessary_ok else NA
                ),
                "conflict_resolution_rate": (
                    f"{resolved / verify:.6f}" if verify and resolved_ok else NA
                ),
            }
        )
    fieldnames = list(out[0].keys())
    with (ANALYSIS / "summary_by_strategy_noise_threshold.csv").open(
        "w", newline=""
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out)
    print(f"wrote summary_by_strategy_noise_threshold.csv ({len(out)} rows)")


def save_pareto_frontier(points: list[dict]) -> None:
    """Per-noise Pareto frontier across all strategies and thresholds."""
    out: list[dict] = []
    for noise in NOISE_RATES:
        subset = [p for p in points if p["noise_rate"] == noise]
        # A point is Pareto-optimal (non-dominated) if no other point has both
        # higher-or-equal accuracy and lower-or-equal questions (one strict).
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
    with (ANALYSIS / "pareto_frontier.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out)
    print(f"wrote pareto_frontier.csv ({len(out)} rows)")


def pointwise_dominance(points: list[dict]) -> list[dict]:
    """joint vs full_two_layer at the same frozen threshold (pointwise Pareto)."""
    out: list[dict] = []
    by_key = {
        (p["strategy"], p["noise_rate"], p["posterior_threshold"]): p
        for p in points
    }
    for noise in NOISE_RATES:
        for threshold in sorted({p["posterior_threshold"] for p in points}):
            j = by_key.get(("joint_new_verify_stop", noise, threshold))
            f = by_key.get(("full_two_layer", noise, threshold))
            if j is None or f is None:
                continue
            # full dominates joint if full is >= accuracy and <= questions (one strict)
            full_dominates = (
                f["accuracy"] >= j["accuracy"]
                and f["average_questions"] <= j["average_questions"]
                and (
                    f["accuracy"] > j["accuracy"]
                    or f["average_questions"] < j["average_questions"]
                )
            )
            out.append(
                {
                    "noise_rate": f"{noise:.1f}",
                    "posterior_threshold": f"{threshold:.2f}",
                    "joint_accuracy": f"{j['accuracy']:.6f}",
                    "full_accuracy": f"{f['accuracy']:.6f}",
                    "accuracy_delta": f"{j['accuracy'] - f['accuracy']:.6f}",
                    "joint_questions": f"{j['average_questions']:.6f}",
                    "full_questions": f"{f['average_questions']:.6f}",
                    "question_delta": f"{j['average_questions'] - f['average_questions']:.6f}",
                    "joint_dominated_by_full": "1" if full_dominates else "0",
                }
            )
    fieldnames = list(out[0].keys())
    with (ANALYSIS / "pointwise_pareto_joint_vs_full.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out)
    print(f"wrote pointwise_pareto_joint_vs_full.csv ({len(out)} rows)")
    return out


def save_oracle_gap_decomposition(points: list[dict]) -> None:
    """Exact additive split of oracle_verify - joint into identification + correction.

    At the same frozen threshold, the three strategies differ only in mechanism:
      identification = oracle_select_same_channel - joint          (selection)
      correction     = oracle_verify - oracle_select_same_channel  (correction)
      total          = oracle_verify - joint = identification + correction
    """
    by_key = {
        (p["strategy"], p["noise_rate"], p["posterior_threshold"]): p
        for p in points
    }
    out: list[dict] = []
    for noise in NOISE_RATES:
        for threshold in sorted({p["posterior_threshold"] for p in points}):
            j = by_key.get(("joint_new_verify_stop", noise, threshold))
            s = by_key.get(("oracle_select_same_channel", noise, threshold))
            v = by_key.get(("oracle_verify", noise, threshold))
            if j is None or s is None or v is None:
                continue
            identification = s["accuracy"] - j["accuracy"]
            correction = v["accuracy"] - s["accuracy"]
            total = v["accuracy"] - j["accuracy"]
            out.append(
                {
                    "noise_rate": f"{noise:.1f}",
                    "posterior_threshold": f"{threshold:.2f}",
                    "joint_accuracy": f"{j['accuracy']:.6f}",
                    "oracle_select_accuracy": f"{s['accuracy']:.6f}",
                    "oracle_verify_accuracy": f"{v['accuracy']:.6f}",
                    "identification_gap": f"{identification:.6f}",
                    "correction_gap": f"{correction:.6f}",
                    "total_gap": f"{total:.6f}",
                    "residual": f"{total - identification - correction:.2e}",
                    "joint_questions": f"{j['average_questions']:.6f}",
                    "oracle_select_questions": f"{s['average_questions']:.6f}",
                    "oracle_verify_questions": f"{v['average_questions']:.6f}",
                }
            )
    fieldnames = list(out[0].keys())
    with (ANALYSIS / "oracle_gap_decomposition.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out)
    print(f"wrote oracle_gap_decomposition.csv ({len(out)} rows)")


def budget_matches(points: list[dict]) -> None:
    """Matched-budget interpolated comparisons using curve_analysis."""
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from powerful_medrag.curve_analysis import (
        compare_at_equal_budget,
        save_budget_matches,
    )
    from powerful_medrag.benchmark import CurvePoint

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

    curve = to_curve(points)
    pairs = [
        ("joint_new_verify_stop", "full_two_layer"),
        ("oracle_verify", "full_two_layer"),
        ("oracle_select_same_channel", "full_two_layer"),
        ("joint_new_verify_stop", "oracle_select_same_channel"),
        ("oracle_select_same_channel", "oracle_verify"),
    ]
    for candidate, reference in pairs:
        matches = compare_at_equal_budget(
            curve,
            curve,
            candidate_strategy=candidate,
            reference_strategy=reference,
        )
        name = f"budget_matches_{candidate}_vs_{reference}.csv"
        save_budget_matches(matches, ANALYSIS / name)
        # summarize
        for noise in NOISE_RATES:
            subset = [m for m in matches if m.noise_rate == noise]
            if not subset:
                continue
            mean_adv = fmean(m.reference_minus_candidate_accuracy for m in subset)
            dominated = sum(m.reference_discretely_dominates for m in subset)
            print(
                f"  budget-match {candidate} vs {reference} noise={noise:.0%}: "
                f"reference advantage={mean_adv:+.4f} dominated={dominated}/{len(subset)}"
            )


def plot_curves(points: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires matplotlib") from exc

    strategies = sorted({p["strategy"] for p in points})
    markers = {
        "full_two_layer": "o",
        "joint_new_verify_stop": "^",
        "oracle_verify": "s",
        "oracle_select_same_channel": "D",
        "retro_utility_u050_b1": "v",
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
                markersize=4,
                label=strategy,
            )
        axis.set_title(f"Answer noise = {noise:.0%}")
        axis.set_xlabel("Average total atomic questions")
        axis.set_ylabel("Top-1 accuracy")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle("Risk vs. questions at frozen stopping thresholds")
    fig.tight_layout()
    fig.savefig(ANALYSIS / "accuracy_questions_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote accuracy_questions_curves.png")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    rows, fieldnames = load_unified()
    completeness_check(rows)
    save_case_outcomes(rows, fieldnames)
    save_summary(rows)
    points = curve_points(rows)
    save_curve_points(points)
    save_pareto_frontier(points)
    pointwise_dominance(points)
    save_oracle_gap_decomposition(points)
    budget_matches(points)
    plot_curves(points)
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
