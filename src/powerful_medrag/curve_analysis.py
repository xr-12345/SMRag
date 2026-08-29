"""Budget-matched comparison of accuracy--question curves."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

from .benchmark import CurvePoint


@dataclass(frozen=True)
class BudgetMatchedPoint:
    noise_rate: float
    candidate_strategy: str
    reference_strategy: str
    candidate_threshold: float
    candidate_questions: float
    candidate_accuracy: float
    reference_threshold_low: float
    reference_threshold_high: float
    reference_accuracy_at_same_questions: float
    reference_minus_candidate_accuracy: float
    reference_discretely_dominates: bool


def load_curve_points(path: str | Path) -> list[CurvePoint]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            CurvePoint(
                strategy=row["strategy"],
                noise_rate=float(row["noise_rate"]),
                posterior_threshold=float(row["posterior_threshold"]),
                cases=int(row["cases"]),
                accuracy=float(row["accuracy"]),
                accuracy_ci_low=float(row["accuracy_ci_low"]),
                accuracy_ci_high=float(row["accuracy_ci_high"]),
                average_questions=float(row["average_questions"]),
                questions_standard_error=float(row["questions_standard_error"]),
                brier_score=float(row["brier_score"]),
            )
            for row in rows
        ]


def compare_at_equal_budget(
    candidate_points: Iterable[CurvePoint],
    reference_points: Iterable[CurvePoint],
    *,
    candidate_strategy: str,
    reference_strategy: str,
) -> list[BudgetMatchedPoint]:
    """Interpolate reference accuracy at every candidate question budget.

    Points outside the reference curve's observed question range are omitted;
    extrapolation would make the comparison depend on an unsupported trend.
    """

    candidate = [
        point for point in candidate_points if point.strategy == candidate_strategy
    ]
    reference = [
        point for point in reference_points if point.strategy == reference_strategy
    ]
    if not candidate:
        raise ValueError(f"candidate strategy not found: {candidate_strategy}")
    if not reference:
        raise ValueError(f"reference strategy not found: {reference_strategy}")

    reference_by_noise: dict[float, list[CurvePoint]] = defaultdict(list)
    for point in reference:
        reference_by_noise[point.noise_rate].append(point)
    for points in reference_by_noise.values():
        points.sort(key=lambda point: point.average_questions)

    comparisons: list[BudgetMatchedPoint] = []
    for point in sorted(
        candidate,
        key=lambda item: (
            item.noise_rate,
            item.average_questions,
            item.posterior_threshold,
        ),
    ):
        curve = reference_by_noise.get(point.noise_rate, [])
        bracket = _bracket(curve, point.average_questions)
        if bracket is None:
            continue
        low, high = bracket
        reference_accuracy = _linear_interpolate(
            low.average_questions,
            low.accuracy,
            high.average_questions,
            high.accuracy,
            point.average_questions,
        )
        dominated = any(
            reference_point.average_questions <= point.average_questions
            and reference_point.accuracy >= point.accuracy
            and (
                reference_point.average_questions < point.average_questions
                or reference_point.accuracy > point.accuracy
            )
            for reference_point in curve
        )
        comparisons.append(
            BudgetMatchedPoint(
                noise_rate=point.noise_rate,
                candidate_strategy=candidate_strategy,
                reference_strategy=reference_strategy,
                candidate_threshold=point.posterior_threshold,
                candidate_questions=point.average_questions,
                candidate_accuracy=point.accuracy,
                reference_threshold_low=low.posterior_threshold,
                reference_threshold_high=high.posterior_threshold,
                reference_accuracy_at_same_questions=reference_accuracy,
                reference_minus_candidate_accuracy=(
                    reference_accuracy - point.accuracy
                ),
                reference_discretely_dominates=dominated,
            )
        )
    return comparisons


def _bracket(
    points: Sequence[CurvePoint], questions: float
) -> tuple[CurvePoint, CurvePoint] | None:
    if not points:
        return None
    if questions < points[0].average_questions or questions > points[-1].average_questions:
        return None
    for low, high in zip(points, points[1:]):
        if low.average_questions <= questions <= high.average_questions:
            return low, high
    if questions == points[-1].average_questions:
        return points[-1], points[-1]
    return None


def _linear_interpolate(
    x_low: float,
    y_low: float,
    x_high: float,
    y_high: float,
    x: float,
) -> float:
    if x_high == x_low:
        return max(y_low, y_high)
    weight = (x - x_low) / (x_high - x_low)
    return y_low + weight * (y_high - y_low)


def save_budget_matches(
    matches: Iterable[BudgetMatchedPoint], path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[field.name for field in fields(BudgetMatchedPoint)],
        )
        writer.writeheader()
        for match in matches:
            writer.writerow(match.__dict__)


def plot_curve_comparison(
    candidate_points: Iterable[CurvePoint],
    reference_points: Iterable[CurvePoint],
    *,
    candidate_strategy: str,
    reference_strategy: str,
    path: str | Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires matplotlib") from exc

    candidate = [
        point for point in candidate_points if point.strategy == candidate_strategy
    ]
    reference = [
        point for point in reference_points if point.strategy == reference_strategy
    ]
    noise_rates = sorted({point.noise_rate for point in candidate})
    if not noise_rates:
        raise ValueError(f"candidate strategy not found: {candidate_strategy}")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=False, sharey=False)
    for axis, noise_rate in zip(axes.flat, noise_rates):
        for points, strategy, marker in (
            (candidate, candidate_strategy, "o"),
            (reference, reference_strategy, "s"),
        ):
            subset = sorted(
                (point for point in points if point.noise_rate == noise_rate),
                key=lambda point: point.average_questions,
            )
            axis.plot(
                [point.average_questions for point in subset],
                [point.accuracy for point in subset],
                marker=marker,
                markersize=4,
                label=strategy,
            )
        axis.set_title(f"Answer noise = {noise_rate:.0%}")
        axis.set_xlabel("Average report turns")
        axis.set_ylabel("Top-1 accuracy")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes.flat[len(noise_rates) :]:
        axis.set_visible(False)
    fig.suptitle("Budget-matched validation curves")
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--candidate-strategy", required=True)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--reference-strategy", required=True)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-figure", required=True, type=Path)
    args = parser.parse_args(argv)

    candidate = load_curve_points(args.candidate_csv)
    reference = load_curve_points(args.reference_csv)
    matches = compare_at_equal_budget(
        candidate,
        reference,
        candidate_strategy=args.candidate_strategy,
        reference_strategy=args.reference_strategy,
    )
    save_budget_matches(matches, args.output_csv)
    plot_curve_comparison(
        candidate,
        reference,
        candidate_strategy=args.candidate_strategy,
        reference_strategy=args.reference_strategy,
        path=args.output_figure,
    )
    print(f"budget-matched points: {len(matches)}")
    for noise_rate in sorted({match.noise_rate for match in matches}):
        subset = [match for match in matches if match.noise_rate == noise_rate]
        mean_advantage = sum(
            match.reference_minus_candidate_accuracy for match in subset
        ) / len(subset)
        dominated = sum(match.reference_discretely_dominates for match in subset)
        print(
            f"noise={noise_rate:.0%}: reference mean advantage="
            f"{mean_advantage:+.4f}; dominated={dominated}/{len(subset)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
