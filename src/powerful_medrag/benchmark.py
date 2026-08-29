"""Paired DDXPlus accuracy--turns--noise experiments and curve rendering."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Mapping, MutableSequence

from .belief import BeliefTracker
from .estimation import DiseaseStateModel
from .questioning import (
    MedRAGReciprocalDegreeSelector,
    NumpyPrevalenceQuestionSelector,
    NumpyQuestionSelector,
)
from .schema import ClinicalCase, FeatureKey
from .simulator import (
    PatientProfile,
    StopRule,
    StructuredPatientSimulator,
    run_dialogue,
)


@dataclass(frozen=True)
class CurvePoint:
    strategy: str
    noise_rate: float
    posterior_threshold: float
    cases: int
    accuracy: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    average_questions: float
    questions_standard_error: float
    brier_score: float


@dataclass(frozen=True)
class CaseOutcome:
    seed: int
    case_id: str
    true_diagnosis: str
    strategy: str
    noise_rate: float
    posterior_threshold: float
    predicted_diagnosis: str
    correct: int
    questions: int
    brier_score: float
    mean_misreport_prior: float = 0.0
    max_misreport_prior: float = 0.0
    gate_activations: int = 0
    clarifications: int = 0
    harmful_reports: int = 0
    detected_harmful_reports: int = 0
    false_clarifications: int = 0
    mitigated_harmful_reports: int = 0
    discarded_correct_reports: int = 0


@dataclass(frozen=True)
class PairedComparison:
    noise_rate: float
    posterior_threshold: float
    cases_per_seed: int
    seeds: int
    eig_accuracy: float
    baseline_accuracy: float
    accuracy_difference: float
    accuracy_difference_ci_low: float
    accuracy_difference_ci_high: float
    eig_average_questions: float
    baseline_average_questions: float
    questions_saved: float
    questions_saved_ci_low: float
    questions_saved_ci_high: float
    eig_only_correct: int
    baseline_only_correct: int
    mcnemar_max_p: float


_PROCESS_CONTEXT: tuple[
    DiseaseStateModel,
    Mapping[str, NumpyQuestionSelector],
    tuple[float, ...],
    int,
    int,
] | None = None


def run_paired_curve_experiment(
    model: DiseaseStateModel,
    cases: Iterable[ClinicalCase],
    *,
    noise_rates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3),
    posterior_thresholds: tuple[float, ...] = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
    max_questions: int = 15,
    seed: int = 2026,
    progress: bool = False,
    disease_to_features: Mapping[str, Iterable[FeatureKey]] | None = None,
    include_prevalence: bool = True,
    workers: int = 1,
    executor_type: str = "thread",
    raw_outcomes: MutableSequence[CaseOutcome] | None = None,
) -> list[CurvePoint]:
    """Run each policy on identical cases and feature-indexed answer noise.

    One maximum-length trajectory is reused for every posterior threshold.  This
    makes the threshold curve exact while avoiding repeated simulation.
    """

    case_list = list(cases)
    if not case_list:
        raise ValueError("benchmark needs at least one case")
    if not noise_rates or not posterior_thresholds:
        raise ValueError("noise rates and thresholds cannot be empty")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if executor_type not in {"thread", "process"}:
        raise ValueError("executor_type must be 'thread' or 'process'")
    selectors = _make_selectors(disease_to_features, include_prevalence)
    observations: dict[tuple[str, float, float], list[tuple[int, int, float]]] = (
        defaultdict(list)
    )

    for case in case_list:
        if case.diagnosis not in model.diseases:
            raise ValueError(f"test disease missing from training model: {case.diagnosis}")
    # Build immutable NumPy likelihood caches before sharing selectors with threads.
    warm_tracker = BeliefTracker(model)
    for selector in selectors.values():
        selector._ensure_numpy_cache(warm_tracker)

    tasks = [
        (case, noise_rate)
        for noise_rate in noise_rates
        for case in case_list
    ]

    def evaluate(task: tuple[ClinicalCase, float]) -> list[CaseOutcome]:
        return _evaluate_case_noise(
            task,
            model=model,
            selectors=selectors,
            posterior_thresholds=posterior_thresholds,
            max_questions=max_questions,
            seed=seed,
        )

    if workers == 1:
        results = map(evaluate, tasks)
        executor = None
    elif executor_type == "process":
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_process_worker,
            initargs=(
                model,
                disease_to_features,
                include_prevalence,
                posterior_thresholds,
                max_questions,
                seed,
            ),
        )
        results = executor.map(
            _process_evaluate_case_noise,
            tasks,
            chunksize=max(1, len(tasks) // (workers * 20)),
        )
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(evaluate, tasks)
    try:
        progress_interval = max(1, len(tasks) // 20)
        for task_index, rows in enumerate(results, start=1):
            if raw_outcomes is not None:
                raw_outcomes.extend(rows)
            for row in rows:
                observations[(row.strategy, row.noise_rate, row.posterior_threshold)].append(
                    (row.correct, row.questions, row.brier_score)
                )
            if progress and (task_index % progress_interval == 0 or task_index == len(tasks)):
                print(f"completed {task_index}/{len(tasks)} case-noise tasks", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()

    points: list[CurvePoint] = []
    for (strategy, noise_rate, threshold), rows in sorted(observations.items()):
        correct_values = [row[0] for row in rows]
        question_values = [row[1] for row in rows]
        accuracy = fmean(correct_values)
        ci_low, ci_high = _wilson_interval(sum(correct_values), len(rows))
        question_se = (
            pstdev(question_values) / math.sqrt(len(question_values))
            if len(question_values) > 1
            else 0.0
        )
        points.append(
            CurvePoint(
                strategy=strategy,
                noise_rate=noise_rate,
                posterior_threshold=threshold,
                cases=len(rows),
                accuracy=accuracy,
                accuracy_ci_low=ci_low,
                accuracy_ci_high=ci_high,
                average_questions=fmean(question_values),
                questions_standard_error=question_se,
                brier_score=fmean(row[2] for row in rows),
            )
        )
    return points


def _make_selectors(
    disease_to_features: Mapping[str, Iterable[FeatureKey]] | None,
    include_prevalence: bool,
) -> dict[str, NumpyQuestionSelector]:
    selectors: dict[str, NumpyQuestionSelector] = {"eig": NumpyQuestionSelector()}
    if disease_to_features is not None:
        selectors["medrag_rdc"] = MedRAGReciprocalDegreeSelector(
            disease_to_features
        )
    if include_prevalence:
        selectors["prevalence"] = NumpyPrevalenceQuestionSelector()
    return selectors


def _evaluate_case_noise(
    task: tuple[ClinicalCase, float],
    *,
    model: DiseaseStateModel,
    selectors: Mapping[str, NumpyQuestionSelector],
    posterior_thresholds: tuple[float, ...],
    max_questions: int,
    seed: int,
) -> list[CaseOutcome]:
    case, noise_rate = task
    profile = PatientProfile.from_noise_rate(noise_rate)
    paired_seed = _stable_seed(seed, case.case_id, noise_rate)
    rows: list[CaseOutcome] = []
    for strategy, selector in selectors.items():
        patient = StructuredPatientSimulator(
            diagnosis=case.diagnosis,
            latent_states=case.states,
            model=model,
            profile=profile,
            seed=paired_seed,
        )
        tracker = BeliefTracker(model, patient.channel)
        for initial in case.initial_observations:
            tracker.update(initial)
        initial_belief = dict(tracker.belief)
        result = run_dialogue(
            patient,
            tracker=tracker,
            selector=selector,
            stop_rule=StopRule(
                max_questions=max_questions,
                posterior_threshold=1.0,
                entropy_threshold=0.0,
                minimum_question_utility=0.0,
            ),
        )
        trajectory: list[Mapping[str, float]] = [initial_belief]
        trajectory.extend(turn.update.posterior for turn in result.turns)
        for threshold in posterior_thresholds:
            turn_count, belief = _stop_at_threshold(trajectory, threshold)
            predicted = max(belief, key=belief.__getitem__)
            correct = int(predicted == case.diagnosis)
            brier = sum(
                (belief[disease] - (1.0 if disease == case.diagnosis else 0.0)) ** 2
                for disease in model.diseases
            )
            rows.append(
                CaseOutcome(
                    seed=seed,
                    case_id=case.case_id,
                    true_diagnosis=case.diagnosis,
                    strategy=strategy,
                    noise_rate=noise_rate,
                    posterior_threshold=threshold,
                    predicted_diagnosis=predicted,
                    correct=correct,
                    questions=turn_count,
                    brier_score=brier,
                )
            )
    return rows


def _initialize_process_worker(
    model: DiseaseStateModel,
    disease_to_features: Mapping[str, Iterable[FeatureKey]] | None,
    include_prevalence: bool,
    posterior_thresholds: tuple[float, ...],
    max_questions: int,
    seed: int,
) -> None:
    global _PROCESS_CONTEXT
    selectors = _make_selectors(disease_to_features, include_prevalence)
    warm_tracker = BeliefTracker(model)
    for selector in selectors.values():
        selector._ensure_numpy_cache(warm_tracker)
    _PROCESS_CONTEXT = (
        model,
        selectors,
        posterior_thresholds,
        max_questions,
        seed,
    )


def _process_evaluate_case_noise(
    task: tuple[ClinicalCase, float],
) -> list[CaseOutcome]:
    if _PROCESS_CONTEXT is None:
        raise RuntimeError("process benchmark worker was not initialized")
    model, selectors, posterior_thresholds, max_questions, seed = _PROCESS_CONTEXT
    return _evaluate_case_noise(
        task,
        model=model,
        selectors=selectors,
        posterior_thresholds=posterior_thresholds,
        max_questions=max_questions,
        seed=seed,
    )


def save_case_outcomes_csv(
    outcomes: Iterable[CaseOutcome], path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CaseOutcome.__dataclass_fields__))
        writer.writeheader()
        for outcome in outcomes:
            writer.writerow(outcome.__dict__)


def load_case_outcomes_csv(path: str | Path) -> list[CaseOutcome]:
    source = Path(path)
    outcomes: list[CaseOutcome] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            outcomes.append(
                CaseOutcome(
                    seed=int(row["seed"]),
                    case_id=row["case_id"],
                    true_diagnosis=row["true_diagnosis"],
                    strategy=row["strategy"],
                    noise_rate=float(row["noise_rate"]),
                    posterior_threshold=float(row["posterior_threshold"]),
                    predicted_diagnosis=row["predicted_diagnosis"],
                    correct=int(row["correct"]),
                    questions=int(row["questions"]),
                    brier_score=float(row["brier_score"]),
                    mean_misreport_prior=float(row.get("mean_misreport_prior", 0.0)),
                    max_misreport_prior=float(row.get("max_misreport_prior", 0.0)),
                    gate_activations=int(row.get("gate_activations", 0)),
                    clarifications=int(row.get("clarifications", 0)),
                    harmful_reports=int(row.get("harmful_reports", 0)),
                    detected_harmful_reports=int(
                        row.get("detected_harmful_reports", 0)
                    ),
                    false_clarifications=int(row.get("false_clarifications", 0)),
                    mitigated_harmful_reports=int(
                        row.get("mitigated_harmful_reports", 0)
                    ),
                    discarded_correct_reports=int(
                        row.get("discarded_correct_reports", 0)
                    ),
                )
            )
    return outcomes


def average_curve_points(points: Iterable[CurvePoint]) -> list[CurvePoint]:
    """Average curve coordinates across independent answer-noise seeds."""

    grouped: dict[tuple[str, float, float], list[CurvePoint]] = defaultdict(list)
    for point in points:
        grouped[(point.strategy, point.noise_rate, point.posterior_threshold)].append(
            point
        )
    averaged: list[CurvePoint] = []
    for (strategy, noise_rate, threshold), rows in sorted(grouped.items()):
        averaged.append(
            CurvePoint(
                strategy=strategy,
                noise_rate=noise_rate,
                posterior_threshold=threshold,
                cases=rows[0].cases,
                accuracy=fmean(row.accuracy for row in rows),
                accuracy_ci_low=fmean(row.accuracy_ci_low for row in rows),
                accuracy_ci_high=fmean(row.accuracy_ci_high for row in rows),
                average_questions=fmean(row.average_questions for row in rows),
                questions_standard_error=fmean(
                    row.questions_standard_error for row in rows
                ),
                brier_score=fmean(row.brier_score for row in rows),
            )
        )
    return averaged


def summarize_paired_outcomes(
    outcomes: Iterable[CaseOutcome],
    *,
    baseline_strategy: str = "medrag_rdc",
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 2026,
) -> list[PairedComparison]:
    """Compute cluster-bootstrap CIs and per-seed exact McNemar tests.

    The same test cases are intentionally reused across noise seeds.  We first
    average each paired difference within case, then bootstrap cases as the
    independent sampling unit.  McNemar is evaluated separately for every seed;
    the reported maximum p-value is a conservative cross-seed summary.
    """

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    index: dict[tuple[int, str, float, float, str], CaseOutcome] = {}
    for row in outcomes:
        key = (
            row.seed,
            row.case_id,
            row.noise_rate,
            row.posterior_threshold,
            row.strategy,
        )
        if key in index:
            raise ValueError(f"duplicate case outcome: {key}")
        index[key] = row

    conditions = sorted(
        {
            (row.noise_rate, row.posterior_threshold)
            for row in index.values()
            if row.strategy == "eig"
        }
    )
    comparisons: list[PairedComparison] = []
    for condition_index, (noise_rate, threshold) in enumerate(conditions):
        paired_by_seed: dict[int, list[tuple[CaseOutcome, CaseOutcome]]] = defaultdict(
            list
        )
        for key, eig in index.items():
            seed, case_id, row_noise, row_threshold, strategy = key
            if (
                strategy != "eig"
                or row_noise != noise_rate
                or row_threshold != threshold
            ):
                continue
            baseline_key = (
                seed,
                case_id,
                noise_rate,
                threshold,
                baseline_strategy,
            )
            if baseline_key not in index:
                raise ValueError(f"missing paired baseline outcome: {baseline_key}")
            paired_by_seed[seed].append((eig, index[baseline_key]))
        if not paired_by_seed:
            continue
        case_counts = {len(rows) for rows in paired_by_seed.values()}
        if len(case_counts) != 1:
            raise ValueError("all seeds must contain the same number of cases")

        per_case_accuracy: dict[str, list[float]] = defaultdict(list)
        per_case_questions: dict[str, list[float]] = defaultdict(list)
        eig_accuracy_by_seed: list[float] = []
        baseline_accuracy_by_seed: list[float] = []
        eig_questions_by_seed: list[float] = []
        baseline_questions_by_seed: list[float] = []
        eig_only_total = 0
        baseline_only_total = 0
        mcnemar_p_values: list[float] = []
        expected_case_ids: set[str] | None = None
        for seed, pairs in sorted(paired_by_seed.items()):
            case_ids = {eig.case_id for eig, _ in pairs}
            if expected_case_ids is None:
                expected_case_ids = case_ids
            elif case_ids != expected_case_ids:
                raise ValueError("case IDs differ across noise seeds")
            eig_correct = [eig.correct for eig, _ in pairs]
            baseline_correct = [baseline.correct for _, baseline in pairs]
            eig_questions = [eig.questions for eig, _ in pairs]
            baseline_questions = [baseline.questions for _, baseline in pairs]
            eig_accuracy_by_seed.append(fmean(eig_correct))
            baseline_accuracy_by_seed.append(fmean(baseline_correct))
            eig_questions_by_seed.append(fmean(eig_questions))
            baseline_questions_by_seed.append(fmean(baseline_questions))
            eig_only = sum(
                eig.correct == 1 and baseline.correct == 0
                for eig, baseline in pairs
            )
            baseline_only = sum(
                eig.correct == 0 and baseline.correct == 1
                for eig, baseline in pairs
            )
            eig_only_total += eig_only
            baseline_only_total += baseline_only
            mcnemar_p_values.append(_exact_mcnemar_p(eig_only, baseline_only))
            for eig, baseline in pairs:
                per_case_accuracy[eig.case_id].append(eig.correct - baseline.correct)
                per_case_questions[eig.case_id].append(
                    baseline.questions - eig.questions
                )

        accuracy_case_differences = [
            fmean(values) for _, values in sorted(per_case_accuracy.items())
        ]
        question_case_differences = [
            fmean(values) for _, values in sorted(per_case_questions.items())
        ]
        accuracy_low, accuracy_high = _bootstrap_mean_interval(
            accuracy_case_differences,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 2 * condition_index,
        )
        questions_low, questions_high = _bootstrap_mean_interval(
            question_case_differences,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 2 * condition_index + 1,
        )
        comparisons.append(
            PairedComparison(
                noise_rate=noise_rate,
                posterior_threshold=threshold,
                cases_per_seed=next(iter(case_counts)),
                seeds=len(paired_by_seed),
                eig_accuracy=fmean(eig_accuracy_by_seed),
                baseline_accuracy=fmean(baseline_accuracy_by_seed),
                accuracy_difference=fmean(accuracy_case_differences),
                accuracy_difference_ci_low=accuracy_low,
                accuracy_difference_ci_high=accuracy_high,
                eig_average_questions=fmean(eig_questions_by_seed),
                baseline_average_questions=fmean(baseline_questions_by_seed),
                questions_saved=fmean(question_case_differences),
                questions_saved_ci_low=questions_low,
                questions_saved_ci_high=questions_high,
                eig_only_correct=eig_only_total,
                baseline_only_correct=baseline_only_total,
                mcnemar_max_p=max(mcnemar_p_values),
            )
        )
    return comparisons


def save_paired_comparisons_csv(
    comparisons: Iterable[PairedComparison], path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(PairedComparison.__dataclass_fields__)
        )
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow(comparison.__dict__)


def save_curve_csv(points: Iterable[CurvePoint], path: str | Path) -> None:
    point_list = list(points)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CurvePoint.__dataclass_fields__))
        writer.writeheader()
        for point in point_list:
            writer.writerow(point.__dict__)


def plot_accuracy_turns_noise(points: Iterable[CurvePoint], path: str | Path) -> None:
    """Render accuracy-turn curves, with color encoding total noise rate."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
    except ImportError as exc:
        raise RuntimeError("plotting requires matplotlib") from exc

    point_list = list(points)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    noise_rates = sorted({point.noise_rate for point in point_list})
    strategies = tuple(sorted({point.strategy for point in point_list}))
    line_styles = {"eig": "-", "medrag_rdc": "--", "prevalence": ":"}
    markers = {"eig": "o", "medrag_rdc": "^", "prevalence": "s"}
    display_names = {
        "eig": "EIG",
        "medrag_rdc": "MedRAG-RDC",
        "prevalence": "Prevalence",
    }
    baseline_strategy = "medrag_rdc" if "medrag_rdc" in strategies else "prevalence"
    color_map = colormaps["viridis"]

    fig, (curve_axis, delta_axis) = plt.subplots(1, 2, figsize=(13, 5.2))
    for noise_index, noise_rate in enumerate(noise_rates):
        color = color_map(noise_index / max(1, len(noise_rates) - 1))
        for strategy in strategies:
            subset = sorted(
                (
                    point
                    for point in point_list
                    if point.strategy == strategy and point.noise_rate == noise_rate
                ),
                key=lambda point: point.average_questions,
            )
            curve_axis.plot(
                [point.average_questions for point in subset],
                [point.accuracy for point in subset],
                linestyle=line_styles.get(strategy, "-."),
                marker=markers.get(strategy, "x"),
                markersize=4,
                color=color,
                label=f"{display_names.get(strategy, strategy)}, noise={noise_rate:.0%}",
            )

        eig_by_threshold = {
            point.posterior_threshold: point
            for point in point_list
            if point.strategy == "eig" and point.noise_rate == noise_rate
        }
        baseline_by_threshold = {
            point.posterior_threshold: point
            for point in point_list
            if point.strategy == baseline_strategy and point.noise_rate == noise_rate
        }
        common_thresholds = sorted(set(eig_by_threshold) & set(baseline_by_threshold))
        delta_axis.plot(
            [
                baseline_by_threshold[threshold].average_questions
                - eig_by_threshold[threshold].average_questions
                for threshold in common_thresholds
            ],
            [
                eig_by_threshold[threshold].accuracy
                - baseline_by_threshold[threshold].accuracy
                for threshold in common_thresholds
            ],
            marker="o",
            markersize=4,
            color=color,
            label=f"noise={noise_rate:.0%}",
        )

    curve_axis.set_title("Accuracy–question trade-off")
    curve_axis.set_xlabel("Average additional questions")
    curve_axis.set_ylabel("Top-1 diagnosis accuracy")
    curve_axis.grid(alpha=0.25)
    curve_axis.legend(
        fontsize=8,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )

    delta_axis.axhline(0, color="black", linewidth=0.8)
    delta_axis.axvline(0, color="black", linewidth=0.8)
    delta_axis.set_title(
        f"Paired EIG improvement over "
        f"{display_names.get(baseline_strategy, baseline_strategy)}"
    )
    delta_axis.set_xlabel("Questions saved (baseline - EIG)")
    delta_axis.set_ylabel("Accuracy difference (EIG - baseline)")
    delta_axis.grid(alpha=0.25)
    delta_axis.legend(fontsize=8)
    fig.suptitle("DDXPlus independent test set: accuracy, turns, and answer noise")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _stop_at_threshold(
    trajectory: list[Mapping[str, float]], threshold: float
) -> tuple[int, Mapping[str, float]]:
    for turn_count, belief in enumerate(trajectory):
        if max(belief.values()) >= threshold:
            return turn_count, belief
    return len(trajectory) - 1, trajectory[-1]


def _stable_seed(seed: int, case_id: str, noise_rate: float) -> int:
    digest = hashlib.blake2b(
        f"{seed}|{case_id}|{noise_rate:.8f}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - margin, center + margin


def _bootstrap_mean_interval(
    values: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("bootstrap analysis requires numpy") from exc

    data = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    bootstrap_means: list[float] = []
    remaining = samples
    while remaining:
        batch_size = min(200, remaining)
        indices = generator.integers(
            0, len(data), size=(batch_size, len(data)), endpoint=False
        )
        bootstrap_means.extend(data[indices].mean(axis=1).tolist())
        remaining -= batch_size
    return (
        float(np.percentile(bootstrap_means, 2.5)),
        float(np.percentile(bootstrap_means, 97.5)),
    )


def _exact_mcnemar_p(first_only: int, second_only: int) -> float:
    """Two-sided exact conditional-binomial McNemar p-value."""

    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    log_probabilities = [
        math.lgamma(discordant + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(discordant - successes + 1)
        - discordant * math.log(2.0)
        for successes in range(tail + 1)
    ]
    maximum = max(log_probabilities)
    log_cdf = maximum + math.log(
        sum(math.exp(value - maximum) for value in log_probabilities)
    )
    return min(1.0, 2.0 * math.exp(log_cdf))
