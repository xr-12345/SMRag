"""Paired report-layer ablations on fixed DDXPlus dialogue trajectories."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Mapping, MutableSequence

from .belief import BeliefTracker
from .benchmark import (
    CaseOutcome,
    CurvePoint,
    _bootstrap_mean_interval,
    _exact_mcnemar_p,
    _stable_seed,
    _stop_at_threshold,
    _wilson_interval,
)
from .channel import (
    AnswerChannel,
    answer_channel_without_misreport,
    reliable_answer_channel,
)
from .estimation import DiseaseStateModel
from .gating import (
    HeuristicMisreportGate,
    HistoryReliabilityMisreportGate,
    LearnedMisreportGate,
    MisreportGate,
    OracleMisreportGate,
    SparseSurprisalMisreportGate,
)
from .questioning import NumpyQuestionSelector
from .schema import UNKNOWN, CertaintyCue, ClinicalCase, Observation, VariableSpec
from .simulator import (
    ObservationTransform,
    PatientProfile,
    StopRule,
    StructuredPatientSimulator,
    run_dialogue,
)


FULL_TWO_LAYER = "full_two_layer"
DIRECT_RELIABLE = "direct_reliable"
UNKNOWN_AS_NEGATIVE = "unknown_as_negative"
NO_MISREPORT = "no_misreport"
ADAPTIVE_HEURISTIC = "adaptive_heuristic"
ADAPTIVE_SPARSE = "adaptive_sparse"
ADAPTIVE_HISTORY = "adaptive_history"
ADAPTIVE_LEARNED = "adaptive_learned"
ORACLE_GATE = "oracle_gate"
ABLATION_VARIANTS = (
    FULL_TWO_LAYER,
    DIRECT_RELIABLE,
    UNKNOWN_AS_NEGATIVE,
    NO_MISREPORT,
)


@dataclass(frozen=True)
class AblationComparison:
    ablated_strategy: str
    noise_rate: float
    posterior_threshold: float
    cases_per_seed: int
    seeds: int
    full_accuracy: float
    ablated_accuracy: float
    accuracy_difference: float
    accuracy_difference_ci_low: float
    accuracy_difference_ci_high: float
    full_average_questions: float
    ablated_average_questions: float
    question_difference: float
    question_difference_ci_low: float
    question_difference_ci_high: float
    full_brier_score: float
    ablated_brier_score: float
    brier_improvement: float
    brier_improvement_ci_low: float
    brier_improvement_ci_high: float
    full_only_correct: int
    ablated_only_correct: int
    mcnemar_max_p: float


Runtime = tuple[
    NumpyQuestionSelector,
    AnswerChannel,
    ObservationTransform | None,
    MisreportGate | None,
]
_PROCESS_CONTEXT: tuple[
    DiseaseStateModel,
    Mapping[str, Runtime],
    tuple[float, ...],
    int,
    int,
] | None = None


def unknown_or_uncertain_as_negative(
    observation: Observation, spec: VariableSpec
) -> Observation:
    """Naive missing-is-negative ablation used only at inference time."""

    if observation.value != UNKNOWN and observation.certainty != CertaintyCue.UNCERTAIN:
        return replace(observation, certainty=CertaintyCue.CERTAIN)
    negative = "absent" if "absent" in spec.values else spec.values[0]
    return replace(
        observation,
        value=negative,
        certainty=CertaintyCue.CERTAIN,
    )


def build_ablation_runtime(
    name: str, *, learned_gate: LearnedMisreportGate | None = None
) -> Runtime:
    if name == FULL_TWO_LAYER:
        return NumpyQuestionSelector(), AnswerChannel(), None, None
    if name == DIRECT_RELIABLE:
        return NumpyQuestionSelector(), reliable_answer_channel(), None, None
    if name == UNKNOWN_AS_NEGATIVE:
        return (
            NumpyQuestionSelector(),
            reliable_answer_channel(),
            unknown_or_uncertain_as_negative,
            None,
        )
    if name == NO_MISREPORT:
        return NumpyQuestionSelector(), answer_channel_without_misreport(), None, None
    if name == ADAPTIVE_HEURISTIC:
        return (
            NumpyQuestionSelector(),
            answer_channel_without_misreport(),
            None,
            HeuristicMisreportGate(),
        )
    if name == ADAPTIVE_SPARSE:
        return (
            NumpyQuestionSelector(),
            answer_channel_without_misreport(),
            None,
            SparseSurprisalMisreportGate(),
        )
    if name == ADAPTIVE_HISTORY:
        return (
            NumpyQuestionSelector(),
            answer_channel_without_misreport(),
            None,
            HistoryReliabilityMisreportGate(),
        )
    if name == ADAPTIVE_LEARNED:
        if learned_gate is None:
            raise ValueError("adaptive_learned requires a fitted learned gate")
        return (
            NumpyQuestionSelector(),
            answer_channel_without_misreport(),
            None,
            learned_gate,
        )
    if name == ORACLE_GATE:
        return (
            NumpyQuestionSelector(),
            answer_channel_without_misreport(),
            None,
            OracleMisreportGate(),
        )
    raise ValueError(f"unknown ablation strategy: {name}")


def run_ablation_curve_experiment(
    model: DiseaseStateModel,
    cases: Iterable[ClinicalCase],
    *,
    variants: tuple[str, ...] = ABLATION_VARIANTS,
    noise_rates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3),
    posterior_thresholds: tuple[float, ...] = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
    max_questions: int = 15,
    seed: int = 2026,
    progress: bool = False,
    workers: int = 1,
    executor_type: str = "thread",
    raw_outcomes: MutableSequence[CaseOutcome] | None = None,
    learned_gate: LearnedMisreportGate | None = None,
) -> list[CurvePoint]:
    """Compare inference ablations under identical feature-indexed reports."""

    case_list = list(cases)
    if not case_list:
        raise ValueError("ablation needs at least one case")
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("ablation variants must be non-empty and unique")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if executor_type not in {"thread", "process"}:
        raise ValueError("executor_type must be 'thread' or 'process'")
    for case in case_list:
        if case.diagnosis not in model.diseases:
            raise ValueError(f"test disease missing from model: {case.diagnosis}")

    runtimes = _make_runtimes(model, variants, learned_gate=learned_gate)
    tasks = [
        (case, noise_rate)
        for noise_rate in noise_rates
        for case in case_list
    ]

    def evaluate(task: tuple[ClinicalCase, float]) -> list[CaseOutcome]:
        return _evaluate_case_noise(
            task,
            model=model,
            runtimes=runtimes,
            posterior_thresholds=posterior_thresholds,
            max_questions=max_questions,
            seed=seed,
        )

    if workers == 1:
        executor = None
        results = map(evaluate, tasks)
    elif executor_type == "process":
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_process_worker,
            initargs=(
                model,
                variants,
                learned_gate,
                posterior_thresholds,
                max_questions,
                seed,
            ),
        )
        results = executor.map(
            _process_evaluate,
            tasks,
            chunksize=max(1, len(tasks) // (workers * 20)),
        )
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(evaluate, tasks)

    observations: dict[tuple[str, float, float], list[tuple[int, int, float]]] = (
        defaultdict(list)
    )
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
                print(
                    f"completed {task_index}/{len(tasks)} ablation case-noise tasks",
                    flush=True,
                )
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


def _make_runtimes(
    model: DiseaseStateModel,
    variants: tuple[str, ...],
    *,
    learned_gate: LearnedMisreportGate | None = None,
) -> dict[str, Runtime]:
    runtimes = {
        name: build_ablation_runtime(name, learned_gate=learned_gate)
        for name in variants
    }
    for selector, channel, _, gate in runtimes.values():
        selector._ensure_numpy_cache(
            BeliefTracker(model, channel, misreport_gate=gate)
        )
    return runtimes


def _evaluate_case_noise(
    task: tuple[ClinicalCase, float],
    *,
    model: DiseaseStateModel,
    runtimes: Mapping[str, Runtime],
    posterior_thresholds: tuple[float, ...],
    max_questions: int,
    seed: int,
) -> list[CaseOutcome]:
    case, noise_rate = task
    profile = PatientProfile.from_noise_rate(noise_rate)
    paired_seed = _stable_seed(seed, case.case_id, noise_rate)
    rows: list[CaseOutcome] = []
    for strategy, (selector, inference_channel, transform, gate) in runtimes.items():
        # Patient generation always uses the full channel.  Only inference is ablated.
        patient = StructuredPatientSimulator(
            diagnosis=case.diagnosis,
            latent_states=case.states,
            model=model,
            profile=profile,
            seed=paired_seed,
        )
        tracker = BeliefTracker(
            model,
            inference_channel,
            misreport_gate=gate,
        )
        for initial in case.initial_observations:
            interpreted = (
                transform(initial, model.specs[initial.key])
                if transform is not None
                else initial
            )
            tracker.update(interpreted)
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
            observation_transform=transform,
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
            gate_priors = [
                turn.update.misreport_prior for turn in result.turns[:turn_count]
            ]
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
                    mean_misreport_prior=(
                        fmean(gate_priors) if gate_priors else 0.0
                    ),
                    max_misreport_prior=max(gate_priors, default=0.0),
                    gate_activations=(
                        sum(
                            prior > gate.activation_threshold
                            for prior in gate_priors
                        )
                        if gate is not None
                        else 0
                    ),
                )
            )
    return rows


def _initialize_process_worker(
    model: DiseaseStateModel,
    variants: tuple[str, ...],
    learned_gate: LearnedMisreportGate | None,
    posterior_thresholds: tuple[float, ...],
    max_questions: int,
    seed: int,
) -> None:
    global _PROCESS_CONTEXT
    _PROCESS_CONTEXT = (
        model,
        _make_runtimes(model, variants, learned_gate=learned_gate),
        posterior_thresholds,
        max_questions,
        seed,
    )


def _process_evaluate(task: tuple[ClinicalCase, float]) -> list[CaseOutcome]:
    if _PROCESS_CONTEXT is None:
        raise RuntimeError("ablation process worker was not initialized")
    model, runtimes, thresholds, max_questions, seed = _PROCESS_CONTEXT
    return _evaluate_case_noise(
        task,
        model=model,
        runtimes=runtimes,
        posterior_thresholds=thresholds,
        max_questions=max_questions,
        seed=seed,
    )


def summarize_ablation_comparisons(
    outcomes: Iterable[CaseOutcome],
    *,
    full_strategy: str = FULL_TWO_LAYER,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 2026,
) -> list[AblationComparison]:
    """Paired, case-clustered full-vs-ablated comparisons."""

    index: dict[tuple[int, str, float, float, str], CaseOutcome] = {}
    strategies: set[str] = set()
    for row in outcomes:
        key = (
            row.seed,
            row.case_id,
            row.noise_rate,
            row.posterior_threshold,
            row.strategy,
        )
        if key in index:
            raise ValueError(f"duplicate ablation outcome: {key}")
        index[key] = row
        strategies.add(row.strategy)
    if full_strategy not in strategies:
        raise ValueError(f"missing full strategy {full_strategy}")

    conditions = sorted(
        {
            (row.noise_rate, row.posterior_threshold)
            for row in index.values()
            if row.strategy == full_strategy
        }
    )
    comparisons: list[AblationComparison] = []
    comparison_index = 0
    for ablated_strategy in sorted(strategies - {full_strategy}):
        for noise_rate, threshold in conditions:
            pairs_by_seed: dict[int, list[tuple[CaseOutcome, CaseOutcome]]] = (
                defaultdict(list)
            )
            for key, full in index.items():
                seed, case_id, row_noise, row_threshold, strategy = key
                if (
                    strategy != full_strategy
                    or row_noise != noise_rate
                    or row_threshold != threshold
                ):
                    continue
                ablated_key = (
                    seed,
                    case_id,
                    noise_rate,
                    threshold,
                    ablated_strategy,
                )
                if ablated_key not in index:
                    raise ValueError(f"missing paired ablation outcome: {ablated_key}")
                pairs_by_seed[seed].append((full, index[ablated_key]))
            case_counts = {len(pairs) for pairs in pairs_by_seed.values()}
            if not pairs_by_seed or len(case_counts) != 1:
                raise ValueError("ablation seeds have inconsistent case counts")

            accuracy_by_case: dict[str, list[float]] = defaultdict(list)
            questions_by_case: dict[str, list[float]] = defaultdict(list)
            brier_by_case: dict[str, list[float]] = defaultdict(list)
            full_accuracy_by_seed: list[float] = []
            ablated_accuracy_by_seed: list[float] = []
            full_questions_by_seed: list[float] = []
            ablated_questions_by_seed: list[float] = []
            full_brier_by_seed: list[float] = []
            ablated_brier_by_seed: list[float] = []
            full_only_total = 0
            ablated_only_total = 0
            p_values: list[float] = []
            expected_case_ids: set[str] | None = None
            for seed, pairs in sorted(pairs_by_seed.items()):
                case_ids = {full.case_id for full, _ in pairs}
                if expected_case_ids is None:
                    expected_case_ids = case_ids
                elif case_ids != expected_case_ids:
                    raise ValueError("ablation case IDs differ across seeds")
                full_accuracy_by_seed.append(fmean(full.correct for full, _ in pairs))
                ablated_accuracy_by_seed.append(
                    fmean(ablated.correct for _, ablated in pairs)
                )
                full_questions_by_seed.append(fmean(full.questions for full, _ in pairs))
                ablated_questions_by_seed.append(
                    fmean(ablated.questions for _, ablated in pairs)
                )
                full_brier_by_seed.append(
                    fmean(full.brier_score for full, _ in pairs)
                )
                ablated_brier_by_seed.append(
                    fmean(ablated.brier_score for _, ablated in pairs)
                )
                full_only = sum(
                    full.correct == 1 and ablated.correct == 0
                    for full, ablated in pairs
                )
                ablated_only = sum(
                    full.correct == 0 and ablated.correct == 1
                    for full, ablated in pairs
                )
                full_only_total += full_only
                ablated_only_total += ablated_only
                p_values.append(_exact_mcnemar_p(full_only, ablated_only))
                for full, ablated in pairs:
                    accuracy_by_case[full.case_id].append(
                        full.correct - ablated.correct
                    )
                    questions_by_case[full.case_id].append(
                        full.questions - ablated.questions
                    )
                    brier_by_case[full.case_id].append(
                        ablated.brier_score - full.brier_score
                    )

            accuracy_differences = [
                fmean(values) for _, values in sorted(accuracy_by_case.items())
            ]
            question_differences = [
                fmean(values) for _, values in sorted(questions_by_case.items())
            ]
            brier_improvements = [
                fmean(values) for _, values in sorted(brier_by_case.items())
            ]
            accuracy_low, accuracy_high = _bootstrap_mean_interval(
                accuracy_differences,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 3 * comparison_index,
            )
            questions_low, questions_high = _bootstrap_mean_interval(
                question_differences,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 3 * comparison_index + 1,
            )
            brier_low, brier_high = _bootstrap_mean_interval(
                brier_improvements,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 3 * comparison_index + 2,
            )
            comparison_index += 1
            comparisons.append(
                AblationComparison(
                    ablated_strategy=ablated_strategy,
                    noise_rate=noise_rate,
                    posterior_threshold=threshold,
                    cases_per_seed=next(iter(case_counts)),
                    seeds=len(pairs_by_seed),
                    full_accuracy=fmean(full_accuracy_by_seed),
                    ablated_accuracy=fmean(ablated_accuracy_by_seed),
                    accuracy_difference=fmean(accuracy_differences),
                    accuracy_difference_ci_low=accuracy_low,
                    accuracy_difference_ci_high=accuracy_high,
                    full_average_questions=fmean(full_questions_by_seed),
                    ablated_average_questions=fmean(ablated_questions_by_seed),
                    question_difference=fmean(question_differences),
                    question_difference_ci_low=questions_low,
                    question_difference_ci_high=questions_high,
                    full_brier_score=fmean(full_brier_by_seed),
                    ablated_brier_score=fmean(ablated_brier_by_seed),
                    brier_improvement=fmean(brier_improvements),
                    brier_improvement_ci_low=brier_low,
                    brier_improvement_ci_high=brier_high,
                    full_only_correct=full_only_total,
                    ablated_only_correct=ablated_only_total,
                    mcnemar_max_p=max(p_values),
                )
            )
    return comparisons


def save_ablation_comparisons_csv(
    comparisons: Iterable[AblationComparison], path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(AblationComparison.__dataclass_fields__)
        )
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow(comparison.__dict__)


def plot_ablation_curves(points: Iterable[CurvePoint], path: str | Path) -> None:
    """Plot one accuracy-question panel per answer-noise level."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("ablation plotting requires matplotlib") from exc

    point_list = list(points)
    noise_rates = sorted({point.noise_rate for point in point_list})
    if len(noise_rates) > 4:
        raise ValueError("ablation plot supports at most four noise rates")
    colors = {
        FULL_TWO_LAYER: "#6a1b9a",
        ADAPTIVE_HEURISTIC: "#ff7f0e",
        ADAPTIVE_SPARSE: "#17becf",
        ADAPTIVE_HISTORY: "#8c564b",
        DIRECT_RELIABLE: "#1f77b4",
        UNKNOWN_AS_NEGATIVE: "#d62728",
        NO_MISREPORT: "#2ca02c",
    }
    labels = {
        FULL_TWO_LAYER: "Full two-layer",
        ADAPTIVE_HEURISTIC: "Adaptive heuristic",
        ADAPTIVE_SPARSE: "Adaptive sparse",
        ADAPTIVE_HISTORY: "Adaptive history",
        DIRECT_RELIABLE: "Direct/reliable",
        UNKNOWN_AS_NEGATIVE: "Unknown→negative",
        NO_MISREPORT: "No misreport state",
    }
    markers = {
        FULL_TWO_LAYER: "o",
        ADAPTIVE_HEURISTIC: "D",
        ADAPTIVE_SPARSE: "P",
        ADAPTIVE_HISTORY: "v",
        DIRECT_RELIABLE: "s",
        UNKNOWN_AS_NEGATIVE: "X",
        NO_MISREPORT: "^",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), sharex=False, sharey=True)
    flat_axes = list(axes.flat)
    strategies = tuple(sorted({point.strategy for point in point_list}))
    for axis_index, axis in enumerate(flat_axes):
        if axis_index >= len(noise_rates):
            axis.set_visible(False)
            continue
        noise_rate = noise_rates[axis_index]
        for strategy in strategies:
            subset = sorted(
                (
                    point
                    for point in point_list
                    if point.noise_rate == noise_rate and point.strategy == strategy
                ),
                key=lambda point: point.posterior_threshold,
            )
            axis.plot(
                [point.average_questions for point in subset],
                [point.accuracy for point in subset],
                color=colors.get(strategy),
                marker=markers.get(strategy, "x"),
                linewidth=1.8,
                markersize=4.5,
                label=labels.get(strategy, strategy),
            )
        axis.set_title(f"Latent unreliable-mode mass: {noise_rate:.0%}")
        axis.set_xlabel("Average additional questions")
        axis.set_ylabel("Top-1 diagnosis accuracy")
        axis.grid(alpha=0.25)
    handles, legend_labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("DDXPlus report-layer ablation: accuracy–question trade-off")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
