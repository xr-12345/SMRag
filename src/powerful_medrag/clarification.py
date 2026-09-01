"""PaMis-style anomaly detection followed by controlled clarification.

This is a structured DDXPlus proxy, not a reproduction of PaMis's dialogue
entity graph or natural-language question generator.  A surprising answer
triggers a repeated, independently sampled report about the same evidence.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Mapping, MutableSequence

from .belief import BeliefTracker
from .benchmark import (
    CaseOutcome,
    CurvePoint,
    _stable_seed,
    _wilson_interval,
)
from .channel import AnswerChannel, ReportMode, answer_channel_without_misreport
from .estimation import DiseaseStateModel
from .gating import HeuristicMisreportGate
from .questioning import NumpyQuestionSelector
from .schema import UNKNOWN, CertaintyCue, ClinicalCase, Observation
from .simulator import PatientProfile, StructuredPatientSimulator


PAMIS_STYLE_S25 = "pamis_style_s25"
PAMIS_STYLE_S30 = "pamis_style_s30"
PAMIS_STYLE_S35 = "pamis_style_s35"
PAMIS_STYLE_S40 = "pamis_style_s40"
RETRO_RISK_R50_B1 = "retro_risk_r50_b1"
RETRO_UTILITY_U010_B1 = "retro_utility_u010_b1"
RETRO_UTILITY_U025_B1 = "retro_utility_u025_b1"
RETRO_UTILITY_U050_B1 = "retro_utility_u050_b1"
RETRO_UTILITY_U025_B2 = "retro_utility_u025_b2"
HYBRID_U025_B1 = "hybrid_s30_u025_b1"
HYBRID_U025_B2 = "hybrid_s30_u025_b2"
PAMIS_STYLE_VARIANTS = (
    PAMIS_STYLE_S25,
    PAMIS_STYLE_S30,
    PAMIS_STYLE_S35,
    PAMIS_STYLE_S40,
)
RETROSPECTIVE_VARIANTS = (
    RETRO_RISK_R50_B1,
    RETRO_UTILITY_U010_B1,
    RETRO_UTILITY_U025_B1,
    RETRO_UTILITY_U050_B1,
    RETRO_UTILITY_U025_B2,
    HYBRID_U025_B1,
    HYBRID_U025_B2,
)
_SURPRISAL_THRESHOLDS = {
    PAMIS_STYLE_S25: 2.5,
    PAMIS_STYLE_S30: 3.0,
    PAMIS_STYLE_S35: 3.5,
    PAMIS_STYLE_S40: 4.0,
}


@dataclass(frozen=True)
class SurprisalClarificationProtocol:
    """Detect a locally anomalous report and request one confirmation."""

    surprisal_threshold: float = 3.0

    def __post_init__(self) -> None:
        if self.surprisal_threshold < 0:
            raise ValueError("surprisal threshold cannot be negative")

    def score(self, tracker: BeliefTracker, observation: Observation) -> float:
        probability = HeuristicMisreportGate.predictive_probability(
            tracker, observation
        )
        return -math.log(min(max(probability, 1e-12), 1.0))

    def should_clarify(
        self, tracker: BeliefTracker, observation: Observation
    ) -> bool:
        return (
            observation.value != UNKNOWN
            and observation.certainty == CertaintyCue.CERTAIN
            and self.score(tracker, observation) >= self.surprisal_threshold
        )

    @staticmethod
    def resolve(
        original: Observation, clarification: Observation
    ) -> Observation:
        """Keep one agreed report; abstain when the two reports disagree."""

        if original.key != clarification.key:
            raise ValueError("clarification must concern the same feature")
        if original.value != clarification.value:
            return Observation(key=original.key, value=UNKNOWN)
        if original.value == UNKNOWN:
            return Observation(key=original.key, value=UNKNOWN)
        certainty = (
            CertaintyCue.CERTAIN
            if original.certainty == clarification.certainty == CertaintyCue.CERTAIN
            else CertaintyCue.UNCERTAIN
        )
        return Observation(key=original.key, value=original.value, certainty=certainty)


@dataclass(frozen=True)
class RetrospectiveScore:
    report_index: int
    error_probability: float
    diagnostic_influence: float
    score: float
    retrieval_impact: float = 0.0


@dataclass(frozen=True)
class RetrospectiveClarificationProtocol:
    """Rank past reports using leave-one-out error risk and influence."""

    score_threshold: float
    max_clarifications: int = 1
    use_diagnostic_influence: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("retrospective score threshold must be in [0, 1]")
        if self.max_clarifications <= 0:
            raise ValueError("retrospective clarification budget must be positive")

    def rank(
        self,
        model: DiseaseStateModel,
        initial_observations: tuple[Observation, ...],
        reports: tuple[Observation, ...],
        *,
        excluded: set[int] | None = None,
    ) -> list[RetrospectiveScore]:
        """Score reports without using the simulated noise rate or true state."""

        excluded_indices = excluded or set()
        full_tracker = _replay_tracker(model, initial_observations, reports)
        detector_channel = AnswerChannel()
        inference_channel = answer_channel_without_misreport()
        report_key_counts: dict[object, int] = defaultdict(int)
        for report in reports:
            report_key_counts[report.key] += 1
        initial_keys = {observation.key for observation in initial_observations}
        scores: list[RetrospectiveScore] = []
        for index, report in enumerate(reports):
            if index in excluded_indices or report.value == UNKNOWN:
                continue
            if report_key_counts[report.key] == 1 and report.key not in initial_keys:
                likelihoods = _independent_observation_likelihoods(
                    model,
                    report,
                    inference_channel,
                )
                leave_one_out_belief = BeliefTracker._normalize(
                    {
                        disease: full_tracker.belief[disease] / likelihoods[disease]
                        for disease in model.diseases
                    }
                )
            else:
                leave_one_out_belief = _replay_tracker(
                    model,
                    initial_observations,
                    reports[:index] + reports[index + 1 :],
                ).belief
            error_probability = _report_error_probability(
                model,
                leave_one_out_belief,
                report,
                detector_channel,
            )
            influence = 0.5 * sum(
                abs(full_tracker.belief[disease] - leave_one_out_belief[disease])
                for disease in model.diseases
            )
            score = (
                error_probability * influence
                if self.use_diagnostic_influence
                else error_probability
            )
            scores.append(
                RetrospectiveScore(
                    report_index=index,
                    error_probability=error_probability,
                    diagnostic_influence=influence,
                    score=score,
                )
            )
        return sorted(scores, key=lambda item: item.score, reverse=True)


@dataclass(frozen=True)
class ClarificationRuntime:
    selector: NumpyQuestionSelector
    online: SurprisalClarificationProtocol | None = None
    retrospective: RetrospectiveClarificationProtocol | None = None


@dataclass(frozen=True)
class ClarificationSummary:
    strategy: str
    noise_rate: float
    posterior_threshold: float
    cases: int
    average_primary_questions: float
    average_clarifications: float
    clarification_rate: float
    harmful_reports: int
    detection_precision: float | None
    detection_recall: float | None
    mitigation_rate: float | None
    discarded_correct_reports_per_case: float


@dataclass(frozen=True)
class _ClarificationEvent:
    completion_turn: int
    clarified: bool
    harmful: bool
    false_clarification: bool
    mitigated_harmful: bool
    discarded_correct: bool


@dataclass(frozen=True)
class _TrajectorySnapshot:
    turns: int
    belief: Mapping[str, float]
    events: int


_PROCESS_CONTEXT: tuple[
    DiseaseStateModel,
    Mapping[str, ClarificationRuntime],
    tuple[float, ...],
    int,
    int,
] | None = None


def build_clarification_runtime(name: str) -> ClarificationRuntime:
    if name in _SURPRISAL_THRESHOLDS:
        return ClarificationRuntime(
            selector=NumpyQuestionSelector(),
            online=SurprisalClarificationProtocol(_SURPRISAL_THRESHOLDS[name]),
        )
    configurations: dict[
        str,
        tuple[float, int, bool, bool],
    ] = {
        RETRO_RISK_R50_B1: (0.50, 1, False, False),
        RETRO_UTILITY_U010_B1: (0.010, 1, True, False),
        RETRO_UTILITY_U025_B1: (0.025, 1, True, False),
        RETRO_UTILITY_U050_B1: (0.050, 1, True, False),
        RETRO_UTILITY_U025_B2: (0.025, 2, True, False),
        HYBRID_U025_B1: (0.025, 1, True, True),
        HYBRID_U025_B2: (0.025, 2, True, True),
    }
    try:
        threshold, budget, use_influence, hybrid = configurations[name]
    except KeyError as exc:
        raise ValueError(f"unknown clarification strategy: {name}") from exc
    return ClarificationRuntime(
        selector=NumpyQuestionSelector(),
        online=(SurprisalClarificationProtocol(3.0) if hybrid else None),
        retrospective=RetrospectiveClarificationProtocol(
            score_threshold=threshold,
            max_clarifications=budget,
            use_diagnostic_influence=use_influence,
        ),
    )


def run_clarification_curve_experiment(
    model: DiseaseStateModel,
    cases: Iterable[ClinicalCase],
    *,
    variants: tuple[str, ...] = PAMIS_STYLE_VARIANTS,
    noise_rates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3),
    posterior_thresholds: tuple[float, ...] = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
    max_questions: int = 15,
    seed: int = 2026,
    progress: bool = False,
    workers: int = 1,
    executor_type: str = "thread",
    raw_outcomes: MutableSequence[CaseOutcome] | None = None,
) -> list[CurvePoint]:
    """Evaluate clarification thresholds on paired structured dialogues."""

    case_list = list(cases)
    if not case_list:
        raise ValueError("clarification experiment needs at least one case")
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("clarification variants must be non-empty and unique")
    if not noise_rates or not posterior_thresholds:
        raise ValueError("noise rates and posterior thresholds cannot be empty")
    if max_questions <= 0 or workers <= 0:
        raise ValueError("max_questions and workers must be positive")
    if executor_type not in {"thread", "process"}:
        raise ValueError("executor_type must be 'thread' or 'process'")
    for case in case_list:
        if case.diagnosis not in model.diseases:
            raise ValueError(f"validation disease missing from model: {case.diagnosis}")

    runtimes = _make_runtimes(model, variants)
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
            initargs=(model, variants, posterior_thresholds, max_questions, seed),
        )
        results = executor.map(
            _process_evaluate,
            tasks,
            chunksize=max(1, len(tasks) // (workers * 20)),
        )
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(evaluate, tasks)

    observations: dict[tuple[str, float, float], list[CaseOutcome]] = defaultdict(list)
    try:
        progress_interval = max(1, len(tasks) // 20)
        for task_index, rows in enumerate(results, start=1):
            if raw_outcomes is not None:
                raw_outcomes.extend(rows)
            for row in rows:
                observations[(row.strategy, row.noise_rate, row.posterior_threshold)].append(row)
            if progress and (task_index % progress_interval == 0 or task_index == len(tasks)):
                print(
                    f"completed {task_index}/{len(tasks)} clarification case-noise tasks",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()

    points: list[CurvePoint] = []
    for (strategy, noise_rate, threshold), rows in sorted(observations.items()):
        correct_values = [row.correct for row in rows]
        question_values = [row.questions for row in rows]
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
                accuracy=fmean(correct_values),
                accuracy_ci_low=ci_low,
                accuracy_ci_high=ci_high,
                average_questions=fmean(question_values),
                questions_standard_error=question_se,
                brier_score=fmean(row.brier_score for row in rows),
            )
        )
    return points


def _make_runtimes(
    model: DiseaseStateModel, variants: tuple[str, ...]
) -> dict[str, ClarificationRuntime]:
    channel = answer_channel_without_misreport()
    runtimes = {name: build_clarification_runtime(name) for name in variants}
    for runtime in runtimes.values():
        runtime.selector._ensure_numpy_cache(BeliefTracker(model, channel))
    return runtimes


def _evaluate_case_noise(
    task: tuple[ClinicalCase, float],
    *,
    model: DiseaseStateModel,
    runtimes: Mapping[str, ClarificationRuntime],
    posterior_thresholds: tuple[float, ...],
    max_questions: int,
    seed: int,
) -> list[CaseOutcome]:
    case, noise_rate = task
    paired_seed = _stable_seed(seed, case.case_id, noise_rate)
    profile = PatientProfile.from_noise_rate(noise_rate)
    rows: list[CaseOutcome] = []
    for strategy, runtime in runtimes.items():
        if runtime.retrospective is None:
            rows.extend(
                _run_online_curve(
                    case,
                    model=model,
                    profile=profile,
                    paired_seed=paired_seed,
                    seed=seed,
                    strategy=strategy,
                    runtime=runtime,
                    posterior_thresholds=posterior_thresholds,
                    max_questions=max_questions,
                    noise_rate=noise_rate,
                )
            )
            continue
        for threshold in posterior_thresholds:
            rows.append(
                _run_budgeted_retrospective_dialogue(
                    case,
                    model=model,
                    profile=profile,
                    paired_seed=paired_seed,
                    seed=seed,
                    strategy=strategy,
                    runtime=runtime,
                    posterior_threshold=threshold,
                    max_questions=max_questions,
                    noise_rate=noise_rate,
                )
            )
    return rows


def _run_online_curve(
    case: ClinicalCase,
    *,
    model: DiseaseStateModel,
    profile: PatientProfile,
    paired_seed: int,
    seed: int,
    strategy: str,
    runtime: ClarificationRuntime,
    posterior_thresholds: tuple[float, ...],
    max_questions: int,
    noise_rate: float,
) -> list[CaseOutcome]:
    if runtime.online is None:
        raise ValueError("online curve needs an online clarification protocol")
    patient = StructuredPatientSimulator(
        diagnosis=case.diagnosis,
        latent_states=case.states,
        model=model,
        profile=profile,
        seed=paired_seed,
    )
    tracker = _replay_tracker(model, case.initial_observations, ())
    asked = {observation.key for observation in case.initial_observations}
    events: list[_ClarificationEvent] = []
    snapshots = [_TrajectorySnapshot(turns=0, belief=dict(tracker.belief), events=0)]
    turns = 0
    while turns < max_questions:
        ranking = runtime.selector.rank(tracker, excluded=asked)
        if not ranking:
            break
        key = ranking[0].key
        original, original_mode = patient.answer(key)
        turns += 1
        event, resolved = _resolve_report_event(
            patient,
            original,
            original_mode,
            turns=turns,
            clarify=(
                turns < max_questions
                and runtime.online.should_clarify(tracker, original)
            ),
        )
        if event.clarified:
            turns += 1
            event = replace(event, completion_turn=turns)
        tracker.update(resolved)
        asked.add(key)
        events.append(event)
        snapshots.append(
            _TrajectorySnapshot(
                turns=turns,
                belief=dict(tracker.belief),
                events=len(events),
            )
        )

    rows: list[CaseOutcome] = []
    for threshold in posterior_thresholds:
        snapshot = _stop_at_threshold(snapshots, threshold)
        rows.append(
            _case_outcome(
                case,
                seed=seed,
                strategy=strategy,
                noise_rate=noise_rate,
                posterior_threshold=threshold,
                belief=snapshot.belief,
                turns=snapshot.turns,
                events=events[: snapshot.events],
                diseases=model.diseases,
            )
        )
    return rows


def _run_budgeted_retrospective_dialogue(
    case: ClinicalCase,
    *,
    model: DiseaseStateModel,
    profile: PatientProfile,
    paired_seed: int,
    seed: int,
    strategy: str,
    runtime: ClarificationRuntime,
    posterior_threshold: float,
    max_questions: int,
    noise_rate: float,
) -> CaseOutcome:
    protocol = runtime.retrospective
    if protocol is None:
        raise ValueError("retrospective dialogue needs a retrospective protocol")
    patient = StructuredPatientSimulator(
        diagnosis=case.diagnosis,
        latent_states=case.states,
        model=model,
        profile=profile,
        seed=paired_seed,
    )
    reports: list[Observation] = []
    events: list[_ClarificationEvent] = []
    audited: set[int] = set()
    tracker = _replay_tracker(model, case.initial_observations, ())
    asked = {observation.key for observation in case.initial_observations}
    turns = 0
    retrospective_used = 0

    while turns < max_questions:
        top_probability = max(tracker.belief.values())
        remaining_audits = protocol.max_clarifications - retrospective_used
        audit_window = (
            top_probability >= posterior_threshold
            or (remaining_audits > 0 and turns >= max_questions - remaining_audits)
        )
        if audit_window and remaining_audits > 0:
            excluded = audited | {
                index for index, event in enumerate(events) if event.clarified
            }
            scores = protocol.rank(
                model,
                case.initial_observations,
                tuple(reports),
                excluded=excluded,
            )
            if scores and scores[0].score >= protocol.score_threshold:
                target = scores[0].report_index
                current = reports[target]
                clarification, _ = patient.answer(current.key)
                turns += 1
                resolved = SurprisalClarificationProtocol.resolve(
                    current,
                    clarification,
                )
                true_state = patient.latent_states.get(current.key, UNKNOWN)
                original_wrong = (
                    true_state != UNKNOWN
                    and current.value != UNKNOWN
                    and current.value != true_state
                )
                resolved_wrong = (
                    true_state != UNKNOWN
                    and resolved.value != UNKNOWN
                    and resolved.value != true_state
                )
                event = events[target]
                events[target] = replace(
                    event,
                    completion_turn=turns,
                    clarified=True,
                    false_clarification=not event.harmful,
                    mitigated_harmful=event.harmful and not resolved_wrong,
                    discarded_correct=(
                        not original_wrong
                        and current.value != UNKNOWN
                        and resolved.value == UNKNOWN
                    ),
                )
                reports[target] = resolved
                audited.add(target)
                retrospective_used += 1
                tracker = _replay_tracker(
                    model,
                    case.initial_observations,
                    tuple(reports),
                )
                continue

        if top_probability >= posterior_threshold:
            break
        ranking = runtime.selector.rank(tracker, excluded=asked)
        if not ranking:
            break
        key = ranking[0].key
        original, original_mode = patient.answer(key)
        turns += 1
        clarify_online = (
            runtime.online is not None
            and turns < max_questions
            and runtime.online.should_clarify(tracker, original)
        )
        event, resolved = _resolve_report_event(
            patient,
            original,
            original_mode,
            turns=turns,
            clarify=clarify_online,
        )
        if event.clarified:
            turns += 1
            event = replace(event, completion_turn=turns)
        reports.append(resolved)
        events.append(event)
        tracker.update(resolved)
        asked.add(key)

    return _case_outcome(
        case,
        seed=seed,
        strategy=strategy,
        noise_rate=noise_rate,
        posterior_threshold=posterior_threshold,
        belief=tracker.belief,
        turns=turns,
        events=events,
        diseases=model.diseases,
    )


def _resolve_report_event(
    patient: StructuredPatientSimulator,
    original: Observation,
    original_mode: ReportMode,
    *,
    turns: int,
    clarify: bool,
) -> tuple[_ClarificationEvent, Observation]:
    true_state = patient.latent_states.get(original.key, UNKNOWN)
    original_wrong = (
        true_state != UNKNOWN
        and original.value != UNKNOWN
        and original.value != true_state
    )
    harmful = original_mode == ReportMode.MISREPORTED and original_wrong
    resolved = original
    if clarify:
        clarification, _ = patient.answer(original.key)
        resolved = SurprisalClarificationProtocol.resolve(original, clarification)
    resolved_wrong = (
        true_state != UNKNOWN
        and resolved.value != UNKNOWN
        and resolved.value != true_state
    )
    return (
        _ClarificationEvent(
            completion_turn=turns,
            clarified=clarify,
            harmful=harmful,
            false_clarification=clarify and not harmful,
            mitigated_harmful=clarify and harmful and not resolved_wrong,
            discarded_correct=(
                clarify
                and not original_wrong
                and original.value != UNKNOWN
                and resolved.value == UNKNOWN
            ),
        ),
        resolved,
    )


def _case_outcome(
    case: ClinicalCase,
    *,
    seed: int,
    strategy: str,
    noise_rate: float,
    posterior_threshold: float,
    belief: Mapping[str, float],
    turns: int,
    events: Iterable[_ClarificationEvent],
    diseases: tuple[str, ...],
) -> CaseOutcome:
    event_list = list(events)
    predicted = max(belief, key=belief.__getitem__)
    brier = sum(
        (belief[disease] - (1.0 if disease == case.diagnosis else 0.0)) ** 2
        for disease in diseases
    )
    clarifications = sum(event.clarified for event in event_list)
    return CaseOutcome(
        seed=seed,
        case_id=case.case_id,
        true_diagnosis=case.diagnosis,
        strategy=strategy,
        noise_rate=noise_rate,
        posterior_threshold=posterior_threshold,
        predicted_diagnosis=predicted,
        correct=int(predicted == case.diagnosis),
        questions=turns,
        brier_score=brier,
        gate_activations=clarifications,
        clarifications=clarifications,
        harmful_reports=sum(event.harmful for event in event_list),
        detected_harmful_reports=sum(
            event.clarified and event.harmful for event in event_list
        ),
        false_clarifications=sum(
            event.false_clarification for event in event_list
        ),
        mitigated_harmful_reports=sum(
            event.mitigated_harmful for event in event_list
        ),
        discarded_correct_reports=sum(
            event.discarded_correct for event in event_list
        ),
    )


def _replay_tracker(
    model: DiseaseStateModel,
    initial_observations: Iterable[Observation],
    reports: Iterable[Observation],
) -> BeliefTracker:
    tracker = BeliefTracker(model, answer_channel_without_misreport())
    for observation in initial_observations:
        tracker.update(observation)
    for report in reports:
        tracker.update(report)
    return tracker


def _report_error_probability(
    model: DiseaseStateModel,
    leave_one_out_belief: Mapping[str, float],
    report: Observation,
    detector_channel: AnswerChannel,
) -> float:
    if report.value == UNKNOWN:
        return 0.0
    spec = model.specs[report.key]
    predictive_states = {
        state: sum(
            leave_one_out_belief[disease]
            * model.state_distribution(disease, report.key)[state]
            for disease in model.diseases
        )
        for state in spec.values
    }
    state_weights = {
        state: probability
        * detector_channel.marginal_probability(
            report.value,
            state,
            spec.values,
            report.certainty,
        )
        for state, probability in predictive_states.items()
    }
    denominator = sum(state_weights.values())
    if denominator <= 0:
        return 0.0
    return 1.0 - state_weights[report.value] / denominator


def _independent_observation_likelihoods(
    model: DiseaseStateModel,
    report: Observation,
    channel: AnswerChannel,
) -> dict[str, float]:
    spec = model.specs[report.key]
    mode_prior = channel.parameters.cue_priors[report.certainty]
    return {
        disease: sum(
            state_probability
            * channel.marginal_probability(
                report.value,
                state,
                spec.values,
                report.certainty,
                mode_prior,
            )
            for state, state_probability in model.state_distribution(
                disease,
                report.key,
            ).items()
        )
        for disease in model.diseases
    }


def _stop_at_threshold(
    snapshots: list[_TrajectorySnapshot], threshold: float
) -> _TrajectorySnapshot:
    for snapshot in snapshots:
        if max(snapshot.belief.values()) >= threshold:
            return snapshot
    return snapshots[-1]


def _initialize_process_worker(
    model: DiseaseStateModel,
    variants: tuple[str, ...],
    posterior_thresholds: tuple[float, ...],
    max_questions: int,
    seed: int,
) -> None:
    global _PROCESS_CONTEXT
    _PROCESS_CONTEXT = (
        model,
        _make_runtimes(model, variants),
        posterior_thresholds,
        max_questions,
        seed,
    )


def _process_evaluate(task: tuple[ClinicalCase, float]) -> list[CaseOutcome]:
    if _PROCESS_CONTEXT is None:
        raise RuntimeError("clarification process worker was not initialized")
    model, runtimes, thresholds, max_questions, seed = _PROCESS_CONTEXT
    return _evaluate_case_noise(
        task,
        model=model,
        runtimes=runtimes,
        posterior_thresholds=thresholds,
        max_questions=max_questions,
        seed=seed,
    )


def summarize_clarification_outcomes(
    outcomes: Iterable[CaseOutcome],
) -> list[ClarificationSummary]:
    grouped: dict[tuple[str, float, float], list[CaseOutcome]] = defaultdict(list)
    for row in outcomes:
        grouped[(row.strategy, row.noise_rate, row.posterior_threshold)].append(row)
    summaries: list[ClarificationSummary] = []
    for (strategy, noise_rate, threshold), rows in sorted(grouped.items()):
        clarifications = sum(row.clarifications for row in rows)
        harmful = sum(row.harmful_reports for row in rows)
        detected = sum(row.detected_harmful_reports for row in rows)
        false_clarifications = sum(row.false_clarifications for row in rows)
        mitigated = sum(row.mitigated_harmful_reports for row in rows)
        primary_questions = sum(
            row.questions - row.clarifications for row in rows
        )
        summaries.append(
            ClarificationSummary(
                strategy=strategy,
                noise_rate=noise_rate,
                posterior_threshold=threshold,
                cases=len(rows),
                average_primary_questions=primary_questions / len(rows),
                average_clarifications=clarifications / len(rows),
                clarification_rate=(
                    clarifications / primary_questions
                    if primary_questions
                    else 0.0
                ),
                harmful_reports=harmful,
                detection_precision=(
                    detected / (detected + false_clarifications)
                    if detected + false_clarifications
                    else None
                ),
                detection_recall=detected / harmful if harmful else None,
                mitigation_rate=mitigated / detected if detected else None,
                discarded_correct_reports_per_case=(
                    sum(row.discarded_correct_reports for row in rows) / len(rows)
                ),
            )
        )
    return summaries


def save_clarification_summaries_csv(
    summaries: Iterable[ClarificationSummary], path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(ClarificationSummary)]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)
