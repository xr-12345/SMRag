"""Cohort-level metrics for diagnosis, calibration, noise and question burden."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Callable

from .channel import ReportMode
from .estimation import DiseaseStateModel
from .questioning import QuestionSelector
from .simulator import (
    PatientProfile,
    StopRule,
    StructuredPatientSimulator,
    run_dialogue,
)


@dataclass(frozen=True)
class CohortMetrics:
    cases: int
    accuracy: float
    average_questions: float
    average_brier_score: float
    unknown_answer_rate: float
    misreport_detection_auroc: float | None


def evaluate_simulated_cohort(
    model: DiseaseStateModel,
    *,
    cases_per_disease: int = 100,
    profile: PatientProfile | None = None,
    stop_rule: StopRule | None = None,
    selector_factory: Callable[[], QuestionSelector] = QuestionSelector,
    seed: int = 1000,
) -> CohortMetrics:
    results = []
    brier_scores: list[float] = []
    mode_labels: list[int] = []
    mode_scores: list[float] = []
    unknown_answers = 0
    answer_count = 0

    for disease_index, disease in enumerate(model.diseases):
        for case_index in range(cases_per_disease):
            case_seed = seed + disease_index * cases_per_disease + case_index
            patient = StructuredPatientSimulator.sample_case(
                model,
                disease,
                profile=profile,
                seed=case_seed,
            )
            result = run_dialogue(
                patient,
                selector=selector_factory(),
                stop_rule=stop_rule,
            )
            results.append(result)
            brier_scores.append(
                sum(
                    (
                        result.belief[candidate]
                        - (1.0 if candidate == disease else 0.0)
                    )
                    ** 2
                    for candidate in model.diseases
                )
            )
            for turn in result.turns:
                answer_count += 1
                if turn.observation.value == "__unknown__":
                    unknown_answers += 1
                mode_labels.append(
                    1 if turn.true_report_mode == ReportMode.MISREPORTED else 0
                )
                mode_scores.append(turn.update.misreport_probability)

    return CohortMetrics(
        cases=len(results),
        accuracy=fmean(1.0 if result.correct else 0.0 for result in results),
        average_questions=fmean(len(result.turns) for result in results),
        average_brier_score=fmean(brier_scores),
        unknown_answer_rate=(unknown_answers / answer_count if answer_count else 0.0),
        misreport_detection_auroc=_binary_auroc(mode_labels, mode_scores),
    )


def _binary_auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    favorable_pairs = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                favorable_pairs += 1.0
            elif positive == negative:
                favorable_pairs += 0.5
    return favorable_pairs / (len(positives) * len(negatives))

