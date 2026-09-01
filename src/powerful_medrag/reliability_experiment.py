"""Small dependency-free reliability-policy experiments and metrics."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import fmean
from typing import Iterable

from .belief import BeliefTracker
from .channel import AnswerChannel, answer_channel_without_misreport
from .decision import (
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    run_reliability_aware_dialogue,
)
from .estimation import DiseaseStateModel
from .gate_analysis import _expected_calibration_error
from .gating import HistoryReliabilityMisreportGate, LearnedMisreportGate
from .questioning import NumpyQuestionSelector, RandomQuestionSelector
from .schema import ClinicalCase
from .simulator import (
    PatientProfile,
    StopRule,
    StructuredPatientSimulator,
    run_dialogue,
)


@dataclass(frozen=True)
class ReliabilityCaseOutcome:
    seed: int
    case_id: str
    strategy: str
    noise_rate: float
    true_diagnosis: str
    predicted_diagnosis: str
    correct_top1: int
    correct_top3: int
    confidence: float
    brier_score: float
    new_questions: int
    verification_questions: int
    total_atomic_questions: int
    interaction_turns: int
    unnecessary_verifications: int
    resolved_wrong_reports: int
    premature_stop: int
    uncertain_output: int
    stop_reason: str


@dataclass(frozen=True)
class ReliabilitySummary:
    strategy: str
    noise_rate: float
    cases: int
    top1_accuracy: float
    top3_accuracy: float
    average_new_questions: float
    average_verification_questions: float
    average_total_atomic_questions: float
    average_interaction_turns: float
    brier_score: float
    expected_calibration_error: float
    unnecessary_verification_rate: float | None
    conflict_resolution_rate: float | None
    premature_stop_rate: float
    uncertain_output_rate: float


def run_reliability_experiment(
    model: DiseaseStateModel,
    cases: Iterable[ClinicalCase],
    *,
    noise_rates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3),
    seeds: tuple[int, ...] = (2026, 2027, 2028),
    max_total_turns: int = 8,
    policy_config: ReliabilityAwarePolicyConfig | None = None,
    learned_verification_gate: LearnedMisreportGate | None = None,
    strategies: tuple[str, ...] = (
        "random_reliable",
        "ordinary_eig_reliable",
        "full_two_layer",
        "adaptive_history",
        "joint_new_verify_stop",
    ),
) -> list[ReliabilityCaseOutcome]:
    case_list = list(cases)
    if not case_list or not seeds or not noise_rates:
        raise ValueError("pilot cases, seeds, and noise rates must be non-empty")
    allowed_strategies = {
        "random_reliable",
        "ordinary_eig_reliable",
        "full_two_layer",
        "adaptive_history",
        "joint_new_verify_stop",
        "joint_learned_gate",
        "oracle_verify",
        "oracle_select_same_channel",
    }
    unknown_strategies = set(strategies) - allowed_strategies
    if not strategies or unknown_strategies:
        raise ValueError(f"unknown reliability strategies: {sorted(unknown_strategies)}")
    if "joint_learned_gate" in strategies and learned_verification_gate is None:
        raise ValueError(
            "joint_learned_gate requires a learned_verification_gate "
            "(load it via gate_learning.load_learned_gate)"
        )
    rows: list[ReliabilityCaseOutcome] = []
    for seed in seeds:
        for noise_rate in noise_rates:
            for case in case_list:
                paired_seed = _pilot_seed(seed, case.case_id, noise_rate)
                for strategy in strategies:
                    patient = StructuredPatientSimulator(
                        diagnosis=case.diagnosis,
                        latent_states=case.states,
                        model=model,
                        profile=PatientProfile.from_noise_rate(noise_rate),
                        seed=paired_seed,
                    )
                    if strategy in (
                        "joint_new_verify_stop",
                        "joint_learned_gate",
                        "oracle_verify",
                        "oracle_select_same_channel",
                    ):
                        tracker = BeliefTracker(
                            model,
                            answer_channel_without_misreport(),
                            misreport_gate=HistoryReliabilityMisreportGate(),
                        )
                        policy = ReliabilityAwareActionPolicy(
                            config=(
                                policy_config
                                or ReliabilityAwarePolicyConfig(
                                    max_total_turns=max_total_turns,
                                )
                            ),
                            oracle_selection=(
                                strategy
                                in ("oracle_verify", "oracle_select_same_channel")
                            ),
                            oracle_correction=(strategy == "oracle_verify"),
                            verification_risk_gate=(
                                learned_verification_gate
                                if strategy == "joint_learned_gate"
                                else None
                            ),
                        )
                        result = run_reliability_aware_dialogue(
                            patient,
                            policy=policy,
                            tracker=tracker,
                            initial_observations=case.initial_observations,
                        )
                        belief = result.belief
                        new_questions = result.new_questions
                        verifications = result.verification_questions
                        unnecessary = sum(
                            turn.verification_was_unnecessary for turn in result.turns
                        )
                        resolved = sum(
                            turn.verification_resolved_wrong_report
                            for turn in result.turns
                        )
                        stop_reason = result.stop_reason
                        uncertain = int(result.uncertain_output)
                    else:
                        if strategy == "full_two_layer":
                            channel = AnswerChannel()
                            gate = None
                            selector = NumpyQuestionSelector()
                        elif strategy == "adaptive_history":
                            channel = answer_channel_without_misreport()
                            gate = HistoryReliabilityMisreportGate()
                            selector = NumpyQuestionSelector()
                        else:
                            channel = answer_channel_without_misreport()
                            gate = None
                            selector = (
                                RandomQuestionSelector(seed=seed)
                                if strategy == "random_reliable"
                                else NumpyQuestionSelector()
                            )
                        result = run_dialogue(
                            patient,
                            tracker=BeliefTracker(model, channel, misreport_gate=gate),
                            selector=selector,
                            stop_rule=StopRule(
                                max_questions=max_total_turns,
                                posterior_threshold=(
                                    policy_config.posterior_threshold
                                    if policy_config is not None
                                    else 0.85
                                ),
                                entropy_threshold=0.0,
                                minimum_question_utility=0.0,
                            ),
                            initial_observations=case.initial_observations,
                        )
                        belief = result.belief
                        new_questions = len(result.turns)
                        verifications = 0
                        unnecessary = 0
                        resolved = 0
                        stop_reason = result.stop_reason
                        uncertain = 0
                    ranking = sorted(belief, key=belief.__getitem__, reverse=True)
                    predicted = ranking[0]
                    confidence = belief[predicted]
                    brier = sum(
                        (
                            belief[disease]
                            - (1.0 if disease == case.diagnosis else 0.0)
                        )
                        ** 2
                        for disease in model.diseases
                    )
                    total_turns = new_questions + verifications
                    confident_stop = (
                        "posterior" in stop_reason
                        or "confidence" in stop_reason
                    )
                    rows.append(
                        ReliabilityCaseOutcome(
                            seed=seed,
                            case_id=case.case_id,
                            strategy=strategy,
                            noise_rate=noise_rate,
                            true_diagnosis=case.diagnosis,
                            predicted_diagnosis=predicted,
                            correct_top1=int(predicted == case.diagnosis),
                            correct_top3=int(case.diagnosis in ranking[:3]),
                            confidence=confidence,
                            brier_score=brier,
                            new_questions=new_questions,
                            verification_questions=verifications,
                            total_atomic_questions=total_turns,
                            interaction_turns=total_turns,
                            unnecessary_verifications=unnecessary,
                            resolved_wrong_reports=resolved,
                            premature_stop=int(
                                predicted != case.diagnosis and confident_stop
                            ),
                            uncertain_output=uncertain,
                            stop_reason=stop_reason,
                        )
                    )
    return rows


# Backward-compatible descriptive alias for the dependency-free toy CLI.
run_toy_reliability_pilot = run_reliability_experiment


def summarize_reliability_outcomes(
    outcomes: Iterable[ReliabilityCaseOutcome],
) -> list[ReliabilitySummary]:
    groups: dict[tuple[str, float], list[ReliabilityCaseOutcome]] = defaultdict(list)
    for row in outcomes:
        groups[(row.strategy, row.noise_rate)].append(row)
    summaries: list[ReliabilitySummary] = []
    for (strategy, noise_rate), rows in sorted(groups.items()):
        verification_count = sum(row.verification_questions for row in rows)
        resolved_count = sum(row.resolved_wrong_reports for row in rows)
        labels = [row.correct_top1 for row in rows]
        confidences = [row.confidence for row in rows]
        summaries.append(
            ReliabilitySummary(
                strategy=strategy,
                noise_rate=noise_rate,
                cases=len(rows),
                top1_accuracy=fmean(labels),
                top3_accuracy=fmean(row.correct_top3 for row in rows),
                average_new_questions=fmean(row.new_questions for row in rows),
                average_verification_questions=fmean(
                    row.verification_questions for row in rows
                ),
                average_total_atomic_questions=fmean(
                    row.total_atomic_questions for row in rows
                ),
                average_interaction_turns=fmean(row.interaction_turns for row in rows),
                brier_score=fmean(row.brier_score for row in rows),
                expected_calibration_error=_expected_calibration_error(
                    labels, confidences
                ),
                unnecessary_verification_rate=(
                    sum(row.unnecessary_verifications for row in rows)
                    / verification_count
                    if verification_count
                    else None
                ),
                conflict_resolution_rate=(
                    resolved_count / verification_count if verification_count else None
                ),
                premature_stop_rate=fmean(row.premature_stop for row in rows),
                uncertain_output_rate=fmean(row.uncertain_output for row in rows),
            )
        )
    return summaries


def save_reliability_csv(
    rows: Iterable[object], row_type: type, path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[field.name for field in fields(row_type)]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _pilot_seed(seed: int, case_id: str, noise_rate: float) -> int:
    import hashlib

    digest = hashlib.blake2b(
        f"{seed}|{case_id}|{noise_rate:.8f}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")
