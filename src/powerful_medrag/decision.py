"""Joint new-question, verification, and stop decisions.

This is a transparent one-step policy, not a claim of solving the full POMDP.
Verification value uses label-free leave-one-out error risk and diagnostic
influence as observable proxies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol

from .belief import BeliefTracker
from .channel import ReportMode
from .clarification import (
    RetrospectiveClarificationProtocol,
    SurprisalClarificationProtocol,
)
from .questioning import NumpyQuestionSelector, QuestionScore, QuestionSelector
from .schema import UNKNOWN, FeatureKey, Observation, VariableSpec
from .simulator import StructuredPatientSimulator


class ActionKind(str, Enum):
    NEW = "new"
    VERIFY = "verify"
    STOP = "stop"


class SafetyConstraint(Protocol):
    """Dataset-supplied safety rule; this repository provides no default labels."""

    def is_clear(
        self, tracker: BeliefTracker, asked: set[FeatureKey]
    ) -> bool: ...


@dataclass(frozen=True)
class RequiredFeatureSafetyConstraint:
    """Require a source-backed set of red-flag questions before confident stop."""

    required_features: frozenset[FeatureKey]

    def is_clear(self, tracker: BeliefTracker, asked: set[FeatureKey]) -> bool:
        del tracker
        return self.required_features <= asked


@dataclass(frozen=True)
class HighRiskDiseaseSafetyConstraint:
    """Block stop while source-labelled high-risk disease mass is unresolved."""

    high_risk_diseases: frozenset[str]
    maximum_unresolved_probability: float = 0.05

    def __post_init__(self) -> None:
        if not 0 <= self.maximum_unresolved_probability <= 1:
            raise ValueError("risk probability threshold must be in [0, 1]")

    def is_clear(self, tracker: BeliefTracker, asked: set[FeatureKey]) -> bool:
        del asked
        unknown = self.high_risk_diseases - set(tracker.model.diseases)
        if unknown:
            raise ValueError(f"unknown high-risk diseases: {sorted(unknown)}")
        return sum(tracker.belief[disease] for disease in self.high_risk_diseases) <= (
            self.maximum_unresolved_probability
        )


@dataclass(frozen=True)
class ReliabilityAwarePolicyConfig:
    posterior_threshold: float = 0.85
    posterior_margin_threshold: float = 0.70
    minimum_action_utility: float = 0.03
    suspicious_report_threshold: float = 0.05
    new_question_cost_weight: float = 0.01
    verification_cost: float = 0.03
    reliability_information_weight: float = 0.01
    decision_impact_weight: float = 0.25
    maximum_verifications: int = 1
    minimum_unreliable_history_cues_for_verification: int = 1
    max_total_turns: int = 15

    def __post_init__(self) -> None:
        probabilities = (
            self.posterior_threshold,
            self.posterior_margin_threshold,
            self.suspicious_report_threshold,
        )
        if min(probabilities) < 0 or max(probabilities) > 1:
            raise ValueError("policy probability thresholds must be in [0, 1]")
        if min(
            self.minimum_action_utility,
            self.new_question_cost_weight,
            self.verification_cost,
            self.reliability_information_weight,
            self.decision_impact_weight,
        ) < 0:
            raise ValueError("policy costs and weights cannot be negative")
        if (
            self.maximum_verifications < 0
            or self.minimum_unreliable_history_cues_for_verification < 0
            or self.max_total_turns <= 0
        ):
            raise ValueError("policy turn budgets are invalid")


@dataclass(frozen=True)
class ActionScore:
    kind: ActionKind
    utility: float
    disease_information_gain: float = 0.0
    reliability_information_gain: float = 0.0
    decision_impact: float = 0.0
    burden: float = 0.0
    key: FeatureKey | None = None
    report_index: int | None = None
    explanation: str = ""


@dataclass(frozen=True)
class PolicyTurn:
    index: int
    action: ActionScore
    observation: Observation | None
    true_report_mode: ReportMode | None
    belief: Mapping[str, float]
    verification_changed_report: bool = False
    verification_was_unnecessary: bool = False
    verification_resolved_wrong_report: bool = False


@dataclass(frozen=True)
class ReliabilityAwareDialogueResult:
    true_diagnosis: str
    predicted_diagnosis: str
    belief: Mapping[str, float]
    turns: tuple[PolicyTurn, ...]
    stop_reason: str
    new_questions: int
    verification_questions: int
    uncertain_output: bool

    @property
    def correct(self) -> bool:
        return self.true_diagnosis == self.predicted_diagnosis


class ReliabilityAwareActionPolicy:
    def __init__(
        self,
        *,
        selector: QuestionSelector | None = None,
        config: ReliabilityAwarePolicyConfig | None = None,
        safety_constraint: SafetyConstraint | None = None,
    ) -> None:
        self.selector = selector or NumpyQuestionSelector()
        self.config = config or ReliabilityAwarePolicyConfig()
        self.safety_constraint = safety_constraint
        self.retrospective = RetrospectiveClarificationProtocol(
            score_threshold=0.0,
            max_clarifications=max(1, self.config.maximum_verifications),
            use_diagnostic_influence=True,
        )

    def rank_actions(
        self,
        tracker: BeliefTracker,
        *,
        initial_observations: tuple[Observation, ...],
        reports: tuple[Observation, ...],
        asked: set[FeatureKey],
        verified_report_indices: set[int],
        verification_count: int,
    ) -> list[ActionScore]:
        actions: list[ActionScore] = []
        for question in self.selector.rank(tracker, excluded=asked):
            actions.append(self._new_action(question, tracker.model.specs[question.key]))
        unreliable_history_cues = sum(
            update.observation.value == UNKNOWN
            or update.observation.certainty.value == "uncertain"
            for update in tracker.history
        )
        verification_gate_open = (
            unreliable_history_cues
            >= self.config.minimum_unreliable_history_cues_for_verification
        )
        if (
            verification_count < self.config.maximum_verifications
            and reports
            and verification_gate_open
        ):
            scores = self.retrospective.rank(
                tracker.model,
                initial_observations,
                reports,
                excluded=verified_report_indices,
            )
            actions.extend(self._verification_action(score) for score in scores)
        return sorted(actions, key=lambda action: action.utility, reverse=True)

    def choose_action(
        self,
        tracker: BeliefTracker,
        *,
        initial_observations: tuple[Observation, ...],
        reports: tuple[Observation, ...],
        asked: set[FeatureKey],
        verified_report_indices: set[int],
        verification_count: int,
    ) -> ActionScore:
        actions = self.rank_actions(
            tracker,
            initial_observations=initial_observations,
            reports=reports,
            asked=asked,
            verified_report_indices=verified_report_indices,
            verification_count=verification_count,
        )
        best = actions[0] if actions else None
        all_report_scores = (
            self.retrospective.rank(
                tracker.model,
                initial_observations,
                reports,
                excluded=verified_report_indices,
            )
            if reports
            else []
        )
        suspicious = max((score.score for score in all_report_scores), default=0.0)
        ranked = tracker.ranked_diseases()
        top_probability = ranked[0][1]
        margin = top_probability - (ranked[1][1] if len(ranked) > 1 else 0.0)
        confidence_ready = (
            top_probability >= self.config.posterior_threshold
            or margin >= self.config.posterior_margin_threshold
        )
        utility_low = best is None or best.utility <= self.config.minimum_action_utility
        reliability_ready = suspicious <= self.config.suspicious_report_threshold
        safety_ready = (
            self.safety_constraint is None
            or self.safety_constraint.is_clear(tracker, asked)
        )
        if confidence_ready and utility_low and reliability_ready and safety_ready:
            return ActionScore(
                kind=ActionKind.STOP,
                utility=0.0,
                explanation="confidence, marginal utility, and report-risk checks passed",
            )
        if best is not None and best.utility > 0:
            return best
        return ActionScore(
            kind=ActionKind.STOP,
            utility=0.0,
            explanation=(
                "no positive-utility acquisition action or a reliability/safety check remains; "
                "return uncertainty"
            ),
        )

    def _new_action(self, question: QuestionScore, spec: VariableSpec) -> ActionScore:
        burden = self.config.new_question_cost_weight * spec.cost
        utility = question.expected_information_gain - burden
        return ActionScore(
            kind=ActionKind.NEW,
            utility=utility,
            disease_information_gain=question.expected_information_gain,
            decision_impact=question.expected_information_gain,
            burden=burden,
            key=question.key,
            explanation="expected disease entropy reduction minus new-question burden",
        )

    def _verification_action(self, score) -> ActionScore:
        reliability_gain = _binary_entropy(score.error_probability)
        disease_gain = score.error_probability * score.diagnostic_influence
        utility = (
            disease_gain
            + self.config.reliability_information_weight * reliability_gain
            + self.config.decision_impact_weight * score.diagnostic_influence
            - self.config.verification_cost
        )
        return ActionScore(
            kind=ActionKind.VERIFY,
            utility=utility,
            disease_information_gain=score.error_probability,
            reliability_information_gain=reliability_gain,
            decision_impact=score.diagnostic_influence,
            burden=self.config.verification_cost,
            report_index=score.report_index,
            explanation=(
                "leave-one-out report-error risk, diagnostic influence, and verification burden"
            ),
        )


def run_reliability_aware_dialogue(
    patient: StructuredPatientSimulator,
    *,
    policy: ReliabilityAwareActionPolicy | None = None,
    tracker: BeliefTracker | None = None,
    initial_observations: tuple[Observation, ...] = (),
) -> ReliabilityAwareDialogueResult:
    policy = policy or ReliabilityAwareActionPolicy()
    tracker = tracker or BeliefTracker(patient.model, patient.channel)
    for observation in initial_observations:
        tracker.update(observation)
    asked = {observation.key for observation in initial_observations}
    reports: list[Observation] = []
    verified: set[int] = set()
    turns: list[PolicyTurn] = []
    verification_count = 0
    stop_reason = "total_turn_budget"
    uncertain_output = False

    while len(turns) < policy.config.max_total_turns:
        action = policy.choose_action(
            tracker,
            initial_observations=initial_observations,
            reports=tuple(reports),
            asked=asked,
            verified_report_indices=verified,
            verification_count=verification_count,
        )
        if action.kind == ActionKind.STOP:
            stop_reason = action.explanation
            uncertain_output = "uncertainty" in action.explanation
            break
        if action.kind == ActionKind.NEW:
            if action.key is None:
                raise RuntimeError("new-question action lacks a feature key")
            observation, true_mode = patient.answer(action.key)
            reports.append(observation)
            tracker.update(observation)
            asked.add(action.key)
            turns.append(
                PolicyTurn(
                    index=len(turns) + 1,
                    action=action,
                    observation=observation,
                    true_report_mode=true_mode,
                    belief=dict(tracker.belief),
                )
            )
            continue

        report_index = action.report_index
        if report_index is None or report_index >= len(reports):
            raise RuntimeError("verification action has an invalid report index")
        original = reports[report_index]
        clarification, true_mode = patient.answer(original.key)
        resolved = SurprisalClarificationProtocol.resolve(original, clarification)
        true_state = patient.latent_states.get(original.key, UNKNOWN)
        original_wrong = (
            true_state != UNKNOWN
            and original.value != UNKNOWN
            and original.value != true_state
        )
        resolved_wrong = (
            true_state != UNKNOWN
            and resolved.value != UNKNOWN
            and resolved.value != true_state
        )
        reports[report_index] = resolved
        verified.add(report_index)
        verification_count += 1
        tracker = _replay_with_runtime(
            tracker,
            initial_observations=initial_observations,
            reports=tuple(reports),
        )
        turns.append(
            PolicyTurn(
                index=len(turns) + 1,
                action=action,
                observation=clarification,
                true_report_mode=true_mode,
                belief=dict(tracker.belief),
                verification_changed_report=(
                    resolved.value != original.value
                    or resolved.certainty != original.certainty
                ),
                verification_was_unnecessary=not original_wrong,
                verification_resolved_wrong_report=original_wrong and not resolved_wrong,
            )
        )

    predicted = max(tracker.belief, key=tracker.belief.__getitem__)
    return ReliabilityAwareDialogueResult(
        true_diagnosis=patient.diagnosis,
        predicted_diagnosis=predicted,
        belief=dict(tracker.belief),
        turns=tuple(turns),
        stop_reason=stop_reason,
        new_questions=sum(turn.action.kind == ActionKind.NEW for turn in turns),
        verification_questions=sum(
            turn.action.kind == ActionKind.VERIFY for turn in turns
        ),
        uncertain_output=uncertain_output,
    )


def clarification_question(spec: VariableSpec, original: Observation) -> str:
    """Generate a neutral structured recheck prompt without accusing the patient."""

    question = spec.question or f"关于 {spec.key.display_name()} 的情况是什么？"
    context = spec.key.display_name()
    return (
        f"为了确认我理解得准确，请结合当时的具体情况再回想一下（{context}）："
        f"{question} 如果不确定，也可以直接说不知道。"
    )


def _replay_with_runtime(
    previous: BeliefTracker,
    *,
    initial_observations: tuple[Observation, ...],
    reports: tuple[Observation, ...],
) -> BeliefTracker:
    tracker = BeliefTracker(
        previous.model,
        previous.channel,
        misreport_gate=previous.misreport_gate,
    )
    for observation in initial_observations:
        tracker.update(observation)
    for observation in reports:
        tracker.update(observation)
    return tracker


def _binary_entropy(probability: float) -> float:
    if probability <= 0 or probability >= 1:
        return 0.0
    return -probability * math.log2(probability) - (1 - probability) * math.log2(
        1 - probability
    )
