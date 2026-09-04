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
    RetrospectiveScore,
    SurprisalClarificationProtocol,
)
from .gating import LearnedMisreportGate
from .questioning import NumpyQuestionSelector, QuestionScore, QuestionSelector
from .retrieval import (
    MedicalRetriever,
    apply_counterfactual,
    build_retrieval_query,
    jaccard,
)
from .schema import UNKNOWN, CertaintyCue, FeatureKey, Observation, VariableSpec
from .simulator import StructuredPatientSimulator


class ActionKind(str, Enum):
    NEW = "new"
    VERIFY = "verify"
    STOP = "stop"


class RetrievalMode(str, Enum):
    """Whether (and how) medical retrieval participates in the decision loop.

    * ``NO_RAG``     — no retrieval at all (the pre-RAG baseline).
    * ``STATIC_RAG`` — build the query once from the presenting complaint, never
      refresh it as the patient answers.
    * ``DYNAMIC_RAG``— rebuild the query and re-retrieve after every answer.
    """

    NO_RAG = "no_rag"
    STATIC_RAG = "static_rag"
    DYNAMIC_RAG = "dynamic_rag"


class RetrievalGateMode(str, Enum):
    """Whether retrieval may open the VerifyOld gate, not merely re-rank it.

    * ``RANK_ONLY`` — retrieval impact only re-ranks VerifyOld candidates that
      the history / learned gate has already allowed (the pre-joint-gate default).
    * ``JOINT_GATE`` — retrieval impact may additionally open the VerifyOld gate
      when a label-free activation value clears ``minimum_action_utility``.
    """

    RANK_ONLY = "rank_only"
    JOINT_GATE = "joint_gate"


class StopAuditMode(str, Enum):
    """How the joint policy's reliability check gates a confident Stop.

    * ``HARD_PROBABILITY_GATE`` — Stop is blocked while any asked feature's
      joint misreport posterior ``p_i^mode`` exceeds ``suspicious_report_threshold``
      (the pre-Phase-22A behaviour; the default, so old results reproduce).
    * ``DECISION_VALUE_AUDIT`` — Stop is blocked while the gross verification
      gain ``G_t^verify = max_i (R_B(b_t) - E_{y'} R_B(b_{t+1}^{(i,y')}))``
      exceeds ``verification_audit_threshold``; ``max p_mode`` is logged only and
      no longer gates Stop on its own.
    """

    HARD_PROBABILITY_GATE = "hard_probability_gate"
    DECISION_VALUE_AUDIT = "decision_value_audit"


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
    retrieval_mode: RetrievalMode = RetrievalMode.NO_RAG
    retrieval_impact_weight: float = 0.0
    retrieval_top_k: int = 10
    query_disease_top_k: int = 5
    retrieval_gate_mode: RetrievalGateMode = RetrievalGateMode.RANK_ONLY
    verification_audit_threshold: float = 0.03
    verification_advantage_margin: float = 0.03
    stop_audit_mode: StopAuditMode = StopAuditMode.HARD_PROBABILITY_GATE

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
            self.retrieval_impact_weight,
            self.verification_audit_threshold,
            self.verification_advantage_margin,
        ) < 0:
            raise ValueError("policy costs and weights cannot be negative")
        if (
            self.maximum_verifications < 0
            or self.minimum_unreliable_history_cues_for_verification < 0
            or self.max_total_turns <= 0
        ):
            raise ValueError("policy turn budgets are invalid")
        if self.retrieval_top_k <= 0 or self.query_disease_top_k <= 0:
            raise ValueError("retrieval top-k must be positive")


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
    retrieval_triggered: bool = False


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
    verification_triggered_by_retrieval: bool = False


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
    retrieval_log: tuple[dict, ...] = ()

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
        oracle_selection: bool = False,
        oracle_correction: bool = False,
        verification_risk_gate: LearnedMisreportGate | None = None,
        retriever: MedicalRetriever | None = None,
    ) -> None:
        self.selector = selector or NumpyQuestionSelector()
        self.config = config or ReliabilityAwarePolicyConfig()
        self.safety_constraint = safety_constraint
        self.oracle_selection = oracle_selection
        self.oracle_correction = oracle_correction
        self.verification_risk_gate = verification_risk_gate
        self.retriever = retriever
        self.retrospective = RetrospectiveClarificationProtocol(
            score_threshold=0.0,
            max_clarifications=max(1, self.config.maximum_verifications),
            use_diagnostic_influence=True,
        )
        self.last_verification_log: list[dict] = []

    def rank_actions(
        self,
        tracker: BeliefTracker,
        *,
        initial_observations: tuple[Observation, ...],
        reports: tuple[Observation, ...],
        asked: set[FeatureKey],
        verified_report_indices: set[int],
        verification_count: int,
        oracle_states: Mapping[FeatureKey, str] | None = None,
        report_risks: tuple[float, ...] = (),
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
        if self.verification_risk_gate is not None:
            if len(report_risks) != len(reports):
                raise ValueError(
                    "learned verification risks must align one-to-one with reports"
                )
            verification_gate_open = verification_gate_open or any(
                index not in verified_report_indices
                and report.value != UNKNOWN
                and report_risks[index]
                > self.verification_risk_gate.activation_threshold
                for index, report in enumerate(reports)
            )
        if self.oracle_selection and oracle_states is not None:
            verification_gate_open = True

        old_history_gate_open = verification_gate_open
        retrieval_opened_gate = False
        verification_scores: list[RetrospectiveScore] = []

        # Joint gate: retrieval may open VerifyOld even when the history gate is
        # still closed, but only via a label-free activation value clearing the
        # minimum-action bar (it never forces verification).
        if (
            verification_count < self.config.maximum_verifications
            and reports
            and not verification_gate_open
            and self.retriever is not None
            and self.config.retrieval_mode is not RetrievalMode.NO_RAG
            and self.config.retrieval_gate_mode is RetrievalGateMode.JOINT_GATE
        ):
            verification_scores = self._verification_scores(
                tracker,
                initial_observations=initial_observations,
                reports=reports,
                verified_report_indices=verified_report_indices,
                report_risks=report_risks,
            )
            activation_values = [
                self._retrieval_activation_value(
                    score.error_probability, score.retrieval_impact
                )
                for score in verification_scores
            ]
            if (
                activation_values
                and max(activation_values) >= self.config.minimum_action_utility
            ):
                verification_gate_open = True
                retrieval_opened_gate = True

        self.last_verification_log = []
        if (
            verification_count < self.config.maximum_verifications
            and reports
            and verification_gate_open
        ):
            if self.oracle_selection and oracle_states is not None:
                for index in _oracle_wrong_reports(
                    reports, verified_report_indices, oracle_states
                ):
                    actions.append(
                        ActionScore(
                            kind=ActionKind.VERIFY,
                            utility=1.0,
                            report_index=index,
                            explanation=(
                                "oracle verification of a truly-misreported answer"
                            ),
                        )
                    )
            else:
                if not verification_scores:
                    verification_scores = self._verification_scores(
                        tracker,
                        initial_observations=initial_observations,
                        reports=reports,
                        verified_report_indices=verified_report_indices,
                        report_risks=report_risks,
                    )
                best_asknew_utility = max(
                    (
                        action.utility
                        for action in actions
                        if action.kind is ActionKind.NEW
                    ),
                    default=None,
                )
                candidate_logs: list[dict] = []
                for score in verification_scores:
                    existing, final = self._verification_utility(score)
                    actions.append(
                        self._verification_action(
                            score, retrieval_triggered=retrieval_opened_gate
                        )
                    )
                    candidate_logs.append(
                        {
                            "report_index": score.report_index,
                            "evidence_code": reports[score.report_index].key.name,
                            "error_probability": round(score.error_probability, 6),
                            "raw_retrieval_impact": round(
                                1.0 - score.retrieval_impact, 6
                            ),
                            "normalized_retrieval_impact": round(
                                score.retrieval_impact, 6
                            ),
                            "retrieval_activation_value": round(
                                self._retrieval_activation_value(
                                    score.error_probability,
                                    score.retrieval_impact,
                                ),
                                6,
                            ),
                            "existing_verification_utility": round(existing, 6),
                            "final_verification_utility": round(final, 6),
                            "old_history_gate_open": old_history_gate_open,
                            "opened_by_retrieval": retrieval_opened_gate,
                            "chosen_action": None,
                            "best_asknew_utility": (
                                round(best_asknew_utility, 6)
                                if best_asknew_utility is not None
                                else None
                            ),
                        }
                    )
                self.last_verification_log = candidate_logs

        ranked = sorted(actions, key=lambda action: action.utility, reverse=True)
        if self.last_verification_log:
            chosen_action = ranked[0].kind.value if ranked else None
            for entry in self.last_verification_log:
                entry["chosen_action"] = chosen_action
        return ranked

    def choose_action(
        self,
        tracker: BeliefTracker,
        *,
        initial_observations: tuple[Observation, ...],
        reports: tuple[Observation, ...],
        asked: set[FeatureKey],
        verified_report_indices: set[int],
        verification_count: int,
        oracle_states: Mapping[FeatureKey, str] | None = None,
        report_risks: tuple[float, ...] = (),
    ) -> ActionScore:
        actions = self.rank_actions(
            tracker,
            initial_observations=initial_observations,
            reports=reports,
            asked=asked,
            verified_report_indices=verified_report_indices,
            verification_count=verification_count,
            oracle_states=oracle_states,
            report_risks=report_risks,
        )
        best = actions[0] if actions else None
        if self.oracle_selection and oracle_states is not None:
            suspicious = (
                1.0
                if verification_count < self.config.maximum_verifications
                and _oracle_wrong_reports(
                    reports, verified_report_indices, oracle_states
                )
                else 0.0
            )
        else:
            all_report_scores = (
                self._verification_scores(
                    tracker,
                    initial_observations=initial_observations,
                    reports=reports,
                    verified_report_indices=verified_report_indices,
                    report_risks=report_risks,
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

    def _verification_scores(
        self,
        tracker: BeliefTracker,
        *,
        initial_observations: tuple[Observation, ...],
        reports: tuple[Observation, ...],
        verified_report_indices: set[int],
        report_risks: tuple[float, ...],
    ) -> list[RetrospectiveScore]:
        scores = self.retrospective.rank(
            tracker.model,
            initial_observations,
            reports,
            excluded=verified_report_indices,
        )
        if self.verification_risk_gate is not None:
            if len(report_risks) != len(reports):
                raise ValueError(
                    "learned verification risks must align one-to-one with reports"
                )
            scores = [
                RetrospectiveScore(
                    report_index=score.report_index,
                    error_probability=report_risks[score.report_index],
                    diagnostic_influence=score.diagnostic_influence,
                    score=report_risks[score.report_index]
                    * score.diagnostic_influence,
                )
                for score in scores
            ]
        if self.retriever is not None and self.config.retrieval_mode is not RetrievalMode.NO_RAG:
            impacts = self._retrieval_impacts(tracker, reports, verified_report_indices)
            scores = [
                RetrospectiveScore(
                    report_index=score.report_index,
                    error_probability=score.error_probability,
                    diagnostic_influence=score.diagnostic_influence,
                    score=score.score,
                    retrieval_impact=impacts.get(score.report_index, 0.0),
                )
                for score in scores
            ]
        return scores

    def _retrieval_impacts(
        self,
        tracker: BeliefTracker,
        reports: tuple[Observation, ...],
        verified_report_indices: set[int],
    ) -> dict[int, float]:
        """Per-report retrieval impact: how much does deleting one report change
        the BM25 result?  A report whose removal reorders the retrieved snippets
        is evidence worth verifying.  Returns {report_index: impact in [0, 1]}."""
        retriever = self.retriever
        if retriever is None:
            return {}
        ranked = tracker.ranked_diseases()
        top_k = self.config.query_disease_top_k
        k = self.config.retrieval_top_k
        feature_names = _readable_feature_names(tracker)
        baseline_query = build_retrieval_query(
            reports, ranked, top_k=top_k, feature_names=feature_names
        )
        baseline_hits = retriever.retrieve(baseline_query.text, k=k)
        impacts: dict[int, float] = {}
        for index, report in enumerate(reports):
            if index in verified_report_indices or report.value == UNKNOWN:
                continue
            altered = apply_counterfactual(reports, index, "delete")
            after_query = build_retrieval_query(
                altered, ranked, top_k=top_k, feature_names=feature_names
            )
            after_hits = retriever.retrieve(after_query.text, k=k)
            # 0 = retrieval unchanged, 1 = retrieval totally reordered.
            impacts[index] = min(1.0, max(0.0, 1.0 - jaccard(baseline_hits, after_hits)))
        return impacts

    def _verification_utility(self, score) -> tuple[float, float]:
        """Return (existing, final) verification utilities for one candidate.

        ``existing`` is the pre-retrieval utility (disease gain + reliability
        gain + decision influence - cost); ``final`` adds the retrieval-impact
        bonus so a re-ranking-worthy report edges out its peers.  Both are
        label-free: neither reads latent state nor the true disease.
        """
        reliability_gain = _binary_entropy(score.error_probability)
        disease_gain = score.error_probability * score.diagnostic_influence
        existing = (
            disease_gain
            + self.config.reliability_information_weight * reliability_gain
            + self.config.decision_impact_weight * score.diagnostic_influence
            - self.config.verification_cost
        )
        final = existing + (
            self.config.retrieval_impact_weight
            * score.error_probability
            * getattr(score, "retrieval_impact", 0.0)
        )
        return existing, final

    def _retrieval_activation_value(
        self, error_probability: float, normalized_retrieval_impact: float
    ) -> float:
        """Label-free value of opening VerifyOld through retrieval impact.

        ``error_probability * normalized_retrieval_impact`` is the expected
        payoff of correcting a likely-wrong, retrieval-relevant report, minus
        the verification cost.  No diagnostic influence, no latent state, no
        true disease enter this quantity.
        """
        return error_probability * normalized_retrieval_impact - self.config.verification_cost

    def _verification_action(
        self, score, retrieval_triggered: bool = False
    ) -> ActionScore:
        reliability_gain = _binary_entropy(score.error_probability)
        _, utility = self._verification_utility(score)
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
            retrieval_triggered=retrieval_triggered,
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
    report_risks: list[float] = []
    verified: set[int] = set()
    turns: list[PolicyTurn] = []
    verification_count = 0
    stop_reason = "total_turn_budget"
    uncertain_output = False
    retrieval_log: list[dict] = []

    retriever = policy.retriever
    mode = policy.config.retrieval_mode
    feature_names = _readable_feature_names(tracker)
    if retriever is not None and mode is RetrievalMode.STATIC_RAG:
        # Build the query once from the presenting complaint, never refresh it.
        static_query = build_retrieval_query(
            initial_observations,
            (),
            top_k=policy.config.query_disease_top_k,
            feature_names=feature_names,
        )
        hits = retriever.retrieve(static_query.text, k=policy.config.retrieval_top_k)
        retrieval_log.append(
            {
                "turn": 0,
                "mode": "static_rag",
                "query": static_query.text,
                "hit_ids": [h.snippet_id for h in hits],
                "hit_titles": [h.title for h in hits],
            }
        )

    while len(turns) < policy.config.max_total_turns:
        if retriever is not None and mode is RetrievalMode.DYNAMIC_RAG:
            # Rebuild the query and re-retrieve after the latest answer.
            query = build_retrieval_query(
                tuple(reports),
                tracker.ranked_diseases(),
                top_k=policy.config.query_disease_top_k,
                feature_names=feature_names,
            )
            hits = retriever.retrieve(query.text, k=policy.config.retrieval_top_k)
            retrieval_log.append(
                {
                    "turn": len(turns) + 1,
                    "mode": "dynamic_rag",
                    "query": query.text,
                    "hit_ids": [h.snippet_id for h in hits],
                    "hit_titles": [h.title for h in hits],
                }
            )
        action = policy.choose_action(
            tracker,
            initial_observations=initial_observations,
            reports=tuple(reports),
            asked=asked,
            verified_report_indices=verified,
            verification_count=verification_count,
            oracle_states=(
                patient.latent_states
                if policy.oracle_selection or policy.oracle_correction
                else None
            ),
            report_risks=tuple(report_risks),
        )
        if action.kind == ActionKind.STOP:
            stop_reason = action.explanation
            uncertain_output = "uncertainty" in action.explanation
            break
        if action.kind == ActionKind.NEW:
            if action.key is None:
                raise RuntimeError("new-question action lacks a feature key")
            observation, true_mode = patient.answer(action.key)
            learned_risk = 0.0
            if policy.verification_risk_gate is not None:
                learned_risk = policy.verification_risk_gate.probability(
                    tracker, observation
                )
            reports.append(observation)
            report_risks.append(learned_risk)
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
        if policy.oracle_correction:
            true_state = patient.latent_states.get(original.key, UNKNOWN)
            clarification = Observation(
                key=original.key,
                value=true_state,
                certainty=(
                    CertaintyCue.CERTAIN if true_state != UNKNOWN else CertaintyCue.NONE
                ),
            )
            true_mode = ReportMode.CERTAIN
            resolved = clarification
        else:
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
                verification_triggered_by_retrieval=action.retrieval_triggered,
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
        retrieval_log=tuple(retrieval_log),
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


def _readable_feature_names(tracker: BeliefTracker) -> dict:
    """Map each FeatureKey to a human-readable clinical term for retrieval.

    DDXPlus FeatureKey.name is an opaque evidence code ("E_146"); the readable
    text lives in ``VariableSpec.question`` (the official ``question_en``).  When
    the question carries no ASCII words (e.g. the toy model's Chinese prompts),
    fall back to the feature name, which the BM25 tokenizer can actually index.
    """
    mapping: dict = {}
    for key, spec in tracker.model.specs.items():
        question = spec.question or ""
        mapping[key] = (
            question if any(ch.isascii() and ch.isalnum() for ch in question) else key.name
        )
    return mapping


def _binary_entropy(probability: float) -> float:
    if probability <= 0 or probability >= 1:
        return 0.0
    return -probability * math.log2(probability) - (1 - probability) * math.log2(
        1 - probability
    )


def _oracle_wrong_reports(
    reports: tuple[Observation, ...],
    verified_report_indices: set[int],
    oracle_states: Mapping[FeatureKey, str],
) -> list[int]:
    """Indices of reports whose value disagrees with the true state.

    Oracle-only bookkeeping: it reads the simulator's privileged true state
    (``oracle_states``), so it must never be used by a deployable policy.
    """
    wrong: list[int] = []
    for index, report in enumerate(reports):
        if index in verified_report_indices or report.value == UNKNOWN:
            continue
        true_state = oracle_states.get(report.key, UNKNOWN)
        if true_state != UNKNOWN and report.value != true_state:
            wrong.append(index)
    return wrong
