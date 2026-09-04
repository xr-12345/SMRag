"""Joint-channel Brier-audit policy (Phase 8B).

This is the decision layer on top of :class:`JointReliabilityBeliefTracker`.  It
reuses the **corrected** Phase 8A audit logic (Prompt #18) -- gross verification
gain only *blocks* Stop, and AskNew / VerifyOld are compared on a net-Brier scale
with a global advantage margin ``tau_A`` -- but every quantity is now derived
from the ONE shared :class:`JointReportChannel`:

* AskNew value:  ``V_new(j)  = R_B(b_t) - E_{y~P(y|b_t)}[R_B(b_{t+1}^{j,y})] - C_new,j``
  where ``P(y|b_t)`` and the posterior both come from the joint tracker.
* VerifyOld value: ``V_verify(i) = R_B(b_t) - E_{y'~P(y'|H_t,Y_i,v)}[R_B(b_{t+1})] - C_verify``
  where the re-ask prediction ``P(y'|H_t,Y_i,v)`` is the shared channel's
  ``reask_predictive`` (not an independent second channel).
* Stop reliability: ``max_i p_i^mode <= suspicious_report_threshold`` (the joint
  misreport posterior), replacing the old leave-one-out ``score.score`` proxy.

The policy is **standalone** (not a ``ReliabilityAwareActionPolicy``) because it
needs the joint tracker rather than ``BeliefTracker``; it still reuses
``ActionKind`` / ``ActionScore`` / ``brier_risk``.  It never reads the true
disease, latent state, noise label, or true wrongness.
"""

from __future__ import annotations

from typing import Mapping

from .action_value import brier_risk
from .decision import (
    ActionKind,
    ActionScore,
    PolicyTurn,
    ReliabilityAwareDialogueResult,
    ReliabilityAwarePolicyConfig,
    StopAuditMode,
)
from .estimation import DiseaseStateModel
from .joint_reliability_belief import JointReliabilityBeliefTracker
from .joint_report_channel import JointReportChannel, VerificationType
from .schema import UNKNOWN, CertaintyCue, FeatureKey, Observation
from .simulator import StructuredPatientSimulator


class JointChannelBrierAuditPolicy:
    """Unified-Brier AskNew / VerifyOld / Stop on the joint reliability tracker.

    AskNew is valued on the Brier scale (``argmax_j V_new(j)`` is Brier-best,
    not EIG-best), VerifyOld is valued on the same Brier scale, and Stop uses
    the joint misreport posterior as its reliability check.
    """

    def __init__(
        self,
        model: DiseaseStateModel,
        *,
        channel: JointReportChannel | None = None,
        tracker: JointReliabilityBeliefTracker | None = None,
        config: ReliabilityAwarePolicyConfig | None = None,
        safety_constraint=None,
        initial_observations: tuple[Observation, ...] = (),
    ) -> None:
        self.model = model
        self.channel = channel or JointReportChannel()
        self.config = config or ReliabilityAwarePolicyConfig()
        self.safety_constraint = safety_constraint
        if tracker is not None:
            self.tracker = tracker
        else:
            self.tracker = JointReliabilityBeliefTracker(model, self.channel)
            for observation in initial_observations:
                self.tracker.observe_single(observation)
        self.last_asknew_log: list[dict] = []
        self.last_verification_log: list[dict] = []
        self._last_best_new: ActionScore | None = None
        self._last_best_verify: ActionScore | None = None
        self.last_decision_log: dict = {}

    # -- helpers ------------------------------------------------------------ #

    @property
    def asked(self) -> set[FeatureKey]:
        return set(self.tracker.memory.keys())

    def _verification_count(self) -> int:
        return sum(1 for bundle in self.tracker.memory.values() if bundle.is_verified)

    def _max_p_wrong(self) -> float | None:
        """``max_i p_i^wrong`` over asked features, skipping UNKNOWN non-responses.

        ``p_wrong`` is ``None`` for a first answer of ``UNKNOWN``; those rows are
        excluded (never coerced to 0.0 or 1.0).  Returns ``None`` when no asked
        feature has an explicit answer.
        """
        values = []
        for key in self.asked:
            wrong = self.tracker.p_wrong(key)
            if wrong is not None:
                values.append(wrong)
        return max(values) if values else None

    # -- AskNew: every unasked question on the Brier scale ------------------- #

    def _asknew_value(self, key: FeatureKey) -> float:
        spec = self.model.specs[key]
        c_new = self.config.new_question_cost_weight * spec.cost
        current_risk = brier_risk(self.tracker.belief)
        distribution = self.tracker.single_answer_predictive(key)
        expected_risk = 0.0
        for answer, probability in distribution.items():
            posterior = self.tracker.posterior_after_single(
                key, Observation(key=key, value=answer, certainty=CertaintyCue.NONE)
            )
            expected_risk += probability * brier_risk(posterior)
        return current_risk - expected_risk - c_new

    def _asknew_brier_values(self) -> list[tuple[FeatureKey, float]]:
        values: list[tuple[FeatureKey, float]] = []
        for key, spec in self.model.specs.items():
            if not spec.askable:
                continue
            if key in self.tracker.memory:
                continue
            values.append((key, self._asknew_value(key)))
        return values

    # -- VerifyOld: every unverified, non-UNKNOWN report --------------------- #

    def gross_verify_gain(self, key: FeatureKey) -> float:
        """``G_i^verify = R_B(b_t) - E_{y'} R_B(b_{t+1}^{(i,y')})``.

        The expected Brier-risk reduction of verifying ``key``, **before** the
        verification cost.  This is what the ``decision_value_audit`` stop mode
        compares against ``verification_audit_threshold``; it must never have the
        cost subtracted (see :meth:`net_verify_value`).
        """
        current_risk = brier_risk(self.tracker.belief)
        distribution = self.tracker.reask_predictive(key)
        expected_risk = 0.0
        for answer, probability in distribution.items():
            posterior = self.tracker.posterior_after_verification(
                key, Observation(key=key, value=answer, certainty=CertaintyCue.NONE)
            )
            expected_risk += probability * brier_risk(posterior)
        return current_risk - expected_risk

    def net_verify_value(self, key: FeatureKey) -> float:
        """``V_verify(i) = G_i^verify - C_verify``.

        The net VerifyOld value used in the AskNew / VerifyOld action comparison;
        the gross gain with the verification cost subtracted.
        """
        return self.gross_verify_gain(key) - self.config.verification_cost

    def _verify_value(self, key: FeatureKey) -> float:
        """Backward-compatible alias for :meth:`net_verify_value`."""
        return self.net_verify_value(key)

    def _verifyold_brier_values(self) -> list[tuple[FeatureKey, float, float]]:
        if self._verification_count() >= self.config.maximum_verifications:
            return []
        values: list[tuple[FeatureKey, float, float]] = []
        for key, bundle in self.tracker.memory.items():
            if bundle.is_verified or bundle.original.value == UNKNOWN:
                continue
            g_verify = self.gross_verify_gain(key)
            v_verify = g_verify - self.config.verification_cost
            values.append((key, v_verify, g_verify))
        return values

    # -- unified ranking ---------------------------------------------------- #

    def rank_actions(self) -> list[ActionScore]:
        actions: list[ActionScore] = []
        new_actions: list[ActionScore] = []
        verify_actions: list[ActionScore] = []

        asknew_log: list[dict] = []
        for key, v_new in self._asknew_brier_values():
            spec = self.model.specs[key]
            new_action = ActionScore(
                kind=ActionKind.NEW,
                utility=v_new,
                burden=self.config.new_question_cost_weight * spec.cost,
                key=key,
                explanation=(
                    "AskNew V_new = R_B(b_t) - E R_B(b_{t+1}) - C_new,j "
                    "(joint-channel Brier risk-reduction unit)"
                ),
            )
            actions.append(new_action)
            new_actions.append(new_action)
            asknew_log.append(
                {
                    "key": key.name,
                    "question_cost": round(float(spec.cost), 6),
                    "v_new": round(v_new, 6),
                }
            )
        self.last_asknew_log = asknew_log

        verification_log: list[dict] = []
        for key, v_verify, g_verify in self._verifyold_brier_values():
            verify_action = ActionScore(
                kind=ActionKind.VERIFY,
                utility=v_verify,
                burden=self.config.verification_cost,
                key=key,
                explanation=(
                    "VerifyOld V_verify = R_B(b_t) - E R_B(b_{t+1}) - C_verify "
                    "(joint-channel re-ask prediction)"
                ),
            )
            actions.append(verify_action)
            verify_actions.append(verify_action)
            verification_log.append(
                {
                    "key": key.name,
                    "v_verify": round(v_verify, 6),
                    "g_verify": round(g_verify, 6),
                }
            )
        self.last_verification_log = verification_log

        self._last_best_new = (
            max(new_actions, key=lambda action: action.utility) if new_actions else None
        )
        self._last_best_verify = (
            max(verify_actions, key=lambda action: action.utility)
            if verify_actions
            else None
        )
        return sorted(actions, key=lambda action: action.utility, reverse=True)

    # -- stop decision (corrected Phase 8A audit) ---------------------------- #

    def choose_action(self) -> ActionScore:
        self.tracker.disease_belief()
        actions = self.rank_actions()
        best = actions[0] if actions else None
        best_new = self._last_best_new
        best_verify = self._last_best_verify

        max_mode_misreport = max(
            (self.tracker.p_mode_misreported(key) for key in self.asked),
            default=0.0,
        )
        max_p_wrong = self._max_p_wrong()

        ranked = self.tracker.ranked_diseases()
        top_probability = ranked[0][1]
        margin = top_probability - (ranked[1][1] if len(ranked) > 1 else 0.0)
        confidence_ready = (
            top_probability >= self.config.posterior_threshold
            or margin >= self.config.posterior_margin_threshold
        )
        utility_low = best is None or best.utility <= self.config.minimum_action_utility
        probability_gate_ready = (
            max_mode_misreport <= self.config.suspicious_report_threshold
        )
        safety_ready = (
            self.safety_constraint is None
            or self.safety_constraint.is_clear(self.tracker, self.asked)
        )

        verification_count = self._verification_count()
        max_gross_verify_gain = (
            best_verify.utility + self.config.verification_cost
            if best_verify is not None
            and verification_count < self.config.maximum_verifications
            else 0.0
        )
        gross_gain_audit_blocks = (
            best_verify is not None
            and verification_count < self.config.maximum_verifications
            and max_gross_verify_gain > self.config.verification_audit_threshold
        )

        mode = self.config.stop_audit_mode
        if mode is StopAuditMode.HARD_PROBABILITY_GATE:
            # Old behaviour exactly: Stop needs BOTH the joint-misreport gate
            # (max p_mode <= tau_p) AND the gross-gain audit (G <= tau_V).
            stop_reliability_ready = (
                probability_gate_ready and not gross_gain_audit_blocks
            )
        elif mode is StopAuditMode.DECISION_VALUE_AUDIT:
            # New behaviour: only the gross verification gain gates Stop;
            # max p_mode is logged but no longer a stop condition.
            stop_reliability_ready = (
                max_gross_verify_gain <= self.config.verification_audit_threshold
            )
        else:  # pragma: no cover - defensive against future enum values
            raise ValueError(f"unknown stop audit mode: {mode!r}")

        # Backward-compatible combined readiness (old-mode semantics).
        reliability_ready = probability_gate_ready
        audit_blocks_stop = gross_gain_audit_blocks
        base_stop_ready = (
            confidence_ready and utility_low and reliability_ready and safety_ready
        )
        stop_ready = (
            confidence_ready and utility_low and stop_reliability_ready and safety_ready
        )

        if not confidence_ready:
            stop_block_reason = "confidence"
        elif not stop_reliability_ready:
            stop_block_reason = "reliability"
        elif not utility_low:
            stop_block_reason = "utility"
        elif not safety_ready:
            stop_block_reason = "safety"
        else:
            stop_block_reason = None

        self.last_decision_log = {
            # Backward-compatible fields (Phase 8A/8B/8C).
            "best_new_key": (best_new.key.name if best_new is not None else None),
            "best_new_net_value": (
                round(best_new.utility, 6) if best_new is not None else None
            ),
            "best_verify_key": (
                best_verify.key.name if best_verify is not None else None
            ),
            "best_verify_gross_gain": (
                round(max_gross_verify_gain, 6) if best_verify is not None else None
            ),
            "best_verify_net_value": (
                round(best_verify.utility, 6) if best_verify is not None else None
            ),
            "max_mode_misreport": round(max_mode_misreport, 6),
            "suspicious_report_threshold": self.config.suspicious_report_threshold,
            "reliability_ready": reliability_ready,
            "verification_audit_threshold": self.config.verification_audit_threshold,
            "verification_advantage_margin": self.config.verification_advantage_margin,
            "base_stop_ready": base_stop_ready,
            "audit_blocks_stop": audit_blocks_stop,
            # Phase 22A decision-value-audit fields.
            "stop_audit_mode": mode.value,
            "posterior_confidence": round(top_probability, 6),
            "posterior_margin": round(margin, 6),
            "max_p_mode": round(max_mode_misreport, 6),
            "max_p_wrong": (round(max_p_wrong, 6) if max_p_wrong is not None else None),
            "max_gross_verify_gain": round(max_gross_verify_gain, 6),
            "best_net_verify_value": (
                round(best_verify.utility, 6) if best_verify is not None else None
            ),
            "best_net_new_value": (
                round(best_new.utility, 6) if best_new is not None else None
            ),
            "stop_confidence_ready": confidence_ready,
            "stop_reliability_ready": stop_reliability_ready,
            "stop_utility_ready": utility_low,
            "stop_safety_ready": safety_ready,
            "stop_ready": stop_ready,
            "stop_block_reason": stop_block_reason,
        }

        if stop_ready:
            self.last_decision_log["chosen_action"] = ActionKind.STOP.value
            self.last_decision_log["chosen_index"] = None
            return ActionScore(
                kind=ActionKind.STOP,
                utility=0.0,
                explanation=(
                    "confidence, marginal utility, and joint-misreport checks passed"
                    if mode is StopAuditMode.HARD_PROBABILITY_GATE
                    else "confidence, marginal utility, and decision-value checks passed"
                ),
            )

        if best_new is not None and best_verify is not None:
            if (
                best_verify.utility
                > best_new.utility + self.config.verification_advantage_margin
            ):
                self.last_decision_log["chosen_action"] = ActionKind.VERIFY.value
                self.last_decision_log["chosen_index"] = best_verify.key.token
                return best_verify
            self.last_decision_log["chosen_action"] = ActionKind.NEW.value
            self.last_decision_log["chosen_index"] = best_new.key.token
            return best_new
        if best_new is not None:
            self.last_decision_log["chosen_action"] = ActionKind.NEW.value
            self.last_decision_log["chosen_index"] = best_new.key.token
            return best_new
        if best_verify is not None and best_verify.utility > 0:
            self.last_decision_log["chosen_action"] = ActionKind.VERIFY.value
            self.last_decision_log["chosen_index"] = best_verify.key.token
            return best_verify
        self.last_decision_log["chosen_action"] = ActionKind.STOP.value
        self.last_decision_log["chosen_index"] = None
        return ActionScore(
            kind=ActionKind.STOP,
            utility=0.0,
            explanation=(
                "no positive-utility acquisition action or a reliability/safety "
                "check remains; return uncertainty"
            ),
        )


# --------------------------------------------------------------------------- #
# Dialogue runner
# --------------------------------------------------------------------------- #


def run_joint_channel_dialogue(
    patient: StructuredPatientSimulator,
    *,
    policy: JointChannelBrierAuditPolicy | None = None,
) -> ReliabilityAwareDialogueResult:
    """Run AskNew / VerifyOld / Stop with the joint tracker (no UNKNOWN collapse).

    Unlike ``run_reliability_aware_dialogue``, a VerifyOld re-ask is recorded as
    a second answer in the feature's ``ReportBundle`` (via
    ``observe_verification``) -- the two answers are never overwritten and never
    compressed to ``UNKNOWN``.
    """
    policy = policy or JointChannelBrierAuditPolicy(patient.model)
    tracker = policy.tracker
    turns: list[PolicyTurn] = []
    verification_count = 0
    stop_reason = "total_turn_budget"
    uncertain_output = False

    while len(turns) < policy.config.max_total_turns:
        action = policy.choose_action()
        if action.kind is ActionKind.STOP:
            stop_reason = action.explanation
            uncertain_output = "uncertainty" in action.explanation
            break
        if action.kind is ActionKind.NEW:
            if action.key is None:
                raise RuntimeError("new-question action lacks a feature key")
            observation, true_mode = patient.answer(action.key)
            tracker.observe_single(observation)
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

        key = action.key
        if key is None or key not in tracker.memory:
            raise RuntimeError("verification action has an invalid feature key")
        original = tracker.memory[key].original
        clarification, true_mode = patient.answer(key)
        true_state = patient.latent_states.get(key, UNKNOWN)
        original_wrong = (
            true_state != UNKNOWN
            and original.value != UNKNOWN
            and original.value != true_state
        )
        resolved_wrong = (
            true_state != UNKNOWN
            and clarification.value != UNKNOWN
            and clarification.value != true_state
        )
        tracker.observe_verification(key, clarification, VerificationType.REPEAT)
        verification_count += 1
        turns.append(
            PolicyTurn(
                index=len(turns) + 1,
                action=action,
                observation=clarification,
                true_report_mode=true_mode,
                belief=dict(tracker.belief),
                verification_changed_report=(
                    clarification.value != original.value
                    or clarification.certainty != original.certainty
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
        new_questions=sum(turn.action.kind is ActionKind.NEW for turn in turns),
        verification_questions=sum(
            turn.action.kind is ActionKind.VERIFY for turn in turns
        ),
        uncertain_output=uncertain_output,
    )
