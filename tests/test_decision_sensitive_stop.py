"""Phase 22A -- decision-sensitive stop (two stop-audit modes).

Locks in the Prompt #22A change: ``StopAuditMode`` splits the joint policy's
confident-Stop reliability check into two selectable modes, without running any
formal experiment.

* ``HARD_PROBABILITY_GATE`` (default) reproduces the pre-22A behaviour exactly:
  Stop needs BOTH ``max_i p_i^mode <= suspicious_report_threshold`` AND the
  gross verification-gain audit ``G_t^verify <= verification_audit_threshold``.
* ``DECISION_VALUE_AUDIT`` gates Stop on the gross gain alone
  (``G_t^verify <= tau_V``, tau_V = 0.03); ``max_i p_i^mode`` is logged only.

Coverage by spec section 8 (18 categories):

 1.  default mode is ``hard_probability_gate``
 2.  old mode behaviour byte-identical (gross gain = net value + cost; log stays)
 3.  max_p_mode > tau_p blocks old-mode Stop
 4.  new mode is NOT blocked by max_p_mode > tau_p alone
 5.  gross gain <= tau_V allows Stop
 6.  gross gain > tau_V blocks Stop
 7.  strict threshold boundaries (tau_V and tau_p)
 8.  gross gain does not subtract verification cost
 9.  net value subtracts verification cost
 10. blocking Stop does not force VerifyOld
 11. AskNew chosen when its net value is higher
 12. VerifyOld chosen only when > AskNew + tau_A
 13. gross gain is 0 when there is no verifiable report
 14. UNKNOWN is not a VerifyOld candidate
 15. maximum verification budget is enforced
 16. decision log has no privileged (truth/noise) fields; UNKNOWN p_wrong empty
 17. heuristic baseline is unchanged
 18. legacy joint mode trajectory regression (golden, byte-identical)
"""

from __future__ import annotations

import inspect
import unittest
from unittest import mock

from powerful_medrag.action_value import brier_risk
from powerful_medrag.channel import AnswerChannel
from powerful_medrag.decision import (
    ActionKind,
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    StopAuditMode,
)
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.joint_channel_policy import (
    JointChannelBrierAuditPolicy,
    run_joint_channel_dialogue,
)
from powerful_medrag.joint_reliability_belief import JointReliabilityBeliefTracker
from powerful_medrag.joint_report_channel import JointReportChannel, VerificationType
from powerful_medrag.schema import UNKNOWN, CertaintyCue, FeatureKey, Observation
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator
from powerful_medrag.worthiness_policy import WorthinessStrategy, build_policy


def _obs(name, value, certainty=CertaintyCue.CERTAIN):
    return Observation(FeatureKey(name), value, certainty=certainty)


class ToyEnv:
    def __init__(self, *, cases_per_disease=30, rho=0.5):
        cases, specs = generate_toy_cases(cases_per_disease=cases_per_disease, seed=41)
        self.model = DiseaseStateModel.fit(cases, specs)
        self.answer_channel = AnswerChannel()
        self.channel = JointReportChannel(
            self.answer_channel, repeat_mode_persistence=rho
        )
        self.cases = cases
        self.specs = specs

    def tracker(self):
        return JointReliabilityBeliefTracker(self.model, self.channel)

    def policy(self, *, mode=StopAuditMode.HARD_PROBABILITY_GATE, **kwargs):
        cfg = ReliabilityAwarePolicyConfig(stop_audit_mode=mode)
        return JointChannelBrierAuditPolicy(
            self.model, channel=self.channel, config=cfg, **kwargs
        )


def _choose(policy, tracker, *, p_mode=0.0, verify=(), asknew=(), ranked=None):
    """Drive ``choose_action`` deterministically under mocked reliability signals.

    ``verify`` is a list of ``(key, v_verify, g_verify)`` triples as produced by
    ``_verifyold_brier_values``; ``asknew`` is ``(key, v_new)`` pairs.  ``ranked``
    defaults to a high-confidence top-two so ``confidence_ready`` is True.
    """
    ranked = ranked or [("influenza", 0.95), ("common_cold", 0.05)]
    with mock.patch.object(tracker, "p_mode_misreported", return_value=p_mode), mock.patch.object(
        tracker, "ranked_diseases", return_value=ranked
    ), mock.patch.object(
        policy, "_asknew_brier_values", return_value=list(asknew)
    ), mock.patch.object(
        policy, "_verifyold_brier_values", return_value=list(verify)
    ):
        return policy.choose_action()


class _PolicyFixture:
    """A tracker that has observed one feature, plus a policy in a chosen mode."""

    def __init__(self, env, *, mode=StopAuditMode.HARD_PROBABILITY_GATE):
        self.tracker = env.tracker()
        self.tracker.observe_single(_obs("fever", "present"))
        self.policy = env.policy(mode=mode, tracker=self.tracker)


class TestDecisionSensitiveStop(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()

    # 1 -------------------------------------------------------------------- #
    def test_01_default_mode_is_hard_probability_gate(self):
        cfg = ReliabilityAwarePolicyConfig()
        self.assertIs(cfg.stop_audit_mode, StopAuditMode.HARD_PROBABILITY_GATE)
        self.assertEqual(cfg.stop_audit_mode.value, "hard_probability_gate")
        policy = self.env.policy(mode=StopAuditMode.HARD_PROBABILITY_GATE)
        self.assertIs(policy.config.stop_audit_mode, StopAuditMode.HARD_PROBABILITY_GATE)
        # the new enum is a plain str Enum so configs stay JSON-serialisable
        self.assertEqual(StopAuditMode.DECISION_VALUE_AUDIT.value, "decision_value_audit")

    # 2 -------------------------------------------------------------------- #
    def test_02_old_mode_gross_gain_is_net_plus_cost(self):
        # The pre-22A `_verify_value` was the NET value (cost already inside);
        # the old mode must keep that semantics: gross = net + cost.
        fix = _PolicyFixture(self.env, mode=StopAuditMode.HARD_PROBABILITY_GATE)
        key = FeatureKey("fever")
        gross = fix.policy.gross_verify_gain(key)
        net = fix.policy.net_verify_value(key)
        self.assertAlmostEqual(gross, net + self.env.policy().config.verification_cost, places=10)
        # `_verify_value` remains the net alias for backward compatibility
        self.assertAlmostEqual(fix.policy._verify_value(key), net, places=10)

    # 3 -------------------------------------------------------------------- #
    def test_03_high_p_mode_blocks_hard_gate_stop(self):
        fix = _PolicyFixture(self.env, mode=StopAuditMode.HARD_PROBABILITY_GATE)
        # gross gain 0.02 (low) but p_mode 0.5 (high): the probability gate alone
        # must block the confident Stop, and the policy falls through to AskNew.
        action = _choose(
            fix.policy, fix.tracker, p_mode=0.5,
            asknew=[(FeatureKey("dry_cough"), 0.02)],
            verify=[(FeatureKey("fever"), -0.01, 0.02)],
        )
        self.assertIs(action.kind, ActionKind.NEW)
        self.assertFalse(fix.policy.last_decision_log["stop_reliability_ready"])
        self.assertAlmostEqual(fix.policy.last_decision_log["max_p_mode"], 0.5)

    # 4 -------------------------------------------------------------------- #
    def test_04_decision_mode_ignores_high_p_mode_alone(self):
        fix = _PolicyFixture(self.env, mode=StopAuditMode.DECISION_VALUE_AUDIT)
        # Same signals as test_03: p_mode high, gross gain low. The decision-value
        # audit must NOT block Stop (p_mode is only logged), so it Stops.
        action = _choose(
            fix.policy, fix.tracker, p_mode=0.5,
            asknew=[(FeatureKey("dry_cough"), 0.02)],
            verify=[(FeatureKey("fever"), -0.01, 0.02)],
        )
        self.assertIs(action.kind, ActionKind.STOP)
        self.assertTrue(fix.policy.last_decision_log["stop_reliability_ready"])
        self.assertAlmostEqual(fix.policy.last_decision_log["max_p_mode"], 0.5)
        self.assertEqual(
            fix.policy.last_decision_log["stop_audit_mode"], "decision_value_audit"
        )

    # 5 -------------------------------------------------------------------- #
    def test_05_gross_gain_at_or_below_tau_v_allows_stop(self):
        fix = _PolicyFixture(self.env, mode=StopAuditMode.DECISION_VALUE_AUDIT)
        # gross gain 0.03 == tau_V (<= allowed) -> Stop
        action = _choose(
            fix.policy, fix.tracker, p_mode=0.0,
            asknew=[(FeatureKey("dry_cough"), 0.02)],
            verify=[(FeatureKey("fever"), 0.0, 0.03)],
        )
        self.assertIs(action.kind, ActionKind.STOP)
        self.assertTrue(fix.policy.last_decision_log["stop_reliability_ready"])

    # 6 -------------------------------------------------------------------- #
    def test_06_gross_gain_above_tau_v_blocks_stop(self):
        fix = _PolicyFixture(self.env, mode=StopAuditMode.DECISION_VALUE_AUDIT)
        action = _choose(
            fix.policy, fix.tracker, p_mode=0.0,
            asknew=[(FeatureKey("dry_cough"), 0.02)],
            verify=[(FeatureKey("fever"), 0.02, 0.05)],
        )
        self.assertIs(action.kind, ActionKind.NEW)
        self.assertFalse(fix.policy.last_decision_log["stop_reliability_ready"])
        self.assertAlmostEqual(fix.policy.last_decision_log["max_gross_verify_gain"], 0.05)

    # 7 -------------------------------------------------------------------- #
    def test_07_strict_threshold_boundaries(self):
        # tau_V = 0.03: gross gain exactly 0.03 allows; anything above blocks.
        for gross, expect_stop in [(0.03, True), (0.031, False)]:
            fix = _PolicyFixture(self.env, mode=StopAuditMode.DECISION_VALUE_AUDIT)
            v_verify = gross - 0.03
            action = _choose(
                fix.policy, fix.tracker, p_mode=0.0,
                asknew=[(FeatureKey("dry_cough"), 0.02)],
                verify=[(FeatureKey("fever"), v_verify, gross)],
            )
            self.assertIs(
                action.kind, ActionKind.STOP if expect_stop else ActionKind.NEW
            )
        # tau_p = 0.05: p_mode exactly 0.05 allows; anything above blocks.
        for p_mode, expect_stop in [(0.05, True), (0.051, False)]:
            fix = _PolicyFixture(self.env, mode=StopAuditMode.HARD_PROBABILITY_GATE)
            action = _choose(
                fix.policy, fix.tracker, p_mode=p_mode,
                asknew=[(FeatureKey("dry_cough"), 0.02)],
                verify=[(FeatureKey("fever"), -0.01, 0.02)],
            )
            self.assertIs(
                action.kind, ActionKind.STOP if expect_stop else ActionKind.NEW
            )

    # 8 -------------------------------------------------------------------- #
    def test_08_gross_gain_excludes_verification_cost(self):
        policy = self.env.policy()
        policy.tracker.observe_single(_obs("fever", "present"))
        key = FeatureKey("fever")
        gross = policy.gross_verify_gain(key)
        # manual gross = R_B(b_t) - E R_B(b_{t+1}), with NO cost subtracted
        current = brier_risk(policy.tracker.belief)
        dist = policy.tracker.reask_predictive(key)
        manual = current - sum(
            p * brier_risk(
                policy.tracker.posterior_after_verification(
                    key, _obs("fever", a, CertaintyCue.NONE)
                )
            )
            for a, p in dist.items()
        )
        self.assertAlmostEqual(gross, manual, places=10)

    # 9 -------------------------------------------------------------------- #
    def test_09_net_value_subtracts_cost(self):
        policy = self.env.policy()
        policy.tracker.observe_single(_obs("fever", "present"))
        key = FeatureKey("fever")
        gross = policy.gross_verify_gain(key)
        net = policy.net_verify_value(key)
        self.assertAlmostEqual(net, gross - policy.config.verification_cost, places=10)
        # the action-comparison candidate carries the NET value (v = g - cost)
        row = next(r for r in policy._verifyold_brier_values() if r[0] == key)
        self.assertAlmostEqual(row[1], row[2] - policy.config.verification_cost, places=10)

    # 10 ------------------------------------------------------------------- #
    def test_10_blocked_stop_does_not_force_verifyold(self):
        fix = _PolicyFixture(self.env, mode=StopAuditMode.DECISION_VALUE_AUDIT)
        # gross gain 0.05 blocks Stop, but net verify (0.02) is not above
        # AskNew (0.02) + tau_A (0.03), so the policy must ask NEW, never verify.
        action = _choose(
            fix.policy, fix.tracker, p_mode=0.0,
            asknew=[(FeatureKey("dry_cough"), 0.02)],
            verify=[(FeatureKey("fever"), 0.02, 0.05)],
        )
        self.assertIs(action.kind, ActionKind.NEW)
        self.assertFalse(fix.policy.last_decision_log["stop_reliability_ready"])

    # 11 ------------------------------------------------------------------- #
    def test_11_asknew_chosen_when_net_value_higher(self):
        fix = _PolicyFixture(self.env, mode=StopAuditMode.HARD_PROBABILITY_GATE)
        # net verify 0.05 is NOT > new 0.08 + tau_A 0.03 -> AskNew wins.
        action = _choose(
            fix.policy, fix.tracker, p_mode=0.0,
            ranked=[("influenza", 0.5), ("common_cold", 0.5)],
            asknew=[(FeatureKey("dry_cough"), 0.08)],
            verify=[(FeatureKey("fever"), 0.05, 0.08)],
        )
        self.assertIs(action.kind, ActionKind.NEW)
        self.assertEqual(
            fix.policy.last_decision_log["chosen_index"], FeatureKey("dry_cough").token
        )

    # 12 ------------------------------------------------------------------- #
    def test_12_verifyold_only_with_advantage_margin(self):
        # verify net 0.10 > new 0.06 + tau_A 0.03 -> VerifyOld
        fix = _PolicyFixture(self.env, mode=StopAuditMode.HARD_PROBABILITY_GATE)
        action = _choose(
            fix.policy, fix.tracker, p_mode=0.0,
            ranked=[("influenza", 0.5), ("common_cold", 0.5)],
            asknew=[(FeatureKey("dry_cough"), 0.06)],
            verify=[(FeatureKey("fever"), 0.10, 0.13)],
        )
        self.assertIs(action.kind, ActionKind.VERIFY)
        self.assertEqual(
            fix.policy.last_decision_log["chosen_index"], FeatureKey("fever").token
        )
        # boundary: verify net 0.09 == new 0.06 + 0.03 is NOT strictly greater -> New
        fix2 = _PolicyFixture(self.env, mode=StopAuditMode.HARD_PROBABILITY_GATE)
        action2 = _choose(
            fix2.policy, fix2.tracker, p_mode=0.0,
            ranked=[("influenza", 0.5), ("common_cold", 0.5)],
            asknew=[(FeatureKey("dry_cough"), 0.06)],
            verify=[(FeatureKey("fever"), 0.09, 0.12)],
        )
        self.assertIs(action2.kind, ActionKind.NEW)

    # 13 ------------------------------------------------------------------- #
    def test_13_no_verifiable_report_yields_zero_gross_gain(self):
        tracker = self.env.tracker()
        tracker.observe_single(_obs("fever", "present"))
        tracker.observe_verification(
            FeatureKey("fever"), _obs("fever", "present"), VerificationType.REPEAT
        )
        # maximum_verifications=1 already used -> no verify candidates remain
        policy = self.env.policy(tracker=tracker)
        self.assertEqual(policy._verifyold_brier_values(), [])
        with mock.patch.object(tracker, "p_mode_misreported", return_value=0.0), mock.patch.object(
            tracker, "ranked_diseases",
            return_value=[("influenza", 0.95), ("common_cold", 0.05)],
        ), mock.patch.object(
            policy, "_asknew_brier_values",
            return_value=[(FeatureKey("dry_cough"), 0.02)],
        ):
            policy.choose_action()
        self.assertEqual(policy.last_decision_log["max_gross_verify_gain"], 0.0)
        self.assertIsNone(policy.last_decision_log["best_verify_key"])

    # 14 ------------------------------------------------------------------- #
    def test_14_unknown_report_is_not_verifyold_candidate(self):
        policy = self.env.policy()
        policy.tracker.observe_single(_obs("fever", UNKNOWN, CertaintyCue.NONE))
        policy.tracker.observe_single(_obs("dry_cough", "present"))
        keys = {key for key, _, _ in policy._verifyold_brier_values()}
        self.assertNotIn(FeatureKey("fever"), keys)
        self.assertIn(FeatureKey("dry_cough"), keys)

    # 15 ------------------------------------------------------------------- #
    def test_15_verification_budget_enforced(self):
        policy = self.env.policy()
        policy.tracker.observe_single(_obs("fever", "present"))
        policy.tracker.observe_verification(
            FeatureKey("fever"), _obs("fever", "present"), VerificationType.REPEAT
        )
        self.assertEqual(policy._verifyold_brier_values(), [])
        self.assertEqual(policy._verification_count(), 1)

    # 16 ------------------------------------------------------------------- #
    def test_16_log_has_no_privileged_fields(self):
        policy = self.env.policy()
        policy.tracker.observe_single(_obs("fever", UNKNOWN, CertaintyCue.NONE))
        policy.choose_action()
        log = policy.last_decision_log
        for banned in (
            "true_disease", "true_state", "true_mode", "true_wrong",
            "latent", "oracle", "noise", "clean", "misreport_label",
        ):
            self.assertNotIn(banned, log)
        # UNKNOWN non-response -> max_p_wrong logged as None (empty), never 0/1
        self.assertIsNone(log["max_p_wrong"])
        for field in (
            "stop_audit_mode", "max_gross_verify_gain", "max_p_mode",
            "max_p_wrong", "stop_reliability_ready", "chosen_action",
        ):
            self.assertIn(field, log)

    # 17 ------------------------------------------------------------------- #
    def test_17_heuristic_baseline_unchanged(self):
        self.assertIs(
            type(build_policy(WorthinessStrategy.HEURISTIC_VERIFY)),
            ReliabilityAwareActionPolicy,
        )
        cfg = ReliabilityAwarePolicyConfig()
        # frozen thresholds unchanged by the new mode field
        self.assertEqual(cfg.suspicious_report_threshold, 0.05)
        self.assertEqual(cfg.verification_audit_threshold, 0.03)
        self.assertEqual(cfg.verification_advantage_margin, 0.03)
        self.assertEqual(cfg.verification_cost, 0.03)
        self.assertEqual(cfg.maximum_verifications, 1)

    # 18 ------------------------------------------------------------------- #
    def test_18_legacy_joint_mode_trajectory_regression(self):
        # Golden trajectories captured from the pre-22A code (git HEAD) and
        # confirmed byte-identical under the default HARD_PROBABILITY_GATE mode.
        golden = [
            ("influenza", 3, "newnew", "influenza", 2, 0),
            ("common_cold", 2, "newnewnewnewnewnewverify", "common_cold", 6, 1),
        ]
        for diag, seed, expected_seq, expected_pred, nq, vq in golden:
            env = ToyEnv()
            cfg = ReliabilityAwarePolicyConfig(
                maximum_verifications=1, max_total_turns=8
            )
            policy = JointChannelBrierAuditPolicy(
                env.model, channel=env.channel, config=cfg
            )
            case = next(c for c in env.cases if c.diagnosis == diag)
            patient = StructuredPatientSimulator(
                diagnosis=case.diagnosis,
                latent_states=case.states,
                model=env.model,
                channel=env.answer_channel,
                profile=PatientProfile.from_noise_rate(0.3),
                seed=seed,
            )
            result = run_joint_channel_dialogue(patient, policy=policy)
            self.assertEqual(
                "".join(t.action.kind.value for t in result.turns), expected_seq
            )
            self.assertEqual(result.predicted_diagnosis, expected_pred)
            self.assertEqual(result.new_questions, nq)
            self.assertEqual(result.verification_questions, vq)


class TestDecisionValueModeSmoke(unittest.TestCase):
    def test_decision_value_dialogue_terminates_within_budget(self):
        env = ToyEnv()
        cfg = ReliabilityAwarePolicyConfig(
            maximum_verifications=1,
            max_total_turns=8,
            stop_audit_mode=StopAuditMode.DECISION_VALUE_AUDIT,
        )
        policy = JointChannelBrierAuditPolicy(env.model, channel=env.channel, config=cfg)
        case = next(c for c in env.cases if c.diagnosis == "influenza")
        patient = StructuredPatientSimulator(
            diagnosis=case.diagnosis,
            latent_states=case.states,
            model=env.model,
            channel=env.answer_channel,
            profile=PatientProfile.from_noise_rate(0.3),
            seed=9,
        )
        result = run_joint_channel_dialogue(patient, policy=policy)
        self.assertLessEqual(len(result.turns), cfg.max_total_turns)
        self.assertLessEqual(result.verification_questions, cfg.maximum_verifications)


if __name__ == "__main__":
    unittest.main()
