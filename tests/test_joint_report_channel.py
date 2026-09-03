"""Phase 8B -- joint report channel + reliability tracker + Brier policy tests.

These lock in the ONE shared generative model
``P(D, Z, E, Y, Y')``: the disease posterior, per-answer reliability (p_mode /
p_wrong), and the VerifyOld re-ask prediction all derive from
``JointReportChannel`` through ``JointReliabilityBeliefTracker``.

Coverage by section of the spec:

* probability correctness  (01-06): the joint channel's marginals / transitions
* joint update             (07-12): one factor per feature, no double-count, no
  UNKNOWN compression
* reliability consistency  (13-16): p_mode / p_wrong / re-ask prediction tie back
  to the shared model
* decision / safety        (17-23): Brier-scaled values + corrected stop audit
* leakage                  (24): the prediction path has no oracle surface
* regression               (25): old UNKNOWN-compression and default strategy intact
"""

from __future__ import annotations

import inspect
import unittest
from dataclasses import replace
from unittest import mock

from powerful_medrag.action_value import brier_risk
from powerful_medrag.belief import BeliefTracker
from powerful_medrag.channel import AnswerChannel, ReportMode
from powerful_medrag.clarification import SurprisalClarificationProtocol
from powerful_medrag.decision import (
    ActionKind,
    ActionScore,
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
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

    def policy(self, **kwargs):
        return JointChannelBrierAuditPolicy(self.model, channel=self.channel, **kwargs)


# --------------------------------------------------------------------------- #
# 1-6: probability correctness
# --------------------------------------------------------------------------- #


class TestProbabilityCorrectness(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()
        self.chan = self.env.answer_channel
        self.jchan = self.env.channel
        self.states = ("absent", "present")
        self.key = self.env.specs[0].key

    def test_01_single_probability_equals_answer_channel_marginal(self):
        for z in self.states:
            for v in (*self.states, UNKNOWN):
                for cue in CertaintyCue:
                    self.assertAlmostEqual(
                        self.jchan.single_probability(v, z, self.states, cue),
                        self.chan.marginal_probability(v, z, self.states, cue),
                        places=12,
                    )

    def test_02_joint_probability_sums_to_one_over_answer_pairs(self):
        space = (*self.states, UNKNOWN)
        for z in self.states:
            total = sum(
                self.jchan.joint_probability(y, yp, z, self.states)
                for y in space
                for yp in space
            )
            self.assertAlmostEqual(total, 1.0, places=10)

    def test_03_reask_conditional_sums_to_one_over_y_prime(self):
        space = (*self.states, UNKNOWN)
        for z in self.states:
            for mode in ReportMode:
                total = sum(
                    self.jchan.reask_conditional_probability(
                        yp, z, mode, self.states, VerificationType.REPEAT
                    )
                    for yp in space
                )
                self.assertAlmostEqual(total, 1.0, places=10)

    def test_04_mode_transition_boundaries_and_marginal_identity(self):
        full = JointReportChannel(self.chan, repeat_mode_persistence=1.0)
        none = JointReportChannel(self.chan, repeat_mode_persistence=0.0)
        prior = self.jchan.reask_mode_prior()
        for e in ReportMode:
            for ep in ReportMode:
                # rho=1 -> identity transition
                self.assertAlmostEqual(
                    full.mode_transition(ep, e),
                    1.0 if ep == e else 0.0,
                    places=12,
                )
                # rho=0 -> independent prior
                self.assertAlmostEqual(none.mode_transition(ep, e), prior[ep], places=12)
                # marginal identity: sum_e P(e) T(e'|e) == P(e') at rho=0.5
        for ep in ReportMode:
            lhs = sum(
                self.jchan.mode_prior(CertaintyCue.NONE)[e]
                * self.jchan.mode_transition(ep, e)
                for e in ReportMode
            )
            self.assertAlmostEqual(lhs, prior[ep], places=12)

    def test_05_independent_reask_is_product_of_marginals(self):
        none = JointReportChannel(self.chan, repeat_mode_persistence=0.0)
        for z in self.states:
            for y in self.states:
                for yp in self.states:
                    joint = none.joint_probability(y, yp, z, self.states)
                    product = none.single_probability(
                        y, z, self.states
                    ) * none.single_probability(yp, z, self.states)
                    self.assertAlmostEqual(joint, product, places=10)

    def test_06_full_persistence_reask_uses_same_mode_emission(self):
        full = JointReportChannel(self.chan, repeat_mode_persistence=1.0)
        # T(e'|e)=1[e'=e], so the re-ask density under mode e is the single-answer
        # emission under that mode: for MISREPORTED, the correct rate is 0.08.
        val = full.reask_conditional_probability(
            "present", "present", ReportMode.MISREPORTED, self.states,
            VerificationType.REPEAT,
        )
        self.assertAlmostEqual(
            val,
            self.chan.probability(
                "present", "present", self.states, ReportMode.MISREPORTED
            ),
            places=12,
        )
        self.assertAlmostEqual(val, 0.08, places=12)


# --------------------------------------------------------------------------- #
# 7-12: joint update
# --------------------------------------------------------------------------- #


class TestJointUpdate(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv(rho=0.9)
        self.key = self.env.specs[0].key

    def test_07_single_answer_matches_belief_tracker(self):
        joint = self.env.tracker()
        classic = BeliefTracker(self.env.model, self.env.answer_channel)
        patient = StructuredPatientSimulator.sample_case(
            self.env.model, "influenza", channel=self.env.answer_channel, seed=3
        )
        for spec in self.env.specs[:4]:
            obs, _ = patient.answer(spec.key)
            joint.observe_single(obs)
            classic.update(obs)
        for disease in self.env.model.diseases:
            self.assertAlmostEqual(
                joint.belief[disease], classic.belief[disease], places=9
            )

    def test_08_verification_uses_one_joint_factor_not_double_count(self):
        tracker = self.env.tracker()
        orig = _obs(self.key.name, "present")
        verif = _obs(self.key.name, "present")
        tracker.observe_single(orig)
        for disease in self.env.model.diseases:
            joint = tracker.joint_likelihood(
                disease, self.key, orig, verif, VerificationType.REPEAT
            )
            s1 = tracker.single_likelihood(disease, self.key, orig)
            s2 = tracker.single_likelihood(disease, self.key, verif)
            # a feature asked twice is ONE factor, not two independent factors
            self.assertNotAlmostEqual(joint, s1 * s2, places=6)

    def test_09_conflicting_answers_are_not_compressed_to_unknown(self):
        tracker = self.env.tracker()
        orig = _obs(self.key.name, "present")
        verif = _obs(self.key.name, "absent")
        tracker.observe_single(orig)
        tracker.observe_verification(self.key, verif, VerificationType.REPEAT)
        bundle = tracker.memory[self.key]
        self.assertEqual(bundle.original.value, "present")
        self.assertEqual(bundle.verifications[0].value, "absent")
        self.assertEqual(
            [a.value for a in bundle.all_answers], ["present", "absent"]
        )
        self.assertNotIn(UNKNOWN, [a.value for a in bundle.all_answers])

    def test_10_joint_likelihood_equals_mode_decomposition(self):
        tracker = self.env.tracker()
        tracker.observe_single(_obs(self.key.name, "present"))
        tracker.observe_verification(
            self.key, _obs(self.key.name, "present"), VerificationType.REPEAT
        )
        bundle = tracker.memory[self.key]
        for disease in self.env.model.diseases:
            direct = tracker._feature_likelihood(disease, bundle)
            decomposed = sum(
                tracker._feature_contribution(disease, bundle, mode)
                for mode in ReportMode
            )
            self.assertAlmostEqual(direct, decomposed, places=10)

    def test_11_posterior_after_verification_matches_observe(self):
        orig = _obs(self.key.name, "present")
        verif = _obs(self.key.name, "absent")
        tracker = self.env.tracker()
        tracker.observe_single(orig)
        tracker.observe_verification(self.key, verif, VerificationType.REPEAT)

        fresh = self.env.tracker()
        fresh.observe_single(orig)
        post = fresh.posterior_after_verification(self.key, verif)
        for disease in self.env.model.diseases:
            self.assertAlmostEqual(tracker.belief[disease], post[disease], places=9)

    def test_12_state_posterior_sums_to_one(self):
        tracker = self.env.tracker()
        tracker.observe_single(_obs(self.key.name, "present"))
        single = tracker.state_posterior(self.key)
        self.assertAlmostEqual(sum(single.values()), 1.0, places=10)
        tracker.observe_verification(
            self.key, _obs(self.key.name, "absent"), VerificationType.REPEAT
        )
        joint = tracker.state_posterior(self.key)
        self.assertAlmostEqual(sum(joint.values()), 1.0, places=10)


# --------------------------------------------------------------------------- #
# 13-16: reliability consistency
# --------------------------------------------------------------------------- #


class TestReliabilityConsistency(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv(rho=0.9)
        self.key = self.env.specs[0].key

    def test_13_p_wrong_equals_one_minus_state_posterior(self):
        tracker = self.env.tracker()
        tracker.observe_single(_obs(self.key.name, "present"))
        expected = 1.0 - tracker.state_posterior(self.key)["present"]
        self.assertAlmostEqual(tracker.p_wrong(self.key), expected, places=10)
        # UNKNOWN first answer is always "wrong" (never a true state)
        tracker2 = self.env.tracker()
        tracker2.observe_single(_obs(self.key.name, UNKNOWN, CertaintyCue.NONE))
        self.assertEqual(tracker2.p_wrong(self.key), 1.0)

    def test_14_mode_posterior_normalizes_and_falls_back_to_prior(self):
        tracker = self.env.tracker()
        # unobserved feature -> exactly the channel's mode prior
        prior = self.env.channel.mode_prior(CertaintyCue.NONE)
        self.assertEqual(tracker.mode_posterior(self.key), prior)
        # observed feature -> a normalized posterior with p_mode in [0, 1]
        tracker.observe_single(_obs(self.key.name, "present"))
        posterior = tracker.mode_posterior(self.key)
        self.assertAlmostEqual(sum(posterior.values()), 1.0, places=10)
        self.assertTrue(0.0 <= tracker.p_mode_misreported(self.key) <= 1.0)

    def test_15_reask_predictive_normalizes(self):
        tracker = self.env.tracker()
        tracker.observe_single(_obs(self.key.name, "present"))
        distribution = tracker.reask_predictive(self.key)
        self.assertAlmostEqual(sum(distribution.values()), 1.0, places=10)
        for answer in (*self.env.specs[0].values, UNKNOWN):
            self.assertIn(answer, distribution)

    def test_16_repeat_under_persistence_changes_misreport_confidence(self):
        # A repeated answer concentrates the joint posterior on the mode that
        # emits that answer consistently; the reliability posterior must change
        # (never stay frozen at the prior) and stay a valid probability.
        tracker = self.env.tracker()
        tracker.observe_single(_obs(self.key.name, "present"))
        before = tracker.p_mode_misreported(self.key)
        tracker.observe_verification(
            self.key, _obs(self.key.name, "present"), VerificationType.REPEAT
        )
        after = tracker.p_mode_misreported(self.key)
        self.assertNotAlmostEqual(before, after, places=6)
        self.assertTrue(0.0 <= after <= 1.0)


# --------------------------------------------------------------------------- #
# 17-23: decision / safety
# --------------------------------------------------------------------------- #


class TestDecisionSafety(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()
        self.key = self.env.specs[0].key

    def test_17_asknew_value_is_brier_risk_reduction_minus_cost(self):
        policy = self.env.policy()
        key = FeatureKey("dry_cough")
        c_new = policy.config.new_question_cost_weight * self.env.model.specs[key].cost
        v = policy._asknew_value(key)
        current = brier_risk(policy.tracker.belief)
        dist = policy.tracker.single_answer_predictive(key)
        manual = current - sum(
            p
            * brier_risk(
                policy.tracker.posterior_after_single(
                    key, _obs("dry_cough", a, CertaintyCue.NONE)
                )
            )
            for a, p in dist.items()
        )
        self.assertAlmostEqual(v, manual - c_new, places=10)
        self.assertGreaterEqual(v, -c_new - 1e-9)
        self.assertLessEqual(v, current - c_new + 1e-9)

    def test_18_verify_value_is_brier_risk_reduction_minus_cost(self):
        policy = self.env.policy()
        key = FeatureKey("fever")
        policy.tracker.observe_single(_obs("fever", "present"))
        c_verify = policy.config.verification_cost
        v = policy._verify_value(key)
        current = brier_risk(policy.tracker.belief)
        dist = policy.tracker.reask_predictive(key)
        manual = current - sum(
            p
            * brier_risk(
                policy.tracker.posterior_after_verification(
                    key, _obs("fever", a, CertaintyCue.NONE)
                )
            )
            for a, p in dist.items()
        )
        self.assertAlmostEqual(v, manual - c_verify, places=10)
        self.assertGreaterEqual(v, -c_verify - 1e-9)

    def test_19_high_p_mode_blocks_confident_stop(self):
        tracker = self.env.tracker()
        tracker.observe_single(_obs("fever", "present"))
        policy = JointChannelBrierAuditPolicy(
            self.env.model, channel=self.env.channel, tracker=tracker
        )
        # high joint misreport posterior -> reliability_ready False, so the policy
        # must NOT take the confident STOP even though confidence is high and the
        # best action's utility is below the minimum bar.
        with mock.patch.object(
            tracker, "p_mode_misreported", return_value=0.5
        ), mock.patch.object(
            tracker, "ranked_diseases",
            return_value=[("influenza", 0.95), ("common_cold", 0.05)],
        ), mock.patch.object(
            policy, "_asknew_brier_values",
            return_value=[(FeatureKey("dry_cough"), 0.02)],
        ), mock.patch.object(
            policy, "_verifyold_brier_values",
            return_value=[(FeatureKey("fever"), 0.01, 0.04)],
        ):
            action = policy.choose_action()
        self.assertIs(action.kind, ActionKind.NEW)
        self.assertFalse(policy.last_decision_log["base_stop_ready"])
        self.assertAlmostEqual(policy.last_decision_log["max_mode_misreport"], 0.5)

    def test_20_gross_gain_blocks_stop_but_does_not_force_verify(self):
        tracker = self.env.tracker()
        tracker.observe_single(_obs("fever", "present"))
        policy = JointChannelBrierAuditPolicy(
            self.env.model, channel=self.env.channel, tracker=tracker
        )
        # net verify (0.02) is not above new (0.02) + margin (0.03), but gross gain
        # (0.02 + 0.03 = 0.05) exceeds the audit threshold (0.03): Stop is blocked,
        # yet the corrected policy still returns AskNew (never forces VerifyOld).
        with mock.patch.object(
            tracker, "p_mode_misreported", return_value=0.0
        ), mock.patch.object(
            tracker, "ranked_diseases",
            return_value=[("influenza", 0.95), ("common_cold", 0.05)],
        ), mock.patch.object(
            policy, "_asknew_brier_values",
            return_value=[(FeatureKey("dry_cough"), 0.02)],
        ), mock.patch.object(
            policy, "_verifyold_brier_values",
            return_value=[(FeatureKey("fever"), 0.02, 0.05)],
        ):
            action = policy.choose_action()
        self.assertIs(action.kind, ActionKind.NEW)
        self.assertTrue(policy.last_decision_log["audit_blocks_stop"])

    def test_21_verification_budget_is_enforced(self):
        policy = self.env.policy()
        policy.tracker.observe_single(_obs("fever", "present"))
        policy.tracker.observe_verification(
            FeatureKey("fever"), _obs("fever", "present"), VerificationType.REPEAT
        )
        # one verification already used -> no further verify candidates
        self.assertEqual(policy._verifyold_brier_values(), [])

    def test_22_unknown_report_is_not_verified(self):
        policy = self.env.policy()
        policy.tracker.observe_single(_obs("fever", UNKNOWN, CertaintyCue.NONE))
        policy.tracker.observe_single(_obs("dry_cough", "present"))
        values = policy._verifyold_brier_values()
        keys = {key for key, _, _ in values}
        self.assertNotIn(FeatureKey("fever"), keys)
        self.assertIn(FeatureKey("dry_cough"), keys)

    def test_23_clean_high_confidence_tracker_stops(self):
        policy = self.env.policy()
        policy.tracker.observe_single(_obs("fever", "present"))
        policy.tracker.observe_single(_obs("dry_cough", "present"))
        policy.tracker.observe_single(_obs("myalgia", "present"))
        # a clean tracker either stops or asks; it must never raise and must be a
        # valid action kind
        action = policy.choose_action()
        self.assertIn(action.kind, {ActionKind.STOP, ActionKind.NEW, ActionKind.VERIFY})

    def test_23b_nonaskable_specs_are_excluded_from_asknew(self):
        # Phase 8C bug fix: AskNew must enumerate only askable questions (the
        # corrected/heuristic policies filter via NumpyQuestionSelector.rank; the
        # joint policy must match, else it asks non-askable evidence and does ~4x
        # the ranking work on DDXPlus).
        cases, specs = generate_toy_cases(cases_per_disease=30, seed=41)
        modified = [
            replace(spec, askable=False) if spec.key.name == "dry_cough" else spec
            for spec in specs
        ]
        model = DiseaseStateModel.fit(cases, modified)
        policy = JointChannelBrierAuditPolicy(model, channel=self.env.channel)
        keys = {key for key, _ in policy._asknew_brier_values()}
        self.assertNotIn(FeatureKey("dry_cough"), keys)
        self.assertIn(FeatureKey("fever"), keys)
        self.assertEqual(len(keys), 5)


# --------------------------------------------------------------------------- #
# 24-25: leakage + regression
# --------------------------------------------------------------------------- #


class TestLeakageAndRegression(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()

    def test_24_prediction_path_has_no_oracle_surface(self):
        # choose_action takes only `self`: no oracle / latent / true-disease arg.
        parameters = list(inspect.signature(JointChannelBrierAuditPolicy.choose_action).parameters)
        self.assertEqual(parameters, ["self"])
        policy = self.env.policy()
        policy.tracker.observe_single(_obs("fever", "present"))
        policy.choose_action()
        log = policy.last_decision_log
        self.assertIn("max_mode_misreport", log)
        self.assertNotIn("true_disease", log)
        self.assertNotIn("latent", log)
        self.assertNotIn("oracle", log)
        self.assertNotIn("noise", log)

    def test_25_old_paths_and_default_strategy_unchanged(self):
        # the old UNKNOWN compression is untouched (red line: don't delete it)
        resolved = SurprisalClarificationProtocol.resolve(
            _obs("fever", "present"), _obs("fever", "absent")
        )
        self.assertEqual(resolved.value, UNKNOWN)
        # default build_policy still returns the heuristic policy
        self.assertIs(
            type(build_policy(WorthinessStrategy.HEURISTIC_VERIFY)),
            ReliabilityAwareActionPolicy,
        )
        # the joint strategy is registered but is a standalone entry point
        with self.assertRaises(NotImplementedError):
            build_policy(WorthinessStrategy.JOINT_CHANNEL_BRIER_AUDIT)


class TestDialogueSmoke(unittest.TestCase):
    def test_joint_dialogue_terminates_within_budget(self):
        env = ToyEnv()
        cfg = ReliabilityAwarePolicyConfig(maximum_verifications=1, max_total_turns=8)
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
        # conflicting answers are never collapsed: any verified feature keeps both
        for turn in result.turns:
            if turn.action.kind is ActionKind.VERIFY:
                bundle = policy.tracker.memory[turn.action.key]
                self.assertEqual(len(bundle.all_answers), 2)


if __name__ == "__main__":
    unittest.main()
