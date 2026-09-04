"""Phase 8E -- UNKNOWN semantic fix + protocol_fixed (train_fixed) prior.

Locks in the two changes from Prompt #21:

1. ``p_wrong`` returns ``None`` (undefined) for a ``UNKNOWN`` first answer, and
   ``is_nonresponse`` flags the explicit non-response ``u_i = 1[Y_i = UNKNOWN]``.
   A non-response is not a wrong answer, so it must never enter wrong-report
   ECE/Brier and its rate must be counted separately.
2. ``protocol_fixed_prior`` / ``protocol_fixed_channel_parameters`` expose the
   Phase 8D "train_fixed" prior (cross-noise-average, not cue-conditioned, not
   learned) as an explicit opt-in config, leaving the legacy default untouched.

Coverage by spec category (Prompt #21 section 5):

1.  PRESENT/ABSENT p_wrong stays a valid [0,1] probability
2.  categorical explicit answers compute normally
3.  UNKNOWN returns None/NaN
4.  UNKNOWN excluded from wrong-report ECE/Brier
5.  UNKNOWN rate counted separately
6.  UNKNOWN is not a VerifyOld candidate
7.  Stop audited by p_mode, not affected by UNKNOWN's p_wrong
8.  legacy prior old-path regression
9.  protocol_fixed prior source fixed
10. inference code reads no latent/true/noise/clean info
11. UNKNOWN fix leaves the strategy trajectory identical
12. all old tests pass (covered by running the full suite)
"""

from __future__ import annotations

import inspect
import unittest
from unittest import mock

from powerful_medrag.channel import (
    AnswerChannel,
    ChannelParameters,
    ReportMode,
    protocol_fixed_channel_parameters,
    protocol_fixed_prior,
)
from powerful_medrag.decision import ActionKind, ReliabilityAwarePolicyConfig
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.joint_channel_policy import (
    JointChannelBrierAuditPolicy,
    run_joint_channel_dialogue,
)
from powerful_medrag.joint_reliability_belief import JointReliabilityBeliefTracker
from powerful_medrag.joint_report_channel import JointReportChannel
from powerful_medrag.schema import (
    UNKNOWN,
    CertaintyCue,
    FeatureKey,
    Observation,
    VariableSpec,
)
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator


def _obs(name, value, certainty=CertaintyCue.CERTAIN):
    return Observation(FeatureKey(name), value, certainty=certainty)


class _Env:
    def __init__(self, cases_per_disease=30, rho=0.5):
        cases, specs = generate_toy_cases(cases_per_disease=cases_per_disease, seed=41)
        self.model = DiseaseStateModel.fit(cases, specs)
        self.answer_channel = AnswerChannel()
        self.channel = JointReportChannel(self.answer_channel, repeat_mode_persistence=rho)
        self.cases = cases
        self.specs = specs

    def tracker(self):
        return JointReliabilityBeliefTracker(self.model, self.channel)

    def policy(self):
        return JointChannelBrierAuditPolicy(self.model, channel=self.channel)


def _categorical_model():
    """A two-disease model with a 3-valued categorical feature ``severity``."""
    cat_key = FeatureKey("severity")
    cat_spec = VariableSpec(key=cat_key, values=("mild", "moderate", "severe"))
    disease_counts = {"d_a": 10, "d_b": 10}
    state_counts = {
        "d_a": {cat_key: {"mild": 8, "moderate": 1, "severe": 1}},
        "d_b": {cat_key: {"mild": 1, "moderate": 1, "severe": 8}},
    }
    model = DiseaseStateModel.from_counts(
        specs=[cat_spec], disease_counts=disease_counts, state_counts=state_counts
    )
    return model, cat_key


class TestUnknownSemantics(unittest.TestCase):
    def setUp(self):
        self.env = _Env()

    # 1 -------------------------------------------------------------------- #
    def test_01_present_absent_p_wrong_is_valid_probability(self):
        for value in ("present", "absent"):
            tracker = self.env.tracker()
            tracker.observe_single(_obs("fever", value))
            p = tracker.p_wrong(FeatureKey("fever"))
            self.assertIsNotNone(p)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    # 2 -------------------------------------------------------------------- #
    def test_02_categorical_p_wrong_computes_normally(self):
        model, cat_key = _categorical_model()
        tracker = JointReliabilityBeliefTracker(model, self.env.channel)
        tracker.observe_single(Observation(cat_key, "moderate", CertaintyCue.NONE))
        p = tracker.p_wrong(cat_key)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p, 1.0 - tracker.state_posterior(cat_key)["moderate"], places=10)
        self.assertFalse(tracker.is_nonresponse(cat_key))

    # 3 -------------------------------------------------------------------- #
    def test_03_unknown_p_wrong_returns_none(self):
        tracker = self.env.tracker()
        tracker.observe_single(_obs("fever", UNKNOWN, CertaintyCue.NONE))
        self.assertIsNone(tracker.p_wrong(FeatureKey("fever")))
        self.assertTrue(tracker.is_nonresponse(FeatureKey("fever")))

    # 4 -------------------------------------------------------------------- #
    def test_04_unknown_excluded_from_wrong_report_brier(self):
        # Wrong-report Brier/ECE must skip rows where p_wrong is None (UNKNOWN).
        tracker = self.env.tracker()
        tracker.observe_single(_obs("fever", "present"))
        tracker.observe_single(_obs("dry_cough", UNKNOWN, CertaintyCue.NONE))
        explicit = [k for k in (FeatureKey("fever"), FeatureKey("dry_cough"))
                    if not tracker.is_nonresponse(k)]
        # only the explicit-answer row survives the non-response filter
        self.assertEqual(explicit, [FeatureKey("fever")])
        # a mini Brier over wrong-report only ever sees the explicit row
        pairs = [(tracker.p_wrong(k), 0.0) for k in explicit]
        brier = sum((p - t) ** 2 for p, t in pairs)
        self.assertGreaterEqual(brier, 0.0)
        self.assertLessEqual(brier, 1.0)

    # 5 -------------------------------------------------------------------- #
    def test_05_unknown_rate_counted_separately(self):
        tracker = self.env.tracker()
        tracker.observe_single(_obs("fever", "present"))
        tracker.observe_single(_obs("dry_cough", UNKNOWN, CertaintyCue.NONE))
        tracker.observe_single(_obs("myalgia", "absent"))
        asked = [FeatureKey("fever"), FeatureKey("dry_cough"), FeatureKey("myalgia")]
        unknown_rate = sum(1 for k in asked if tracker.is_nonresponse(k)) / len(asked)
        self.assertAlmostEqual(unknown_rate, 1.0 / 3.0)
        self.assertIsNotNone(tracker.p_wrong(FeatureKey("fever")))
        self.assertIsNone(tracker.p_wrong(FeatureKey("dry_cough")))
        self.assertIsNotNone(tracker.p_wrong(FeatureKey("myalgia")))

    # 6 -------------------------------------------------------------------- #
    def test_06_unknown_is_not_verifyold_candidate(self):
        policy = self.env.policy()
        policy.tracker.observe_single(_obs("fever", UNKNOWN, CertaintyCue.NONE))
        policy.tracker.observe_single(_obs("dry_cough", "present"))
        keys = {k for k, _, _ in policy._verifyold_brier_values()}
        self.assertNotIn(FeatureKey("fever"), keys)
        self.assertIn(FeatureKey("dry_cough"), keys)

    # 7 -------------------------------------------------------------------- #
    def test_07_stop_audited_by_p_mode_not_p_wrong(self):
        tracker = self.env.tracker()
        tracker.observe_single(_obs("fever", "present"))
        policy = JointChannelBrierAuditPolicy(
            self.env.model, channel=self.env.channel, tracker=tracker
        )
        # Stop's reliability check reads p_mode_misreported. p_wrong is only
        # *logged* (max_p_wrong, Phase 22A): its value -- including None for an
        # UNKNOWN non-response -- never drives the stop decision. Patch it to the
        # UNKNOWN extreme (None) and patch p_mode high: the decision must still
        # come from p_mode, and the log must record an empty max_p_wrong.
        with mock.patch.object(tracker, "p_wrong", return_value=None), mock.patch.object(
            tracker, "p_mode_misreported", return_value=0.5
        ):
            action = policy.choose_action()
        self.assertIn(action.kind, {ActionKind.STOP, ActionKind.NEW, ActionKind.VERIFY})
        self.assertEqual(policy.last_decision_log["max_mode_misreport"], 0.5)
        self.assertIsNone(policy.last_decision_log["max_p_wrong"])
        self.assertFalse(policy.last_decision_log["stop_reliability_ready"])

    # 8 -------------------------------------------------------------------- #
    def test_08_legacy_prior_old_path_regression(self):
        tracker = self.env.tracker()
        tracker.observe_single(_obs("fever", "present"))
        expected = 1.0 - tracker.state_posterior(FeatureKey("fever"))["present"]
        self.assertAlmostEqual(tracker.p_wrong(FeatureKey("fever")), expected, places=10)
        # legacy default MISREPORTED prior is untouched (still 0.03 for NONE cue)
        self.assertEqual(
            ChannelParameters().cue_priors[CertaintyCue.NONE][ReportMode.MISREPORTED], 0.03
        )

    # 9 -------------------------------------------------------------------- #
    def test_09_protocol_fixed_prior_source_fixed(self):
        prior = protocol_fixed_prior()
        self.assertAlmostEqual(prior[ReportMode.CERTAIN], 0.75, places=10)
        self.assertAlmostEqual(prior[ReportMode.UNCERTAIN], 0.10, places=10)
        self.assertAlmostEqual(prior[ReportMode.UNKNOWN], 0.075, places=10)
        self.assertAlmostEqual(prior[ReportMode.MISREPORTED], 0.075, places=10)
        self.assertAlmostEqual(sum(prior.values()), 1.0, places=10)
        # pinned source: cross-noise average of PatientProfile.from_noise_rate
        avg = {
            mode: sum(PatientProfile.from_noise_rate(n).mode_prior[mode] for n in (0.2, 0.3)) / 2.0
            for mode in ReportMode
        }
        for mode in ReportMode:
            self.assertAlmostEqual(prior[mode], avg[mode], places=10)
        # protocol_fixed_channel_parameters applies it to every cue (not cue-conditioned)
        params = protocol_fixed_channel_parameters()
        for cue in CertaintyCue:
            self.assertEqual(params.cue_priors[cue], prior)
        # legacy default is NOT overwritten
        self.assertNotEqual(
            ChannelParameters().cue_priors[CertaintyCue.NONE][ReportMode.MISREPORTED],
            prior[ReportMode.MISREPORTED],
        )

    # 10 ------------------------------------------------------------------- #
    def test_10_inference_reads_no_privileged_info(self):
        for name in (
            "p_wrong", "is_nonresponse", "p_mode_misreported",
            "mode_posterior", "state_posterior",
        ):
            params = list(
                inspect.signature(getattr(JointReliabilityBeliefTracker, name)).parameters
            )
            self.assertEqual(params, ["self", "key"], name)

    # 11 ------------------------------------------------------------------- #
    def test_11_unknown_fix_leaves_trajectory_identical(self):
        env = _Env()
        cfg = ReliabilityAwarePolicyConfig(maximum_verifications=1, max_total_turns=8)

        def _run():
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
            return tuple(turn.action.kind.value for turn in result.turns)

        baseline = _run()
        # p_wrong's value -- including None for an UNKNOWN non-response -- must
        # never alter the action trajectory, even though Phase 22A now *logs*
        # max_p_wrong each turn. Forcing p_wrong to None everywhere must leave
        # the exact action sequence unchanged.
        with mock.patch.object(
            JointReliabilityBeliefTracker, "p_wrong", return_value=None
        ):
            patched = _run()
        self.assertEqual(patched, baseline)
        self.assertGreater(len(baseline), 0)


if __name__ == "__main__":
    unittest.main()
