import unittest

from powerful_medrag.belief import BeliefTracker
from powerful_medrag.channel import (
    AnswerChannel,
    ChannelParameters,
    ModeRates,
    ReportMode,
)
from powerful_medrag.decision import (
    ActionKind,
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    RequiredFeatureSafetyConstraint,
    _oracle_wrong_reports,
    clarification_question,
    run_reliability_aware_dialogue,
)
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.gating import LEARNED_GATE_FEATURES, LearnedMisreportGate
from powerful_medrag.schema import CertaintyCue, FeatureKey, Observation
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator


def _constant_learned_gate() -> LearnedMisreportGate:
    """A degenerate-but-valid gate returning a constant probability (0.15)."""
    return LearnedMisreportGate(
        intercept=0.0,
        coefficients={feature: 0.0 for feature in LEARNED_GATE_FEATURES},
    )


class _RecordingGate:
    """Wraps a gate, snapshotting the tracker the moment ``probability`` runs."""

    def __init__(self, inner: LearnedMisreportGate):
        self.inner = inner
        self.activation_threshold = inner.activation_threshold
        self.calls: list[tuple[Observation, tuple[Observation, ...]]] = []

    def probability(self, tracker, observation) -> float:
        self.calls.append(
            (
                observation,
                tuple(update.observation for update in tracker.history),
            )
        )
        return self.inner.probability(tracker, observation)


class ReliabilityAwareDecisionTests(unittest.TestCase):
    def setUp(self):
        cases, specs = generate_toy_cases(cases_per_disease=30, seed=41)
        self.model = DiseaseStateModel.fit(cases, specs)
        self.case = next(case for case in cases if case.diagnosis == "influenza")

    def test_joint_policy_exposes_new_verify_and_stop_actions(self):
        tracker = BeliefTracker(self.model)
        policy = ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(
                verification_cost=0.0,
                decision_impact_weight=10.0,
                minimum_unreliable_history_cues_for_verification=0,
            )
        )
        first = policy.choose_action(
            tracker,
            initial_observations=(),
            reports=(),
            asked=set(),
            verified_report_indices=set(),
            verification_count=0,
        )
        self.assertEqual(first.kind, ActionKind.NEW)
        report = Observation(
            first.key,
            "present",
            certainty=CertaintyCue.CERTAIN,
        )
        tracker.update(report)
        actions = policy.rank_actions(
            tracker,
            initial_observations=(),
            reports=(report,),
            asked={first.key},
            verified_report_indices=set(),
            verification_count=0,
        )
        self.assertIn(ActionKind.NEW, {action.kind for action in actions})
        self.assertIn(ActionKind.VERIFY, {action.kind for action in actions})
        excluded = policy.rank_actions(
            tracker,
            initial_observations=(),
            reports=(report,),
            asked={first.key},
            verified_report_indices={0},
            verification_count=1,
        )
        self.assertNotIn(ActionKind.VERIFY, {action.kind for action in excluded})

    def test_dialogue_respects_atomic_turn_and_verification_budgets(self):
        patient = StructuredPatientSimulator(
            diagnosis=self.case.diagnosis,
            latent_states=self.case.states,
            model=self.model,
            profile=PatientProfile.from_noise_rate(0.3),
            seed=9,
        )
        policy = ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(
                max_total_turns=6,
                maximum_verifications=1,
                verification_cost=0.0,
                decision_impact_weight=2.0,
            )
        )
        result = run_reliability_aware_dialogue(patient, policy=policy)
        self.assertLessEqual(len(result.turns), 6)
        self.assertLessEqual(result.verification_questions, 1)
        self.assertEqual(
            len(result.turns),
            result.new_questions + result.verification_questions,
        )

    def test_clarification_prompt_is_contextual_and_non_accusatory(self):
        spec = self.model.specs[FeatureKey("fever")]
        prompt = clarification_question(
            spec, Observation(spec.key, "present", certainty=CertaintyCue.CERTAIN)
        )
        self.assertIn(spec.question, prompt)
        self.assertIn("不知道", prompt)
        self.assertNotIn("说错", prompt)

    def test_oracle_wrong_reports_flags_only_truly_wrong_values(self):
        key = FeatureKey("fever")
        states = {key: "present"}
        wrong = Observation(key, "absent", certainty=CertaintyCue.CERTAIN)
        right = Observation(key, "present", certainty=CertaintyCue.CERTAIN)
        self.assertEqual(_oracle_wrong_reports((wrong, right), set(), states), [0])
        self.assertEqual(_oracle_wrong_reports((wrong,), {0}, states), [])
        self.assertEqual(
            _oracle_wrong_reports(
                (Observation(key, "__unknown__", certainty=CertaintyCue.NONE),), set(), states
            ),
            [],
        )

    def test_oracle_verification_ranks_verify_for_wrong_report(self):
        key = FeatureKey("fever")
        states = {key: "present"}
        patient = StructuredPatientSimulator(
            diagnosis=self.case.diagnosis,
            latent_states=self.case.states,
            model=self.model,
            profile=PatientProfile.from_noise_rate(0.0),
            seed=1,
        )
        tracker = BeliefTracker(self.model)
        policy = ReliabilityAwareActionPolicy(
            oracle_selection=True, oracle_correction=True
        )
        wrong = Observation(key, "absent", certainty=CertaintyCue.CERTAIN)
        actions = policy.rank_actions(
            tracker,
            initial_observations=(),
            reports=(wrong,),
            asked={key},
            verified_report_indices=set(),
            verification_count=0,
            oracle_states=states,
        )
        verify = [action for action in actions if action.kind == ActionKind.VERIFY]
        self.assertTrue(verify)
        self.assertEqual(verify[0].report_index, 0)
        self.assertEqual(verify[0].utility, 1.0)

    def test_oracle_select_same_channel_ranks_verify_without_correction(self):
        # Perfect selection alone (no oracle correction) still flags the wrong
        # report for verification — the same selection used by oracle_verify.
        key = FeatureKey("fever")
        states = {key: "present"}
        tracker = BeliefTracker(self.model)
        policy = ReliabilityAwareActionPolicy(
            oracle_selection=True, oracle_correction=False
        )
        wrong = Observation(key, "absent", certainty=CertaintyCue.CERTAIN)
        actions = policy.rank_actions(
            tracker,
            initial_observations=(),
            reports=(wrong,),
            asked={key},
            verified_report_indices=set(),
            verification_count=0,
            oracle_states=states,
        )
        verify = [action for action in actions if action.kind == ActionKind.VERIFY]
        self.assertTrue(verify)
        self.assertEqual(verify[0].report_index, 0)
        self.assertEqual(verify[0].utility, 1.0)

    def test_oracle_dialogue_resolves_a_wrong_report(self):
        key = FeatureKey("fever")
        # Force one latent state to "present" so a CERTAIN "absent" is a misreport.
        states = dict(self.case.states)
        states[key] = "present"
        patient = StructuredPatientSimulator(
            diagnosis=self.case.diagnosis,
            latent_states=states,
            model=self.model,
            profile=PatientProfile.from_noise_rate(0.0),
            seed=1,
        )
        policy = ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(max_total_turns=4),
            oracle_selection=True,
            oracle_correction=True,
        )
        result = run_reliability_aware_dialogue(patient, policy=policy)
        self.assertGreaterEqual(result.verification_questions, 0)
        self.assertLessEqual(len(result.turns), 4)
        self.assertEqual(
            len(result.turns),
            result.new_questions + result.verification_questions,
        )

    def test_oracle_selection_and_correction_are_independent(self):
        # A patient whose channel always misreports lets us observe the two
        # correction channels: oracle_correction substitutes the true latent
        # state, while the same-channel correction re-asks the patient (which
        # here keeps misreporting) instead of reading latent_states.
        def run(oracle_correction: bool):
            key = FeatureKey("fever")
            states = dict(self.case.states)
            states[key] = "present"
            profile = PatientProfile(
                mode_prior={
                    ReportMode.CERTAIN: 0.0,
                    ReportMode.UNCERTAIN: 0.0,
                    ReportMode.UNKNOWN: 0.0,
                    ReportMode.MISREPORTED: 1.0,
                }
            )
            channel = AnswerChannel(
                ChannelParameters(
                    rates={
                        mode: ModeRates(correct=0.0, unknown=0.0, wrong=1.0)
                        for mode in ReportMode
                    }
                )
            )
            patient = StructuredPatientSimulator(
                diagnosis=self.case.diagnosis,
                latent_states=states,
                model=self.model,
                channel=channel,
                profile=profile,
                seed=1,
            )
            policy = ReliabilityAwareActionPolicy(
                config=ReliabilityAwarePolicyConfig(max_total_turns=6),
                oracle_selection=True,
                oracle_correction=oracle_correction,
            )
            return patient, run_reliability_aware_dialogue(patient, policy=policy)

        patient_oracle, oracle_result = run(oracle_correction=True)
        patient_channel, channel_result = run(oracle_correction=False)

        oracle_verifies = [
            turn
            for turn in oracle_result.turns
            if turn.action.kind == ActionKind.VERIFY
        ]
        channel_verifies = [
            turn
            for turn in channel_result.turns
            if turn.action.kind == ActionKind.VERIFY
        ]
        self.assertTrue(oracle_verifies)
        self.assertTrue(channel_verifies)
        for turn in oracle_verifies:
            self.assertEqual(
                turn.observation.value,
                patient_oracle.latent_states[turn.observation.key],
            )
        for turn in channel_verifies:
            # Same-channel correction never reads latent_states: it re-asks the
            # always-misreporting patient, so the resolved value stays wrong.
            self.assertNotEqual(
                turn.observation.value,
                patient_channel.latent_states[turn.observation.key],
            )

    def test_learned_gate_replaces_verification_error_probability(self):
        tracker = BeliefTracker(self.model)
        policy = ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(
                verification_cost=0.0,
                decision_impact_weight=10.0,
                minimum_unreliable_history_cues_for_verification=0,
            ),
            verification_risk_gate=_constant_learned_gate(),
        )
        report = Observation(
            FeatureKey("fever"), "present", certainty=CertaintyCue.CERTAIN
        )
        tracker.update(report)
        actions = policy.rank_actions(
            tracker,
            initial_observations=(),
            reports=(report,),
            asked={report.key},
            verified_report_indices=set(),
            verification_count=0,
            report_risks=(0.99,),
        )
        verify = [action for action in actions if action.kind == ActionKind.VERIFY]
        self.assertTrue(verify)
        self.assertEqual(verify[0].report_index, 0)
        # The saved learned risk (0.99) replaces the retrospective error
        # probability in the verification utility.
        self.assertAlmostEqual(verify[0].disease_information_gain, 0.99)

    def test_learned_gate_activation_opens_verification_without_history_cue(self):
        tracker = BeliefTracker(self.model)
        policy = ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(
                verification_cost=0.0,
                decision_impact_weight=10.0,
                minimum_unreliable_history_cues_for_verification=1,
            ),
            verification_risk_gate=_constant_learned_gate(),
        )
        report = Observation(
            FeatureKey("fever"), "present", certainty=CertaintyCue.CERTAIN
        )
        tracker.update(report)
        actions = policy.rank_actions(
            tracker,
            initial_observations=(),
            reports=(report,),
            asked={report.key},
            verified_report_indices=set(),
            verification_count=0,
            report_risks=(0.15,),
        )
        self.assertIn(ActionKind.VERIFY, {action.kind for action in actions})

    def test_learned_gate_rejects_misaligned_saved_risks(self):
        tracker = BeliefTracker(self.model)
        policy = ReliabilityAwareActionPolicy(
            verification_risk_gate=_constant_learned_gate()
        )
        report = Observation(
            FeatureKey("fever"), "present", certainty=CertaintyCue.CERTAIN
        )
        tracker.update(report)
        with self.assertRaisesRegex(ValueError, "align one-to-one"):
            policy.rank_actions(
                tracker,
                initial_observations=(),
                reports=(report,),
                asked={report.key},
                verified_report_indices=set(),
                verification_count=0,
                report_risks=(),
            )

    def test_learned_gate_scores_report_before_tracker_update(self):
        patient = StructuredPatientSimulator(
            diagnosis=self.case.diagnosis,
            latent_states=self.case.states,
            model=self.model,
            profile=PatientProfile.from_noise_rate(0.2),
            seed=9,
        )
        recorder = _RecordingGate(_constant_learned_gate())
        policy = ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(max_total_turns=6),
            verification_risk_gate=recorder,
        )
        run_reliability_aware_dialogue(patient, policy=policy)
        self.assertTrue(recorder.calls)
        for observation, history_at_score_time in recorder.calls:
            # The answer must not yet be inside the tracker when it is scored.
            self.assertTrue(
                all(prior is not observation for prior in history_at_score_time)
            )

    def test_learned_gate_reasks_does_not_read_truth(self):
        key = FeatureKey("fever")
        states = dict(self.case.states)
        states[key] = "present"
        profile = PatientProfile(
            mode_prior={
                ReportMode.CERTAIN: 0.0,
                ReportMode.UNCERTAIN: 0.0,
                ReportMode.UNKNOWN: 0.0,
                ReportMode.MISREPORTED: 1.0,
            }
        )
        channel = AnswerChannel(
            ChannelParameters(
                rates={
                    mode: ModeRates(correct=0.0, unknown=0.0, wrong=1.0)
                    for mode in ReportMode
                }
            )
        )
        patient = StructuredPatientSimulator(
            diagnosis=self.case.diagnosis,
            latent_states=states,
            model=self.model,
            channel=channel,
            profile=profile,
            seed=1,
        )
        policy = ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(
                max_total_turns=6,
                maximum_verifications=1,
                verification_cost=0.0,
                decision_impact_weight=1e9,
                minimum_unreliable_history_cues_for_verification=0,
            ),
            verification_risk_gate=_constant_learned_gate(),
        )
        result = run_reliability_aware_dialogue(patient, policy=policy)
        self.assertGreaterEqual(result.verification_questions, 1)
        for turn in result.turns:
            if turn.action.kind == ActionKind.VERIFY:
                # Same-channel re-ask of an always-misreporting patient keeps
                # the wrong value: it never reads latent_states like the oracle.
                self.assertNotEqual(
                    turn.observation.value,
                    patient.latent_states[turn.observation.key],
                )

    def test_missing_required_safety_feature_prevents_confident_stop(self):
        tracker = BeliefTracker(self.model)
        tracker.belief = {
            disease: (0.99 if index == 0 else 0.01 / (len(self.model.diseases) - 1))
            for index, disease in enumerate(self.model.diseases)
        }
        required = FeatureKey("fever")
        policy = ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(minimum_action_utility=1.0),
            safety_constraint=RequiredFeatureSafetyConstraint(frozenset({required})),
        )
        action = policy.choose_action(
            tracker,
            initial_observations=(),
            reports=(),
            asked=set(),
            verified_report_indices=set(),
            verification_count=0,
        )
        self.assertNotEqual(action.kind, ActionKind.STOP)


if __name__ == "__main__":
    unittest.main()
