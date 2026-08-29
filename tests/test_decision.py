import unittest

from powerful_medrag.belief import BeliefTracker
from powerful_medrag.decision import (
    ActionKind,
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    RequiredFeatureSafetyConstraint,
    clarification_question,
    run_reliability_aware_dialogue,
)
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.schema import CertaintyCue, FeatureKey, Observation
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator


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
