import unittest

from powerful_medrag.decision import ReliabilityAwarePolicyConfig
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.gating import LEARNED_GATE_FEATURES, LearnedMisreportGate
from powerful_medrag.reliability_experiment import run_reliability_experiment


def _constant_learned_gate() -> LearnedMisreportGate:
    return LearnedMisreportGate(
        intercept=0.0,
        coefficients={feature: 0.0 for feature in LEARNED_GATE_FEATURES},
    )


class ReliabilityExperimentTests(unittest.TestCase):
    def setUp(self):
        cases, specs = generate_toy_cases(cases_per_disease=30, seed=41)
        self.model = DiseaseStateModel.fit(cases, specs)
        self.case = next(case for case in cases if case.diagnosis == "influenza")

    def test_non_joint_strategies_respect_policy_threshold(self):
        # Regression for the fairness bug: the non-joint StopRule used to
        # hardcode posterior_threshold=0.85. A near-zero threshold must stop
        # immediately; a near-one threshold must ask at least one question.
        low = run_reliability_experiment(
            self.model,
            [self.case],
            noise_rates=(0.0,),
            seeds=(2026,),
            max_total_turns=5,
            strategies=("full_two_layer",),
            policy_config=ReliabilityAwarePolicyConfig(posterior_threshold=1e-6),
        )
        high = run_reliability_experiment(
            self.model,
            [self.case],
            noise_rates=(0.0,),
            seeds=(2026,),
            max_total_turns=5,
            strategies=("full_two_layer",),
            policy_config=ReliabilityAwarePolicyConfig(posterior_threshold=0.999999),
        )
        self.assertEqual(low[0].new_questions, 0)
        self.assertGreater(high[0].new_questions, low[0].new_questions)

    def test_oracle_select_same_channel_is_accepted_and_runs(self):
        outcomes = run_reliability_experiment(
            self.model,
            [self.case],
            noise_rates=(0.0,),
            seeds=(2026,),
            max_total_turns=5,
            strategies=("oracle_select_same_channel",),
        )
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].strategy, "oracle_select_same_channel")

    def test_joint_learned_gate_requires_gate(self):
        with self.assertRaises(ValueError):
            run_reliability_experiment(
                self.model,
                [self.case],
                noise_rates=(0.0,),
                seeds=(2026,),
                max_total_turns=5,
                strategies=("joint_learned_gate",),
                learned_verification_gate=None,
            )

    def test_joint_learned_gate_runs_and_respects_budgets(self):
        outcomes = run_reliability_experiment(
            self.model,
            [self.case],
            noise_rates=(0.2,),
            seeds=(2026,),
            max_total_turns=6,
            strategies=("joint_learned_gate",),
            learned_verification_gate=_constant_learned_gate(),
            policy_config=ReliabilityAwarePolicyConfig(
                max_total_turns=6, maximum_verifications=1
            ),
        )
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].strategy, "joint_learned_gate")
        self.assertLessEqual(outcomes[0].verification_questions, 1)
        self.assertLessEqual(outcomes[0].interaction_turns, 6)

    def test_joint_new_verify_stop_unaffected_by_learned_gate(self):
        common = dict(
            noise_rates=(0.2,),
            seeds=(2026,),
            max_total_turns=6,
            strategies=("joint_new_verify_stop",),
            policy_config=ReliabilityAwarePolicyConfig(
                max_total_turns=6, maximum_verifications=1
            ),
        )
        without_gate = run_reliability_experiment(self.model, [self.case], **common)
        with_gate = run_reliability_experiment(
            self.model,
            [self.case],
            learned_verification_gate=_constant_learned_gate(),
            **common,
        )
        self.assertEqual(len(without_gate), len(with_gate))
        for left, right in zip(without_gate, with_gate):
            self.assertEqual(left.correct_top1, right.correct_top1)
            self.assertEqual(left.new_questions, right.new_questions)
            self.assertEqual(left.verification_questions, right.verification_questions)
            self.assertEqual(left.resolved_wrong_reports, right.resolved_wrong_reports)


if __name__ == "__main__":
    unittest.main()
