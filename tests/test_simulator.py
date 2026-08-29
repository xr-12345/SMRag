import unittest

from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.schema import FeatureKey
from powerful_medrag.simulator import StopRule, StructuredPatientSimulator, run_dialogue


class SimulatorTests(unittest.TestCase):
    def test_dialogue_respects_question_budget_and_does_not_repeat(self):
        cases, specs = generate_toy_cases(cases_per_disease=30)
        model = DiseaseStateModel.fit(cases, specs)
        patient = StructuredPatientSimulator.sample_case(model, "influenza", seed=2)
        result = run_dialogue(
            patient,
            stop_rule=StopRule(
                max_questions=3,
                posterior_threshold=0.999,
                entropy_threshold=0.0,
            ),
        )
        self.assertLessEqual(len(result.turns), 3)
        keys = [turn.question.key for turn in result.turns]
        self.assertEqual(len(keys), len(set(keys)))

    def test_feature_indexed_noise_is_order_independent(self):
        cases, specs = generate_toy_cases(cases_per_disease=30)
        model = DiseaseStateModel.fit(cases, specs)
        latent_states = {
            key: "present" if "present" in spec.values else spec.values[0]
            for key, spec in model.specs.items()
        }
        first = StructuredPatientSimulator(
            diagnosis="influenza",
            latent_states=latent_states,
            model=model,
            seed=99,
        )
        second = StructuredPatientSimulator(
            diagnosis="influenza",
            latent_states=latent_states,
            model=model,
            seed=99,
        )
        fever = FeatureKey("fever")
        cough = FeatureKey("dry_cough")
        first_fever = first.answer(fever)
        first_cough = first.answer(cough)
        second_cough = second.answer(cough)
        second_fever = second.answer(fever)
        self.assertEqual(first_fever, second_fever)
        self.assertEqual(first_cough, second_cough)


if __name__ == "__main__":
    unittest.main()
