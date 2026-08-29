import unittest

from powerful_medrag.belief import BeliefTracker
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.questioning import (
    MedRAGReciprocalDegreeSelector,
    NumpyQuestionSelector,
    QuestionSelector,
    RandomQuestionSelector,
)
from powerful_medrag.schema import ClinicalCase, FeatureKey, Observation, VariableSpec, make_binary_spec


class QuestionSelectorTests(unittest.TestCase):
    def test_selector_prefers_discriminative_feature(self):
        useful = FeatureKey("useful")
        useless = FeatureKey("useless")
        specs = [make_binary_spec("useful"), make_binary_spec("useless")]
        cases = [
            *[
                ClinicalCase("A", {useful: "present", useless: "present"})
                for _ in range(30)
            ],
            *[
                ClinicalCase("B", {useful: "absent", useless: "present"})
                for _ in range(30)
            ],
        ]
        tracker = BeliefTracker(DiseaseStateModel.fit(cases, specs))
        ranking = QuestionSelector().rank(tracker)
        self.assertEqual(ranking[0].key, useful)
        self.assertGreater(
            ranking[0].expected_information_gain,
            ranking[1].expected_information_gain,
        )

    def test_predicted_answer_distribution_is_normalized(self):
        feature = FeatureKey("feature")
        model = DiseaseStateModel.fit(
            [
                ClinicalCase("A", {feature: "present"}),
                ClinicalCase("B", {feature: "absent"}),
            ],
            [make_binary_spec("feature")],
        )
        score = QuestionSelector().score(BeliefTracker(model), feature)
        self.assertAlmostEqual(sum(score.predicted_answers.values()), 1.0)

    def test_prerequisite_question_is_gated(self):
        root = FeatureKey("rash")
        detail = FeatureKey("rash_color")
        specs = [
            make_binary_spec("rash"),
            VariableSpec(
                key=detail,
                values=("red", "pink"),
                prerequisite=root,
            ),
        ]
        model = DiseaseStateModel.fit(
            [
                ClinicalCase("A", {root: "present", detail: "red"}),
                ClinicalCase("B", {root: "absent", detail: "pink"}),
            ],
            specs,
        )
        tracker = BeliefTracker(model)
        self.assertNotIn(detail, [score.key for score in QuestionSelector().rank(tracker)])
        tracker.update(Observation(root, "present"))
        self.assertIn(detail, [score.key for score in QuestionSelector().rank(tracker)])

    def test_numpy_selector_matches_reference(self):
        feature = FeatureKey("feature")
        model = DiseaseStateModel.fit(
            [
                ClinicalCase("A", {feature: "present"}),
                ClinicalCase("B", {feature: "absent"}),
            ],
            [make_binary_spec("feature")],
        )
        tracker = BeliefTracker(model)
        reference = QuestionSelector().score(tracker, feature)
        optimized = NumpyQuestionSelector().score(tracker, feature)
        self.assertAlmostEqual(
            optimized.expected_information_gain,
            reference.expected_information_gain,
            places=12,
        )

    def test_medrag_reciprocal_degree_prefers_rare_manifestation(self):
        rare = FeatureKey("rare")
        common = FeatureKey("common")
        model = DiseaseStateModel.fit(
            [
                ClinicalCase("A", {rare: "present", common: "present"}),
                ClinicalCase("B", {rare: "absent", common: "present"}),
            ],
            [make_binary_spec("rare"), make_binary_spec("common")],
        )
        selector = MedRAGReciprocalDegreeSelector(
            {
                "A": {rare, common},
                "B": {common},
            },
            candidate_disease_count=2,
        )
        ranking = selector.rank(BeliefTracker(model))
        self.assertEqual(ranking[0].key, rare)
        self.assertEqual(ranking[0].utility, 1.0)

    def test_random_selector_is_seeded_and_non_mutating(self):
        feature_a = FeatureKey("a")
        feature_b = FeatureKey("b")
        model = DiseaseStateModel.fit(
            [
                ClinicalCase("A", {feature_a: "present", feature_b: "absent"}),
                ClinicalCase("B", {feature_a: "absent", feature_b: "present"}),
            ],
            [make_binary_spec("a"), make_binary_spec("b")],
        )
        tracker = BeliefTracker(model)
        first = RandomQuestionSelector(seed=7).rank(tracker)
        second = RandomQuestionSelector(seed=7).rank(tracker)
        self.assertEqual([row.key for row in first], [row.key for row in second])
        self.assertEqual(tracker.history, [])


if __name__ == "__main__":
    unittest.main()
