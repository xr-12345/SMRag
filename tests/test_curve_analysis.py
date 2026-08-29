import unittest

from powerful_medrag.benchmark import CurvePoint
from powerful_medrag.curve_analysis import compare_at_equal_budget


def point(strategy: str, threshold: float, questions: float, accuracy: float):
    return CurvePoint(
        strategy=strategy,
        noise_rate=0.2,
        posterior_threshold=threshold,
        cases=10,
        accuracy=accuracy,
        accuracy_ci_low=accuracy,
        accuracy_ci_high=accuracy,
        average_questions=questions,
        questions_standard_error=0.0,
        brier_score=0.0,
    )


class BudgetComparisonTests(unittest.TestCase):
    def test_interpolates_reference_accuracy_at_candidate_budget(self):
        matches = compare_at_equal_budget(
            [point("candidate", 0.8, 3.0, 0.65)],
            [
                point("reference", 0.7, 2.0, 0.60),
                point("reference", 0.9, 4.0, 0.80),
            ],
            candidate_strategy="candidate",
            reference_strategy="reference",
        )

        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0].reference_accuracy_at_same_questions, 0.70)
        self.assertAlmostEqual(matches[0].reference_minus_candidate_accuracy, 0.05)
        self.assertFalse(matches[0].reference_discretely_dominates)

    def test_marks_discrete_pareto_dominance_and_omits_extrapolation(self):
        matches = compare_at_equal_budget(
            [
                point("candidate", 0.7, 1.0, 0.50),
                point("candidate", 0.8, 3.0, 0.65),
            ],
            [
                point("reference", 0.7, 2.0, 0.70),
                point("reference", 0.9, 4.0, 0.80),
            ],
            candidate_strategy="candidate",
            reference_strategy="reference",
        )

        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].reference_discretely_dominates)


if __name__ == "__main__":
    unittest.main()
