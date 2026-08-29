import unittest

from powerful_medrag.benchmark import CaseOutcome, summarize_paired_outcomes


class BenchmarkStatisticsTests(unittest.TestCase):
    def test_multiseed_summary_preserves_case_pairing(self):
        outcomes = []
        for seed in (1, 2):
            for case_id, eig_correct, baseline_correct, eig_q, baseline_q in (
                ("a", 1, 0, 4, 6),
                ("b", 1, 1, 5, 6),
            ):
                for strategy, correct, questions in (
                    ("eig", eig_correct, eig_q),
                    ("medrag_rdc", baseline_correct, baseline_q),
                ):
                    outcomes.append(
                        CaseOutcome(
                            seed=seed,
                            case_id=case_id,
                            true_diagnosis="A",
                            strategy=strategy,
                            noise_rate=0.1,
                            posterior_threshold=0.85,
                            predicted_diagnosis="A" if correct else "B",
                            correct=correct,
                            questions=questions,
                            brier_score=0.0,
                        )
                    )

        comparison = summarize_paired_outcomes(
            outcomes, bootstrap_samples=100, bootstrap_seed=7
        )[0]
        self.assertEqual(comparison.cases_per_seed, 2)
        self.assertEqual(comparison.seeds, 2)
        self.assertAlmostEqual(comparison.eig_accuracy, 1.0)
        self.assertAlmostEqual(comparison.baseline_accuracy, 0.5)
        self.assertAlmostEqual(comparison.accuracy_difference, 0.5)
        self.assertAlmostEqual(comparison.questions_saved, 1.5)
        self.assertEqual(comparison.eig_only_correct, 2)
        self.assertEqual(comparison.baseline_only_correct, 0)
        self.assertAlmostEqual(comparison.mcnemar_max_p, 1.0)


if __name__ == "__main__":
    unittest.main()
