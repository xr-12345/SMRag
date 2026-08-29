import unittest

from powerful_medrag.belief import BeliefTracker
from powerful_medrag.channel import answer_channel_without_misreport
from powerful_medrag.clarification import (
    RETRO_UTILITY_U010_B1,
    PAMIS_STYLE_S30,
    RetrospectiveClarificationProtocol,
    SurprisalClarificationProtocol,
    run_clarification_curve_experiment,
)
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.schema import UNKNOWN, CertaintyCue, FeatureKey, Observation


class ClarificationProtocolTests(unittest.TestCase):
    def test_protocol_only_flags_confident_known_answer(self):
        cases, specs = generate_toy_cases(cases_per_disease=5, seed=11)
        model = DiseaseStateModel.fit(cases, specs)
        tracker = BeliefTracker(model, answer_channel_without_misreport())
        protocol = SurprisalClarificationProtocol(surprisal_threshold=0.0)
        certain = Observation(
            FeatureKey("fever"), "present", certainty=CertaintyCue.CERTAIN
        )
        uncertain = Observation(
            FeatureKey("fever"), "present", certainty=CertaintyCue.UNCERTAIN
        )
        unknown = Observation(FeatureKey("fever"), UNKNOWN)
        self.assertTrue(protocol.should_clarify(tracker, certain))
        self.assertFalse(protocol.should_clarify(tracker, uncertain))
        self.assertFalse(protocol.should_clarify(tracker, unknown))

    def test_resolution_keeps_agreement_and_abstains_on_conflict(self):
        protocol = SurprisalClarificationProtocol()
        original = Observation(
            FeatureKey("fever"), "present", certainty=CertaintyCue.CERTAIN
        )
        agreement = Observation(
            FeatureKey("fever"), "present", certainty=CertaintyCue.CERTAIN
        )
        conflict = Observation(
            FeatureKey("fever"), "absent", certainty=CertaintyCue.CERTAIN
        )
        self.assertEqual(protocol.resolve(original, agreement).value, "present")
        self.assertEqual(protocol.resolve(original, conflict).value, UNKNOWN)

    def test_small_curve_counts_clarification_inside_budget(self):
        cases, specs = generate_toy_cases(cases_per_disease=4, seed=12)
        model = DiseaseStateModel.fit(cases, specs)
        outcomes = []
        points = run_clarification_curve_experiment(
            model,
            cases[:2],
            variants=(PAMIS_STYLE_S30,),
            noise_rates=(0.2,),
            posterior_thresholds=(0.85,),
            max_questions=3,
            raw_outcomes=outcomes,
        )
        self.assertEqual(len(points), 1)
        self.assertEqual(len(outcomes), 2)
        self.assertTrue(all(row.questions <= 3 for row in outcomes))
        self.assertTrue(
            all(row.clarifications <= row.questions for row in outcomes)
        )

    def test_retrospective_score_is_leave_one_out_and_non_mutating(self):
        cases, specs = generate_toy_cases(cases_per_disease=8, seed=21)
        model = DiseaseStateModel.fit(cases, specs)
        reports = (
            Observation(
                FeatureKey("fever"),
                "present",
                certainty=CertaintyCue.CERTAIN,
            ),
            Observation(
                FeatureKey("itchy_eyes"),
                "absent",
                certainty=CertaintyCue.CERTAIN,
            ),
        )
        protocol = RetrospectiveClarificationProtocol(score_threshold=0.0)

        scores = protocol.rank(model, (), reports)

        self.assertEqual(reports[0].value, "present")
        self.assertEqual({score.report_index for score in scores}, {0, 1})
        self.assertTrue(all(0.0 <= score.error_probability <= 1.0 for score in scores))
        self.assertTrue(all(0.0 <= score.diagnostic_influence <= 1.0 for score in scores))
        self.assertTrue(all(score.score >= 0.0 for score in scores))

    def test_retrospective_curve_respects_total_and_clarification_budget(self):
        cases, specs = generate_toy_cases(cases_per_disease=5, seed=22)
        model = DiseaseStateModel.fit(cases, specs)
        outcomes = []

        run_clarification_curve_experiment(
            model,
            cases[:5],
            variants=(RETRO_UTILITY_U010_B1,),
            noise_rates=(0.3,),
            posterior_thresholds=(0.85,),
            max_questions=4,
            raw_outcomes=outcomes,
        )

        self.assertEqual(len(outcomes), 5)
        self.assertTrue(all(row.questions <= 4 for row in outcomes))
        self.assertTrue(all(row.clarifications <= 1 for row in outcomes))


if __name__ == "__main__":
    unittest.main()
