import unittest

from powerful_medrag.belief import BeliefTracker
from powerful_medrag.channel import AnswerChannel, ReportMode
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.schema import (
    UNKNOWN,
    CertaintyCue,
    ClinicalCase,
    FeatureKey,
    Observation,
    make_binary_spec,
)


def make_separable_model() -> DiseaseStateModel:
    fever = FeatureKey("fever")
    cases = [
        *[ClinicalCase("flu", {fever: "present"}) for _ in range(98)],
        *[ClinicalCase("flu", {fever: "absent"}) for _ in range(2)],
        *[ClinicalCase("allergy", {fever: "absent"}) for _ in range(98)],
        *[ClinicalCase("allergy", {fever: "present"}) for _ in range(2)],
    ]
    return DiseaseStateModel.fit(cases, [make_binary_spec("fever")])


class BeliefTrackerTests(unittest.TestCase):
    def test_unknown_does_not_change_disease_belief(self):
        model = make_separable_model()
        tracker = BeliefTracker(model)
        prior = dict(tracker.belief)
        tracker.update(Observation(FeatureKey("fever"), UNKNOWN))
        for disease in model.diseases:
            self.assertAlmostEqual(tracker.belief[disease], prior[disease])

    def test_positive_answer_updates_posterior(self):
        model = make_separable_model()
        tracker = BeliefTracker(model)
        tracker.update(
            Observation(
                FeatureKey("fever"), "present", certainty=CertaintyCue.CERTAIN
            )
        )
        self.assertGreater(tracker.belief["flu"], 0.9)

    def test_contextual_difference_is_not_a_conflict(self):
        resting = FeatureKey.from_parts("pain", {"activity": "resting"})
        walking = FeatureKey.from_parts("pain", {"activity": "walking"})
        specs = [
            make_binary_spec("pain", context={"activity": "resting"}),
            make_binary_spec("pain", context={"activity": "walking"}),
        ]
        cases = [
            ClinicalCase("A", {resting: "absent", walking: "present"}),
            ClinicalCase("B", {resting: "present", walking: "absent"}),
        ]
        tracker = BeliefTracker(DiseaseStateModel.fit(cases, specs))
        tracker.update(Observation(resting, "absent"))
        result = tracker.update(Observation(walking, "present"))
        self.assertEqual(result.conflicting_turns, ())

    def test_same_context_opposite_answers_are_flagged(self):
        model = make_separable_model()
        tracker = BeliefTracker(model)
        tracker.update(Observation(FeatureKey("fever"), "present"))
        result = tracker.update(Observation(FeatureKey("fever"), "absent"))
        self.assertEqual(result.conflicting_turns, (0,))

    def test_repeated_reports_update_one_shared_latent_state(self):
        model = make_separable_model()
        tracker = BeliefTracker(model)
        key = FeatureKey("fever")
        before = tracker.state_beliefs["flu"][key]["present"]
        tracker.update(Observation(key, "present", certainty=CertaintyCue.CERTAIN))
        after = tracker.state_beliefs["flu"][key]["present"]
        self.assertGreater(after, before)
        tracker.update(Observation(key, UNKNOWN))
        self.assertAlmostEqual(
            tracker.state_beliefs["flu"][key]["present"],
            after,
        )

    def test_inconsistent_answer_raises_misreport_suspicion(self):
        model = make_separable_model()
        tracker = BeliefTracker(model, initial_belief={"flu": 0.99, "allergy": 0.01})
        result = tracker.update(
            Observation(
                FeatureKey("fever"), "absent", certainty=CertaintyCue.CERTAIN
            )
        )
        self.assertGreater(
            result.report_mode_posterior[ReportMode.MISREPORTED],
            AnswerChannel().parameters.cue_priors[CertaintyCue.CERTAIN][
                ReportMode.MISREPORTED
            ],
        )


if __name__ == "__main__":
    unittest.main()
