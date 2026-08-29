import unittest

from powerful_medrag.ablation import (
    ADAPTIVE_HEURISTIC,
    ADAPTIVE_HISTORY,
    ADAPTIVE_SPARSE,
    ABLATION_VARIANTS,
    build_ablation_runtime,
    run_ablation_curve_experiment,
    unknown_or_uncertain_as_negative,
)
from powerful_medrag.belief import BeliefTracker
from powerful_medrag.channel import (
    ReportMode,
    answer_channel_without_misreport,
    reliable_answer_channel,
)
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.gating import (
    HeuristicMisreportGate,
    HistoryReliabilityMisreportGate,
    SparseSurprisalMisreportGate,
)
from powerful_medrag.gate_analysis import _average_precision, _roc_auc
from powerful_medrag.schema import (
    UNKNOWN,
    CertaintyCue,
    FeatureKey,
    Observation,
    make_binary_spec,
)


class ReportLayerAblationTests(unittest.TestCase):
    def test_gate_discrimination_metrics_handle_ties(self):
        labels = [0, 1, 0, 1]
        perfect_scores = [0.1, 0.9, 0.2, 0.8]
        tied_scores = [0.5, 0.5, 0.5, 0.5]
        self.assertAlmostEqual(_roc_auc(labels, perfect_scores), 1.0)
        self.assertAlmostEqual(_average_precision(labels, perfect_scores), 1.0)
        self.assertAlmostEqual(_roc_auc(labels, tied_scores), 0.5)
        self.assertAlmostEqual(_average_precision(labels, tied_scores), 0.5)

    def test_gate_increases_with_surprise_and_respects_caps(self):
        gate = HeuristicMisreportGate()
        ordinary = gate.probability_from_signals(
            predictive_probability=0.5,
            certainty=CertaintyCue.CERTAIN,
            is_unknown=False,
        )
        surprising = gate.probability_from_signals(
            predictive_probability=0.001,
            certainty=CertaintyCue.CERTAIN,
            is_unknown=False,
        )
        conflicting = gate.probability_from_signals(
            predictive_probability=0.001,
            certainty=CertaintyCue.CERTAIN,
            is_unknown=False,
            direct_conflicts=1,
        )
        self.assertAlmostEqual(ordinary, gate.baseline_probability)
        self.assertEqual(surprising, gate.single_answer_cap)
        self.assertEqual(conflicting, gate.conflict_cap)

    def test_adaptive_runtime_injects_dynamic_prior(self):
        cases, specs = generate_toy_cases(cases_per_disease=5, seed=2)
        model = DiseaseStateModel.fit(cases, specs)
        _, channel, _, gate = build_ablation_runtime(ADAPTIVE_HEURISTIC)
        tracker = BeliefTracker(model, channel, misreport_gate=gate)
        result = tracker.update(
            Observation(
                FeatureKey("fever"),
                "present",
                certainty=CertaintyCue.CERTAIN,
            )
        )
        self.assertGreaterEqual(result.misreport_prior, 0.005)
        self.assertLessEqual(result.misreport_prior, 0.15)

    def test_sparse_gate_ignores_nonconfident_and_ordinary_answers(self):
        cases, specs = generate_toy_cases(cases_per_disease=5, seed=3)
        model = DiseaseStateModel.fit(cases, specs)
        _, channel, _, gate = build_ablation_runtime(ADAPTIVE_SPARSE)
        self.assertIsInstance(gate, SparseSurprisalMisreportGate)
        tracker = BeliefTracker(model, channel)
        uncertain = Observation(
            FeatureKey("fever"),
            "present",
            certainty=CertaintyCue.UNCERTAIN,
        )
        self.assertEqual(gate.probability(tracker, uncertain), 0.0)

    def test_history_gate_uses_only_previous_unreliable_cues(self):
        cases, specs = generate_toy_cases(cases_per_disease=5, seed=4)
        model = DiseaseStateModel.fit(cases, specs)
        _, channel, _, gate = build_ablation_runtime(ADAPTIVE_HISTORY)
        self.assertIsInstance(gate, HistoryReliabilityMisreportGate)
        tracker = BeliefTracker(model, channel)
        current = Observation(
            FeatureKey("fever"),
            "present",
            certainty=CertaintyCue.CERTAIN,
        )
        self.assertEqual(gate.probability(tracker, current), 0.0)
        tracker.update(Observation(FeatureKey("dry_cough"), UNKNOWN))
        self.assertGreater(gate.probability(tracker, current), 0.0)

    def test_reliable_channel_ignores_unknown_but_trusts_known_answer(self):
        channel = reliable_answer_channel(epsilon=1e-6)
        states = ("absent", "present")
        self.assertAlmostEqual(
            channel.marginal_probability(UNKNOWN, "absent", states),
            channel.marginal_probability(UNKNOWN, "present", states),
        )
        self.assertGreater(
            channel.marginal_probability("present", "present", states),
            1000
            * channel.marginal_probability("present", "absent", states),
        )

    def test_no_misreport_channel_has_zero_inference_prior(self):
        channel = answer_channel_without_misreport()
        for weights in channel.parameters.cue_priors.values():
            self.assertEqual(weights[ReportMode.MISREPORTED], 0.0)
            self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_unknown_and_uncertain_are_coerced_to_negative(self):
        spec = make_binary_spec("fever")
        unknown = unknown_or_uncertain_as_negative(
            Observation(FeatureKey("fever"), UNKNOWN), spec
        )
        uncertain = unknown_or_uncertain_as_negative(
            Observation(
                FeatureKey("fever"),
                "present",
                certainty=CertaintyCue.UNCERTAIN,
            ),
            spec,
        )
        self.assertEqual(unknown.value, "absent")
        self.assertEqual(uncertain.value, "absent")
        self.assertEqual(uncertain.certainty, CertaintyCue.CERTAIN)

    def test_small_ablation_run_returns_every_variant(self):
        cases, specs = generate_toy_cases(cases_per_disease=4, seed=9)
        model = DiseaseStateModel.fit(cases, specs)
        points = run_ablation_curve_experiment(
            model,
            cases[:3],
            noise_rates=(0.1,),
            posterior_thresholds=(0.85,),
            max_questions=3,
            workers=1,
        )
        self.assertEqual({point.strategy for point in points}, set(ABLATION_VARIANTS))
        self.assertTrue(all(point.cases == 3 for point in points))

    def test_ablation_can_run_one_frozen_candidate(self):
        cases, specs = generate_toy_cases(cases_per_disease=4, seed=10)
        model = DiseaseStateModel.fit(cases, specs)
        points = run_ablation_curve_experiment(
            model,
            cases[:1],
            variants=(ADAPTIVE_HISTORY,),
            noise_rates=(0.2,),
            posterior_thresholds=(0.85,),
            max_questions=2,
        )
        self.assertEqual(points[0].strategy, ADAPTIVE_HISTORY)


if __name__ == "__main__":
    unittest.main()
