import tempfile
import unittest
from pathlib import Path

from powerful_medrag.belief import BeliefTracker
from powerful_medrag.ablation import (
    ADAPTIVE_LEARNED,
    ORACLE_GATE,
    run_ablation_curve_experiment,
)
from powerful_medrag.channel import ReportMode, answer_channel_without_misreport
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.gate_analysis import _brier_score, _expected_calibration_error
from powerful_medrag.gate_learning import (
    GateTrainingExample,
    fit_logistic_gate,
    load_learned_gate,
    save_learned_gate,
)
from powerful_medrag.gating import (
    LEARNED_GATE_FEATURES,
    HeuristicMisreportGate,
    OracleMisreportGate,
)
from powerful_medrag.schema import CertaintyCue, FeatureKey, Observation


def example(surprisal: float, label: int) -> GateTrainingExample:
    features = {feature: 0.0 for feature in LEARNED_GATE_FEATURES}
    features["surprisal"] = surprisal
    features["top_probability"] = 0.6
    features["normalized_belief_entropy"] = 0.5
    return GateTrainingExample(features, label)


class LearnedGateTests(unittest.TestCase):
    def setUp(self):
        cases, specs = generate_toy_cases(cases_per_disease=20, seed=19)
        self.model = DiseaseStateModel.fit(cases, specs)

    def test_fit_calibrate_and_serialize_logistic_gate(self):
        training = [
            *(example(0.2 + index * 0.05, 0) for index in range(20)),
            *(example(3.0 + index * 0.05, 1) for index in range(20)),
        ]
        calibration = [example(0.5, 0), example(0.8, 0), example(3.2, 1), example(3.8, 1)]
        gate = fit_logistic_gate(
            training,
            calibration_examples=calibration,
            iterations=1000,
        )
        tracker = BeliefTracker(self.model, answer_channel_without_misreport())
        ordinary = Observation(
            FeatureKey("fever"), "present", certainty=CertaintyCue.CERTAIN
        )
        ordinary_probability = gate.probability(tracker, ordinary)
        self.assertGreaterEqual(ordinary_probability, gate.minimum_probability)
        self.assertLessEqual(ordinary_probability, gate.single_answer_cap)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            save_learned_gate(gate, path)
            loaded = load_learned_gate(path)
        self.assertEqual(gate.to_dict(), loaded.to_dict())

    def test_oracle_requires_and_uses_only_privileged_marker(self):
        gate = OracleMisreportGate()
        tracker = BeliefTracker(self.model)
        with self.assertRaises(ValueError):
            gate.probability(tracker, Observation(FeatureKey("fever"), "present"))
        marked = Observation(
            FeatureKey("fever"),
            "absent",
            oracle_report_mode=ReportMode.MISREPORTED.value,
        )
        self.assertEqual(gate.probability(tracker, marked), gate.positive_probability)

    def test_one_diagnostically_surprising_answer_remains_capped(self):
        gate = HeuristicMisreportGate()
        probability = gate.probability_from_signals(
            predictive_probability=1e-9,
            certainty=CertaintyCue.CERTAIN,
            is_unknown=False,
            direct_conflicts=0,
        )
        self.assertEqual(probability, gate.single_answer_cap)
        self.assertLess(probability, gate.conflict_cap)

    def test_calibration_metrics(self):
        labels = [0, 0, 1, 1]
        perfect = [0.0, 0.0, 1.0, 1.0]
        self.assertEqual(_brier_score(labels, perfect), 0.0)
        self.assertEqual(_expected_calibration_error(labels, perfect), 0.0)

    def test_learned_and_oracle_ablation_variants_run(self):
        training = [
            *(example(0.2 + index * 0.05, 0) for index in range(10)),
            *(example(3.0 + index * 0.05, 1) for index in range(10)),
        ]
        gate = fit_logistic_gate(training, iterations=200)
        cases, _ = generate_toy_cases(cases_per_disease=1, seed=20)
        points = run_ablation_curve_experiment(
            self.model,
            cases,
            variants=(ADAPTIVE_LEARNED, ORACLE_GATE),
            noise_rates=(0.2,),
            posterior_thresholds=(0.85,),
            max_questions=2,
            learned_gate=gate,
        )
        self.assertEqual(
            {point.strategy for point in points},
            {ADAPTIVE_LEARNED, ORACLE_GATE},
        )


if __name__ == "__main__":
    unittest.main()
