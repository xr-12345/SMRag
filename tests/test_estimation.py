import json
import tempfile
import unittest
from pathlib import Path

from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.schema import ClinicalCase, FeatureKey, make_binary_spec


class DiseaseStateModelTests(unittest.TestCase):
    def setUp(self):
        self.fever = FeatureKey("fever")
        self.specs = [make_binary_spec("fever")]

    def test_missing_is_not_counted_as_absent(self):
        cases = [
            ClinicalCase("A", {self.fever: "present"}),
            ClinicalCase("A", {}),
            ClinicalCase("B", {self.fever: "absent"}),
        ]
        model = DiseaseStateModel.fit(cases, self.specs, alpha=1.0)
        estimate = model.estimates["A"][self.fever]
        self.assertEqual(estimate.observed_count, 1)
        self.assertEqual(estimate.counts["present"], 1)
        self.assertEqual(estimate.counts["absent"], 0)
        self.assertAlmostEqual(estimate.probabilities["present"], 2 / 3)

    def test_beta_bernoulli_laplace_formula(self):
        cases = [
            *[ClinicalCase("A", {self.fever: "present"}) for _ in range(3)],
            ClinicalCase("A", {self.fever: "absent"}),
            ClinicalCase("B", {self.fever: "absent"}),
        ]
        model = DiseaseStateModel.fit(cases, self.specs, alpha=1.0)
        self.assertAlmostEqual(
            model.state_distribution("A", self.fever)["present"], (3 + 1) / (4 + 2)
        )

    def test_round_trip_serialization(self):
        cases = [
            ClinicalCase("A", {self.fever: "present"}),
            ClinicalCase("B", {self.fever: "absent"}),
        ]
        model = DiseaseStateModel.fit(cases, self.specs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            restored = DiseaseStateModel.load(path)
        self.assertEqual(restored.diseases, model.diseases)
        self.assertEqual(
            restored.state_distribution("A", self.fever),
            model.state_distribution("A", self.fever),
        )


if __name__ == "__main__":
    unittest.main()

