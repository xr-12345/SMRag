import csv
import json
import tempfile
import unittest
from pathlib import Path

from powerful_medrag.ddxplus import (
    ddxplus_specs,
    fit_ddxplus_model,
    iter_ddxplus_cases,
)
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.schema import FeatureKey


class DDXPlusAdapterTests(unittest.TestCase):
    def test_binary_categorical_and_multichoice_conversion(self):
        metadata = {
            "E_B": {
                "name": "E_B",
                "data_type": "B",
                "question_en": "Binary?",
            },
            "E_C": {
                "name": "E_C",
                "data_type": "C",
                "question_en": "Category?",
                "default_value": "V_0",
                "possible-values": ["V_0", "V_1"],
            },
            "E_M": {
                "name": "E_M",
                "data_type": "M",
                "question_en": "Which?",
                "default_value": "V_0",
                "possible-values": ["V_0", "V_1", "V_2"],
                "value_meaning": {
                    "V_1": {"en": "one"},
                    "V_2": {"en": "two"},
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "release_evidences.json"
            patient_path = root / "patients.csv"
            evidence_path.write_text(json.dumps(metadata), encoding="utf-8")
            with patient_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["AGE", "SEX", "PATHOLOGY", "EVIDENCES"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "AGE": "30",
                        "SEX": "F",
                        "PATHOLOGY": "A",
                        "EVIDENCES": repr(["E_B", "E_C_@_V_1", "E_M_@_V_2"]),
                    }
                )
                writer.writerow(
                    {
                        "AGE": "40",
                        "SEX": "M",
                        "PATHOLOGY": "B",
                        "EVIDENCES": repr([]),
                    }
                )

            specs = ddxplus_specs(evidence_path)
            cases = list(iter_ddxplus_cases(patient_path, evidence_path))
            fast_model = fit_ddxplus_model(patient_path, evidence_path)

        self.assertEqual(len(specs), 4)
        self.assertEqual(cases[0].states[FeatureKey("E_B")], "present")
        self.assertEqual(cases[1].states[FeatureKey("E_B")], "absent")
        self.assertEqual(cases[0].states[FeatureKey("E_C")], "V_1")
        self.assertEqual(cases[1].states[FeatureKey("E_C")], "V_0")
        option_two = FeatureKey.from_parts("E_M", {"option": "V_2"})
        self.assertEqual(cases[0].states[option_two], "present")
        self.assertEqual(cases[1].states[option_two], "absent")

        model = DiseaseStateModel.fit(iter(cases), specs)
        self.assertEqual(set(model.diseases), {"A", "B"})
        for disease in model.diseases:
            for spec in specs:
                self.assertEqual(
                    fast_model.state_distribution(disease, spec.key),
                    model.state_distribution(disease, spec.key),
                )


if __name__ == "__main__":
    unittest.main()
