"""Phase 4 -- learned verification-worthiness tests (the 12 spec tests)."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

import numpy as np

from powerful_medrag import verification_worthiness as vw

ARTIFACT = Path("artifacts/verimedrag-verification-worthiness-offline")
SRC = Path("src/powerful_medrag")


def _sample_row(**overrides):
    """A minimal VerifyOld row with all deployable + label columns filled."""
    row = {
        "case_id": "c1",
        "diagnosis": "Influenza",
        "noise_rate": "0.2",
        "state_kind": "early",
        "state_index": "0",
        "turn_index": "1",
        "n_reports": "2",
        "retrospective_error_prob": "0.4",
        "retrieval_impact": "0.5",
        "heuristic_verify_utility": "0.15",
        "error_prob_times_impact": "0.2",
        "current_risk": "0.6",
        "v_bayes": "0.05",
        "v_real": "0.02",
        "gross_brier_reduction": "0.05",
        "top1_before_correct": "1",
        "correct_to_wrong": "0.0",
    }
    row.update(overrides)
    return row


class TestFeatureSchema(unittest.TestCase):
    def test_01_schema_has_no_forbidden_fields(self):
        schema = vw.feature_schema()
        names = {f["feature"] for f in schema}
        self.assertEqual(names, set(vw.ALL_FEATURES))
        for f in schema:
            self.assertNotIn(f["feature"], vw.FORBIDDEN_FIELDS,
                             f"{f['feature']} is forbidden as a feature")
            self.assertTrue(f["deployable"] and f["label_free"])
        # forbidden fields are never features
        for name in vw.FORBIDDEN_FIELDS:
            self.assertNotIn(name, vw.ALL_FEATURES)


class TestSplit(unittest.TestCase):
    def test_02_case_grouped_split_no_overlap(self):
        train = Path("artifacts/verimedrag-action-value-audit/action_value_samples.csv")
        val = ARTIFACT / "validation_features.csv"
        if not val.exists():
            self.skipTest("validation_features.csv not yet generated")

        def ids(path):
            with open(path, newline="", encoding="utf-8") as f:
                return {r["case_id"] for r in csv.DictReader(f)}

        train_ids = ids(train)
        val_ids = ids(val)
        self.assertEqual(len(train_ids & val_ids), 0,
                         "train and validation case ids must be disjoint")


class TestLabels(unittest.TestCase):
    def test_03_gain_label_sign_and_cost_deducted_once(self):
        row = _sample_row(gross_brier_reduction="0.08", v_real="0.05")
        labels = vw.derive_labels(row)
        self.assertAlmostEqual(labels["gain"], 0.08)
        self.assertAlmostEqual(labels["value"], 0.05)
        # value == gain - C_verify exactly once (no double deduction)
        self.assertAlmostEqual(labels["value"], labels["gain"] - vw.C_VERIFY)

    def test_04_harm_label_binary_from_correct_to_wrong(self):
        self.assertEqual(vw.derive_labels(_sample_row(correct_to_wrong="0.0"))["harm"], 0.0)
        self.assertEqual(vw.derive_labels(_sample_row(correct_to_wrong="0.125"))["harm"], 1.0)
        self.assertEqual(vw.derive_labels(_sample_row(correct_to_wrong="1.0"))["harm"], 1.0)

    def test_05_benefit_label_sign(self):
        self.assertEqual(vw.derive_labels(_sample_row(v_real="0.01"))["benefit"], 1.0)
        self.assertEqual(vw.derive_labels(_sample_row(v_real="-0.01"))["benefit"], 0.0)


class TestNoOracle(unittest.TestCase):
    def test_06_features_invariant_to_latent_true_disease_noise(self):
        base = _sample_row()
        x0 = vw.extract_feature_vector(base)
        # perturb forbidden fields only -> identical feature vector
        altered = dict(base)
        altered["diagnosis"] = "Pneumonia"
        altered["noise_rate"] = "0.9"
        altered["current_brier"] = "0.123"
        altered["current_true_prob"] = "0.999"
        altered["true_state"] = "present"
        altered["is_report_wrong"] = "1"
        altered["v_real"] = "5.0"
        altered["gross_brier_reduction"] = "5.0"
        altered["correct_to_wrong"] = "1.0"
        x1 = vw.extract_feature_vector(altered)
        np.testing.assert_allclose(x0, x1)


class TestHyperparamHygiene(unittest.TestCase):
    def test_07_validation_not_used_in_hyperparam_selection(self):
        src = (ARTIFACT / "train_model.py").read_text(encoding="utf-8")
        self.assertNotIn("validation_features.csv", src,
                         "train_model.py must not read the validation set")
        # it reads only the train split
        self.assertIn("train_features.csv", src)


class TestShuffle(unittest.TestCase):
    def test_08_shuffled_rag_preserves_marginals(self):
        rows = [_sample_row(case_id=f"c{i%4}", diagnosis=f"d{i%2}",
                            noise_rate=f"0.{2 if i%3 else 3}",
                            turn_index=str(i % 7),
                            retrieval_impact=str(0.1 * (i % 10)),
                            error_prob_times_impact=str(0.01 * (i % 11)))
                for i in range(200)]
        X_real = vw.build_feature_matrix(rows, rag_mode="real")
        X_shuf = vw.shuffle_rag_features(rows, seed=3031)
        base_n = len(vw.BASE_FEATURES)
        # base features unchanged
        np.testing.assert_allclose(X_real[:, :base_n], X_shuf[:, :base_n])
        # RAG column marginals preserved (same multiset, hence same mean/sd)
        for col in range(base_n, X_real.shape[1]):
            self.assertAlmostEqual(X_real[:, col].mean(), X_shuf[:, col].mean(), places=9)
            self.assertAlmostEqual(X_real[:, col].std(), X_shuf[:, col].std(), places=9)
        # link is broken somewhere (RAG columns are not identical)
        self.assertFalse(np.allclose(X_real[:, base_n:], X_shuf[:, base_n:]))

    def test_08b_build_feature_matrix_shuffled_matches_shuffle(self):
        # regression: build_feature_matrix("shuffled") must actually shuffle
        # (previously it returned the same matrix as "real", making the shuffled
        # ablation a no-op duplicate of the real-RAG model).
        rows = [_sample_row(case_id=f"c{i%4}", diagnosis=f"d{i%2}",
                            noise_rate=f"0.{2 if i%3 else 3}",
                            turn_index=str(i % 7),
                            retrieval_impact=str(0.1 * (i % 10)),
                            error_prob_times_impact=str(0.01 * (i % 11)))
                for i in range(200)]
        np.testing.assert_allclose(
            vw.build_feature_matrix(rows, rag_mode="shuffled"),
            vw.shuffle_rag_features(rows, seed=3031),
        )


class TestMatchedBudget(unittest.TestCase):
    def test_09_matched_budget_gives_equal_triggers(self):
        scores = np.array([0.9, 0.1, 0.8, 0.3, 0.7])
        for k in (0, 1, 3, 5, 99):
            mask = vw.top_k_mask(scores, k)
            self.assertEqual(int(mask.sum()), min(k, len(scores)))
        # descending order respected
        mask = vw.top_k_mask(scores, 2)
        self.assertTrue(mask[0] and mask[2])


class TestPolicyScope(unittest.TestCase):
    def test_10_default_policy_unchanged(self):
        # importing the module must not wire the learned model into the policy
        from powerful_medrag.decision import ReliabilityAwarePolicyConfig
        cfg = ReliabilityAwarePolicyConfig()
        self.assertAlmostEqual(cfg.verification_cost, 0.03)
        decision_src = (SRC / "decision.py").read_text(encoding="utf-8")
        self.assertNotIn("verification_worthiness", decision_src,
                         "decision.py must not import the learned model")

    def test_11_asknew_keeps_eig(self):
        # the module is VerifyOld-only: it must not import or define AskNew /
        # EIG scoring (AskNew keeps the existing EIG untouched).
        src = (SRC / "verification_worthiness.py").read_text(encoding="utf-8")
        self.assertNotIn("asknew_value", src)
        self.assertNotIn("asknew_answer_distribution", src)
        self.assertNotIn("expected_information_gain", src)
        self.assertNotIn("from .action_value import", src)


class TestSerialize(unittest.TestCase):
    def test_12_serialize_load_predict_same(self):
        from sklearn.linear_model import LogisticRegression, Ridge
        import joblib, tempfile, os
        X = np.random.default_rng(0).normal(size=(120, 7))
        g = X[:, 0] - X[:, 1]
        h = (X[:, 2] > 0).astype(float)
        gain = Ridge(alpha=1.0).fit(X, g)
        harm = LogisticRegression().fit(X, h)
        model = vw.LearnedWorthinessModel(
            gain_model=gain, harm_model=harm, tau_harm=0.4,
            gain_kind="ridge", harm_kind="logistic", n_features=7, rag_mode="real",
        )
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pkl")
            joblib.dump(model, p)
            loaded = joblib.load(p)
        self.assertEqual(loaded.tau_harm, model.tau_harm)
        self.assertEqual(loaded.n_features, model.n_features)
        np.testing.assert_allclose(loaded.predict(X)[0], model.predict(X)[0])
        np.testing.assert_allclose(loaded.predict(X)[1], model.predict(X)[1])


if __name__ == "__main__":
    unittest.main()
