"""Phase 5 -- online learned-worthiness integration tests (13 spec tests).

Covers the frozen-model load / feature-schema contract, the learned gain + harm
gates, the unified Brier-unit cross-type comparison, the AskNew EIG selection,
the no-true-state prediction path, the RAG / no-RAG feature split, the
heuristic-baseline regression, and the "default stays heuristic" red line.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from powerful_medrag.action_value import asknew_value, verify_value
from powerful_medrag.belief import BeliefTracker
from powerful_medrag.decision import (
    ActionKind,
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    RetrievalMode,
    run_reliability_aware_dialogue,
)
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.questioning import NumpyQuestionSelector
from powerful_medrag.retrieval import MedicalRetriever
from powerful_medrag.schema import UNKNOWN, CertaintyCue, FeatureKey, Observation
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator
from powerful_medrag.verification_worthiness import (
    BASE_FEATURES,
    C_VERIFY,
    LearnedWorthinessModel,
)
from powerful_medrag.worthiness_policy import (
    WorthinessStrategy,
    _validate_n_features,
    build_online_features,
    build_policy,
    load_frozen_worthiness_model,
)

SRC = Path("src/powerful_medrag")
ARTIFACT = Path("artifacts/verimedrag-verification-worthiness-offline")

# Corpus supports "fever"/"myalgia" only, so other toy features leave retrieval
# unchanged (impact 0).
INTEG_CORPUS = [
    {"id": "doc_fever", "title": "Fever", "content": "fever",
     "contents": "fever is elevated body temperature"},
    {"id": "doc_myalgia", "title": "Myalgia", "content": "myalgia",
     "contents": "myalgia is muscle pain"},
]


def _write_corpus(tmpdir: str, docs: list[dict]) -> str:
    path = Path(tmpdir) / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc) + "\n")
    return tmpdir


class _StubGain:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value)


class _StubHarm:
    def __init__(self, value: float):
        self.value = float(value)

    def predict_proba(self, X):
        p = np.zeros((len(X), 2))
        p[:, 1] = self.value
        return p


def _stub_model(gain: float, harm: float, *, tau_harm=0.4129, n_features=7):
    return LearnedWorthinessModel(
        gain_model=_StubGain(gain),
        harm_model=_StubHarm(harm),
        tau_harm=tau_harm,
        gain_kind="stub",
        harm_kind="stub",
        n_features=n_features,
        rag_mode="none" if n_features == 7 else "real",
    )


class ToyEnv:
    """A small toy model + tracker + reports for policy-level tests."""

    def __init__(self):
        cases, specs = generate_toy_cases(cases_per_disease=30, seed=41)
        self.model = DiseaseStateModel.fit(cases, specs)
        self.cases = cases

    def tracker_with(self, reports, *, initial=()):
        tracker = BeliefTracker(self.model)
        for obs in initial:
            tracker.update(obs)
        for obs in reports:
            tracker.update(obs)
        return tracker

    @staticmethod
    def obs(name, value="present", certainty=CertaintyCue.CERTAIN):
        return Observation(FeatureKey(name), value, certainty=certainty)


class TestFrozenModels(unittest.TestCase):
    def test_01_frozen_models_load_with_expected_metadata(self):
        real = load_frozen_worthiness_model("real")
        none = load_frozen_worthiness_model("none")
        self.assertIsInstance(real, LearnedWorthinessModel)
        self.assertEqual(real.gain_kind, "hgb")
        self.assertEqual(real.harm_kind, "hgb")
        self.assertEqual(real.n_features, 9)
        self.assertEqual(real.rag_mode, "real")
        self.assertAlmostEqual(real.tau_harm, 0.4129, places=3)
        self.assertEqual(none.n_features, 7)
        self.assertEqual(none.rag_mode, "none")
        g, h = real.predict(np.zeros((5, 9)))
        self.assertEqual(g.shape, (5,))
        self.assertEqual(h.shape, (5,))
        g2, h2 = none.predict(np.zeros((5, 7)))
        self.assertEqual(g2.shape, (5,))
        self.assertEqual(h2.shape, (5,))


class TestFeatureSchema(unittest.TestCase):
    def test_02_feature_schema_order_matches_json(self):
        schema = json.loads((ARTIFACT / "feature_schema.json").read_text(encoding="utf-8"))
        names = [f["feature"] for f in schema]
        self.assertEqual(names, list(BASE_FEATURES) + ["retrieval_impact",
                                                       "error_prob_times_impact"])
        # build_online_features emits the same order
        class _S:
            error_probability = 0.3
            diagnostic_influence = 0.4
            retrieval_impact = 0.5
        full = build_online_features(_S(), current_risk=0.6, n_reports=2,
                                     turn_index=3, use_rag=True)
        self.assertEqual(full.shape, (9,))
        base = build_online_features(_S(), current_risk=0.6, n_reports=2,
                                     turn_index=3, use_rag=False)
        self.assertEqual(base.shape, (7,))
        self.assertEqual(list(full[:7]), list(base))


class TestGates(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()
        self.cfg = ReliabilityAwarePolicyConfig(retrieval_mode=RetrievalMode.NO_RAG)

    def _learned_rank(self, model, reports):
        policy = build_policy(
            WorthinessStrategy.LEARNED_WORTHINESS_NO_RAG,
            config=self.cfg,
            worthiness_model=model,
        )
        tracker = self.env.tracker_with(reports)
        actions = policy.rank_actions(
            tracker,
            initial_observations=(),
            reports=tuple(reports),
            asked={r.key for r in reports},
            verified_report_indices=set(),
            verification_count=0,
        )
        return actions

    def test_03_gain_gate_blocks_nonpositive_net_value(self):
        # net_value = 0.02 - 0.03 = -0.01 <= 0 -> blocked regardless of harm
        model = _stub_model(gain=0.02, harm=0.1)
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        actions = self._learned_rank(model, reports)
        verify = [a for a in actions if a.kind is ActionKind.VERIFY]
        self.assertFalse(verify)

    def test_04_harm_gate_blocks_high_harm(self):
        # net_value = 0.50 - 0.03 = 0.47 > 0 but harm 0.9 > tau_harm 0.4129
        model = _stub_model(gain=0.50, harm=0.9)
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        actions = self._learned_rank(model, reports)
        verify = [a for a in actions if a.kind is ActionKind.VERIFY]
        self.assertFalse(verify)

    def test_05_both_gates_required(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        both_ok = _stub_model(gain=0.50, harm=0.3)
        gain_fail = _stub_model(gain=0.02, harm=0.3)
        harm_fail = _stub_model(gain=0.50, harm=0.9)
        ok = [a for a in self._learned_rank(both_ok, reports)
              if a.kind is ActionKind.VERIFY]
        self.assertTrue(ok)
        self.assertAlmostEqual(ok[0].utility, 0.50 - C_VERIFY)
        self.assertFalse([a for a in self._learned_rank(gain_fail, reports)
                          if a.kind is ActionKind.VERIFY])
        self.assertFalse([a for a in self._learned_rank(harm_fail, reports)
                          if a.kind is ActionKind.VERIFY])


class TestUnifiedComparison(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()
        self.cfg = ReliabilityAwarePolicyConfig(retrieval_mode=RetrievalMode.NO_RAG)

    def test_06_asknew_selected_by_eig_not_vbayes(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        policy = build_policy(
            WorthinessStrategy.LEARNED_WORTHINESS_NO_RAG,
            config=self.cfg,
            worthiness_model=_stub_model(gain=0.5, harm=0.3),
        )
        tracker = self.env.tracker_with(reports)
        actions = policy.rank_actions(
            tracker, initial_observations=(), reports=tuple(reports),
            asked={r.key for r in reports}, verified_report_indices=set(),
            verification_count=0,
        )
        news = [a for a in actions if a.kind is ActionKind.NEW]
        self.assertEqual(len(news), 1, "only the single EIG-best AskNew is kept")
        eig_best = NumpyQuestionSelector().rank(
            tracker, excluded={r.key for r in reports}
        )[0]
        self.assertEqual(news[0].key, eig_best.key,
                         "AskNew key must be the EIG-best question")
        # its value is V_Bayes_new (Brier), while the EIG is kept for reference
        self.assertAlmostEqual(
            news[0].utility,
            asknew_value(tracker, eig_best.key, C_new=0.03),
        )
        self.assertAlmostEqual(news[0].disease_information_gain,
                               eig_best.expected_information_gain)

    def test_07_cross_type_comparison_uses_brier_units(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        policy = build_policy(
            WorthinessStrategy.LEARNED_WORTHINESS_NO_RAG,
            config=self.cfg,
            worthiness_model=_stub_model(gain=0.5, harm=0.3),
        )
        tracker = self.env.tracker_with(reports)
        actions = policy.rank_actions(
            tracker, initial_observations=(), reports=tuple(reports),
            asked={r.key for r in reports}, verified_report_indices=set(),
            verification_count=0,
        )
        for a in actions:
            if a.kind is ActionKind.NEW:
                # Brier unit, not EIG
                self.assertNotAlmostEqual(
                    a.utility, a.disease_information_gain, places=6,
                )
                self.assertAlmostEqual(
                    a.utility,
                    asknew_value(tracker, a.key, C_new=0.03),
                )
            elif a.kind is ActionKind.VERIFY:
                # learned net_value in Brier units (g_hat - C_verify)
                self.assertAlmostEqual(a.utility, 0.5 - C_VERIFY)


class TestNoTrueState(unittest.TestCase):
    def test_08_prediction_path_reads_no_true_state(self):
        env = ToyEnv()
        cfg = ReliabilityAwarePolicyConfig(retrieval_mode=RetrievalMode.NO_RAG)
        reports = [env.obs("fever"), env.obs("myalgia")]
        policy = build_policy(
            WorthinessStrategy.LEARNED_WORTHINESS_NO_RAG,
            config=cfg, worthiness_model=_stub_model(gain=0.5, harm=0.3),
        )
        tracker = env.tracker_with(reports)
        fake_oracle = {FeatureKey("fever"): "absent"}
        args = dict(
            initial_observations=(), reports=tuple(reports),
            asked={r.key for r in reports}, verified_report_indices=set(),
            verification_count=0,
        )
        a_none = policy.rank_actions(tracker, oracle_states=None, **args)
        a_fake = policy.rank_actions(tracker, oracle_states=fake_oracle, **args)
        self.assertEqual(
            [(a.kind.value, a.utility, a.report_index) for a in a_none],
            [(a.kind.value, a.utility, a.report_index) for a in a_fake],
            "oracle state must not change the learned prediction path",
        )
        # the module never touches latent state / true disease / noise
        src = (SRC / "worthiness_policy.py").read_text(encoding="utf-8")
        self.assertNotIn("latent_states", src)
        self.assertNotIn("true_diagnosis", src)
        self.assertNotIn("oracle_correction", src)


class TestRagSplit(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()
        self.tmp = tempfile.TemporaryDirectory()
        self.retriever = MedicalRetriever(
            _write_corpus(self.tmp.name, INTEG_CORPUS)
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_09_no_rag_uses_no_retrieval_features(self):
        model = load_frozen_worthiness_model("none")
        self.assertEqual(model.n_features, len(BASE_FEATURES))
        self.assertEqual(model.n_features, 7)
        policy = build_policy(
            WorthinessStrategy.LEARNED_WORTHINESS_NO_RAG,
            config=ReliabilityAwarePolicyConfig(retrieval_mode=RetrievalMode.NO_RAG),
            worthiness_model=model,
        )
        self.assertFalse(policy._use_rag)
        # the online feature vector for no_rag is base-only (7 columns), and a
        # non-zero retrieval impact is ignored (not fed to the model).
        class _S:
            error_probability = 0.3
            diagnostic_influence = 0.4
            retrieval_impact = 0.9
        x = build_online_features(_S(), current_risk=0.6, n_reports=2,
                                  turn_index=3, use_rag=False)
        self.assertEqual(x.shape, (7,))
        self.assertEqual(x[3], 0.6)  # current_risk lands in the base block

    def test_10_full_rag_uses_real_retrieval(self):
        model = load_frozen_worthiness_model("real")
        self.assertEqual(model.n_features, len(BASE_FEATURES) + 2)
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        policy = build_policy(
            WorthinessStrategy.LEARNED_WORTHINESS_FULL_RAG,
            config=ReliabilityAwarePolicyConfig(
                retrieval_mode=RetrievalMode.DYNAMIC_RAG,
                retrieval_impact_weight=1.0,
            ),
            worthiness_model=model,
            retriever=self.retriever,
        )
        self.assertTrue(policy._use_rag)
        tracker = self.env.tracker_with(reports)
        policy.rank_actions(
            tracker, initial_observations=(), reports=tuple(reports),
            asked={r.key for r in reports}, verified_report_indices=set(),
            verification_count=0,
        )
        self.assertTrue(policy.last_verification_log)
        impacts = [e["retrieval_impact"] for e in policy.last_verification_log]
        # fever/myalgia are corpus-backed, so some impact is non-zero
        self.assertTrue(any(i > 0.0 for i in impacts))


class TestBaseline(unittest.TestCase):
    def test_11_heuristic_baseline_regression_identical(self):
        env = ToyEnv()
        cfg = ReliabilityAwarePolicyConfig(
            retrieval_mode=RetrievalMode.NO_RAG, max_total_turns=6,
        )
        case = next(c for c in env.cases if c.diagnosis == "influenza")

        def run(policy):
            patient = StructuredPatientSimulator(
                diagnosis=case.diagnosis, latent_states=case.states,
                model=env.model, profile=PatientProfile.from_noise_rate(0.3),
                seed=9,
            )
            return run_reliability_aware_dialogue(
                patient, policy=policy, initial_observations=case.initial_observations
            )

        base = run(ReliabilityAwareActionPolicy(config=cfg))
        built = run(build_policy(WorthinessStrategy.HEURISTIC_VERIFY, config=cfg))
        self.assertEqual(
            [t.action.kind.value for t in base.turns],
            [t.action.kind.value for t in built.turns],
        )
        self.assertEqual(base.stop_reason, built.stop_reason)
        self.assertEqual(base.predicted_diagnosis, built.predicted_diagnosis)
        self.assertIs(
            type(build_policy(WorthinessStrategy.HEURISTIC_VERIFY, config=cfg)),
            ReliabilityAwareActionPolicy,
        )


class TestFailLoudly(unittest.TestCase):
    def test_12_missing_model_fails_loudly(self):
        from powerful_medrag import worthiness_policy as wp
        with mock.patch.object(wp, "FROZEN_MODEL_DIR", Path("/nonexistent-dir")):
            with self.assertRaises(FileNotFoundError):
                load_frozen_worthiness_model("real")
        # wrong feature count raises
        with self.assertRaises(ValueError):
            _validate_n_features(_stub_model(0.5, 0.3, n_features=7), use_rag=True)
        with self.assertRaises(ValueError):
            _validate_n_features(_stub_model(0.5, 0.3, n_features=9), use_rag=False)


class TestDefaultStaysHeuristic(unittest.TestCase):
    def test_13_default_strategy_stays_heuristic(self):
        policy = build_policy(WorthinessStrategy.HEURISTIC_VERIFY)
        self.assertIs(type(policy), ReliabilityAwareActionPolicy)
        # decision.py must not import the learned/online module (default intact)
        decision_src = (SRC / "decision.py").read_text(encoding="utf-8")
        self.assertNotIn("worthiness_policy", decision_src)
        self.assertNotIn("verification_worthiness", decision_src)
        # the default config is unchanged
        cfg = ReliabilityAwarePolicyConfig()
        self.assertAlmostEqual(cfg.verification_cost, 0.03)


if __name__ == "__main__":
    unittest.main()
