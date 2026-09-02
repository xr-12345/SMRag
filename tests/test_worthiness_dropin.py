"""Phase 6 -- VerifyOld drop-in isolation tests (13 spec tests).

The learned worthiness model may act *only* when the heuristic controller chose
VerifyOld, and then only re-rank the report or replace the verification with an
AskNew.  It never touches AskNew, never touches Stop, never opens a new
VerifyOld, and never reads the true disease / latent state / true wrongness /
noise label.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

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
from powerful_medrag.retrieval import MedicalRetriever
from powerful_medrag.schema import CertaintyCue, FeatureKey, Observation
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator
from powerful_medrag.verification_worthiness import (
    BASE_FEATURES,
    LearnedWorthinessModel,
)
from powerful_medrag.worthiness_dropin import (
    DropinPolicy,
    DropinStrategy,
    build_dropin_policy,
)
from powerful_medrag.worthiness_policy import (
    build_online_features,
    load_frozen_worthiness_model,
)

SRC = Path("src/powerful_medrag")


class _ErrorProbGain:
    """Gain head proportional to the first feature (error probability)."""

    def predict(self, X):
        return X[:, 0].astype(float)


class _ConstGain:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value)


class _ConstHarm:
    def __init__(self, value: float):
        self.value = float(value)

    def predict_proba(self, X):
        p = np.zeros((len(X), 2))
        p[:, 1] = self.value
        return p


def _stub_model(gain, harm, *, tau_harm=0.4129, n_features=7):
    if isinstance(gain, (int, float)):
        gain = _ConstGain(gain)
    return LearnedWorthinessModel(
        gain_model=gain,
        harm_model=_ConstHarm(harm),
        tau_harm=tau_harm,
        gain_kind="stub",
        harm_kind="stub",
        n_features=n_features,
        rag_mode="none" if n_features == 7 else "real",
    )


def _verify_config():
    """A config under which the heuristic controller chooses VerifyOld."""
    return ReliabilityAwarePolicyConfig(
        verification_cost=0.0,
        decision_impact_weight=1e9,
        minimum_unreliable_history_cues_for_verification=0,
        retrieval_mode=RetrievalMode.NO_RAG,
    )


class ToyEnv:
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
        return Observation(FeatureKey(name), value, certainty)


class TestDropinBaseline(unittest.TestCase):
    def test_01_heuristic_baseline_regression_identical(self):
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
                patient, policy=policy,
                initial_observations=case.initial_observations,
            )

        base = run(ReliabilityAwareActionPolicy(config=cfg))
        built = run(build_dropin_policy(
            DropinStrategy.HEURISTIC_BASELINE, config=cfg,
        ))
        self.assertIs(
            type(build_dropin_policy(DropinStrategy.HEURISTIC_BASELINE, config=cfg)),
            ReliabilityAwareActionPolicy,
        )
        self.assertEqual(
            [t.action.kind.value for t in base.turns],
            [t.action.kind.value for t in built.turns],
        )
        self.assertEqual(base.stop_reason, built.stop_reason)
        self.assertEqual(base.predicted_diagnosis, built.predicted_diagnosis)


class TestControllerIsolation(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()

    def test_02_learned_cannot_change_asknew(self):
        cfg = ReliabilityAwarePolicyConfig(retrieval_mode=RetrievalMode.NO_RAG)
        policy = build_dropin_policy(
            DropinStrategy.LEARNED_FULL_RERANK, config=cfg,
            worthiness_model=_stub_model(0.5, 0.1),
        )
        tracker = self.env.tracker_with(())  # no reports -> controller asks
        controller = ReliabilityAwareActionPolicy(config=cfg).choose_action(
            tracker, initial_observations=(), reports=(), asked=set(),
            verified_report_indices=set(), verification_count=0,
        )
        self.assertIs(controller.kind, ActionKind.NEW)
        action = policy.choose_action(
            tracker, initial_observations=(), reports=(), asked=set(),
            verified_report_indices=set(), verification_count=0,
        )
        self.assertIs(action.kind, ActionKind.NEW)
        self.assertEqual(action.key, controller.key)
        self.assertEqual(policy.last_dropin_log[-1]["final_action_type"], "new")

    def test_03_learned_cannot_change_stop(self):
        cfg = ReliabilityAwarePolicyConfig(retrieval_mode=RetrievalMode.NO_RAG)
        policy = build_dropin_policy(
            DropinStrategy.LEARNED_FULL_RERANK, config=cfg,
            worthiness_model=_stub_model(0.5, 0.1),
        )
        tracker = BeliefTracker(self.env.model)
        tracker.belief = {
            d: (0.99 if i == 0 else 0.01 / (len(self.env.model.diseases) - 1))
            for i, d in enumerate(self.env.model.diseases)
        }
        controller = ReliabilityAwareActionPolicy(config=cfg).choose_action(
            tracker, initial_observations=(), reports=(), asked=set(),
            verified_report_indices=set(), verification_count=0,
        )
        self.assertIs(controller.kind, ActionKind.STOP)
        action = policy.choose_action(
            tracker, initial_observations=(), reports=(), asked=set(),
            verified_report_indices=set(), verification_count=0,
        )
        self.assertIs(action.kind, ActionKind.STOP)
        self.assertEqual(policy.last_dropin_log[-1]["final_action_type"], "stop")

    def test_04_learned_never_opens_a_new_verify(self):
        cfg = ReliabilityAwarePolicyConfig(retrieval_mode=RetrievalMode.NO_RAG)
        policy = build_dropin_policy(
            DropinStrategy.LEARNED_FULL_RERANK, config=cfg,
            worthiness_model=_stub_model(0.5, 0.1),
        )
        # empty reports -> controller = new
        t_new = self.env.tracker_with(())
        a = policy.choose_action(
            t_new, initial_observations=(), reports=(), asked=set(),
            verified_report_indices=set(), verification_count=0,
        )
        self.assertIsNot(a.kind, ActionKind.VERIFY)
        # concentrated belief -> controller = stop
        t_stop = BeliefTracker(self.env.model)
        t_stop.belief = {
            d: (0.99 if i == 0 else 0.01 / (len(self.env.model.diseases) - 1))
            for i, d in enumerate(self.env.model.diseases)
        }
        b = policy.choose_action(
            t_stop, initial_observations=(), reports=(), asked=set(),
            verified_report_indices=set(), verification_count=0,
        )
        self.assertIsNot(b.kind, ActionKind.VERIFY)


class TestRerank(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()
        self.cfg = _verify_config()

    def test_05_rerank_only_changes_report_index(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        policy = build_dropin_policy(
            DropinStrategy.LEARNED_FULL_RERANK, config=self.cfg,
            worthiness_model=_stub_model(_ErrorProbGain(), 0.1),
        )
        tracker = self.env.tracker_with(reports)
        controller = ReliabilityAwareActionPolicy(config=self.cfg).choose_action(
            tracker, initial_observations=(), reports=tuple(reports),
            asked={r.key for r in reports}, verified_report_indices=set(),
            verification_count=0,
        )
        self.assertIs(controller.kind, ActionKind.VERIFY)
        # expected learned choice = argmax error_probability
        scores = policy._verification_scores(
            tracker, initial_observations=(), reports=tuple(reports),
            verified_report_indices=set(), report_risks=(),
        )
        expected = max(scores, key=lambda s: s.error_probability).report_index
        action = policy.choose_action(
            tracker, initial_observations=(), reports=tuple(reports),
            asked={r.key for r in reports}, verified_report_indices=set(),
            verification_count=0,
        )
        self.assertIs(action.kind, ActionKind.VERIFY)
        self.assertEqual(action.report_index, expected)
        log = policy.last_dropin_log[-1]
        self.assertEqual(log["controller_action_type"], "verify")
        self.assertEqual(log["final_action_type"], "verify")
        self.assertEqual(log["heuristic_selected_report"], controller.report_index)
        self.assertEqual(log["learned_selected_report"], expected)

    def test_06_rerank_falls_back_when_no_safe_candidate(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        # all harm high (> tau_harm) -> no safe candidate -> fallback
        policy = build_dropin_policy(
            DropinStrategy.LEARNED_FULL_RERANK, config=self.cfg,
            worthiness_model=_stub_model(0.5, 0.9),
        )
        tracker = self.env.tracker_with(reports)
        controller = ReliabilityAwareActionPolicy(config=self.cfg).choose_action(
            tracker, initial_observations=(), reports=tuple(reports),
            asked={r.key for r in reports}, verified_report_indices=set(),
            verification_count=0,
        )
        self.assertIs(controller.kind, ActionKind.VERIFY)
        action = policy.choose_action(
            tracker, initial_observations=(), reports=tuple(reports),
            asked={r.key for r in reports}, verified_report_indices=set(),
            verification_count=0,
        )
        self.assertIs(action.kind, ActionKind.VERIFY)
        self.assertEqual(action.report_index, controller.report_index)
        log = policy.last_dropin_log[-1]
        self.assertTrue(log["fallback_to_heuristic"])
        self.assertEqual(log["learned_selected_report"], controller.report_index)


class TestFilter(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()
        self.cfg = _verify_config()

    def test_07_filter_replaces_verify_with_asknew_not_stop(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        # net_value = 0.02 - 0.03 < 0 -> no candidate passes -> replace with AskNew
        policy = build_dropin_policy(
            DropinStrategy.LEARNED_FULL_FILTER, config=self.cfg,
            worthiness_model=_stub_model(0.02, 0.1),
        )
        tracker = self.env.tracker_with(reports)
        controller = ReliabilityAwareActionPolicy(config=self.cfg).choose_action(
            tracker, initial_observations=(), reports=tuple(reports),
            asked={r.key for r in reports}, verified_report_indices=set(),
            verification_count=0,
        )
        self.assertIs(controller.kind, ActionKind.VERIFY)
        action = policy.choose_action(
            tracker, initial_observations=(), reports=tuple(reports),
            asked={r.key for r in reports}, verified_report_indices=set(),
            verification_count=0,
        )
        self.assertIs(action.kind, ActionKind.NEW)
        self.assertIsNot(action.kind, ActionKind.STOP)
        log = policy.last_dropin_log[-1]
        self.assertEqual(log["final_action_type"], "new")
        self.assertTrue(log["replaced_verify_with_asknew"])
        self.assertIsNotNone(action.key)

    def test_08_filter_still_consumes_one_atomic_question(self):
        cfg = ReliabilityAwarePolicyConfig(
            verification_cost=0.0,
            decision_impact_weight=1e9,
            minimum_unreliable_history_cues_for_verification=0,
            retrieval_mode=RetrievalMode.NO_RAG,
            max_total_turns=8,
        )
        case = next(c for c in self.env.cases if c.diagnosis == "influenza")
        policy = build_dropin_policy(
            DropinStrategy.LEARNED_FULL_FILTER, config=cfg,
            worthiness_model=_stub_model(0.02, 0.1),  # always reject -> AskNew
        )
        patient = StructuredPatientSimulator(
            diagnosis=case.diagnosis, latent_states=case.states,
            model=self.env.model, profile=PatientProfile.from_noise_rate(0.3),
            seed=9,
        )
        result = run_reliability_aware_dialogue(
            patient, policy=policy,
            initial_observations=case.initial_observations,
        )
        self.assertEqual(
            len(result.turns),
            result.new_questions + result.verification_questions,
        )


class TestRagSplit(unittest.TestCase):
    def test_09_full_model_uses_rag_features(self):
        model = load_frozen_worthiness_model("real")
        self.assertEqual(model.n_features, len(BASE_FEATURES) + 2)
        policy = build_dropin_policy(
            DropinStrategy.LEARNED_FULL_RERANK,
            config=ReliabilityAwarePolicyConfig(
                retrieval_mode=RetrievalMode.DYNAMIC_RAG,
                retrieval_impact_weight=1.0,
            ),
            worthiness_model=model,
        )
        self.assertTrue(policy._use_rag)
        class _S:
            error_probability = 0.3
            diagnostic_influence = 0.4
            retrieval_impact = 0.5
        x = build_online_features(_S(), current_risk=0.6, n_reports=2,
                                  turn_index=3, use_rag=True)
        self.assertEqual(x.shape, (9,))

    def test_10_norag_model_ignores_rag_features(self):
        model = load_frozen_worthiness_model("none")
        self.assertEqual(model.n_features, len(BASE_FEATURES))
        policy = build_dropin_policy(
            DropinStrategy.LEARNED_NORAG_FILTER,
            config=ReliabilityAwarePolicyConfig(
                retrieval_mode=RetrievalMode.NO_RAG,
            ),
            worthiness_model=model,
        )
        self.assertFalse(policy._use_rag)
        class _S:
            error_probability = 0.3
            diagnostic_influence = 0.4
            retrieval_impact = 0.9
        x = build_online_features(_S(), current_risk=0.6, n_reports=2,
                                  turn_index=3, use_rag=False)
        self.assertEqual(x.shape, (7,))
        self.assertEqual(x[3], 0.6)


class TestNoTrueState(unittest.TestCase):
    def test_11_prediction_path_reads_no_true_state(self):
        env = ToyEnv()
        cfg = _verify_config()
        reports = [env.obs("fever"), env.obs("myalgia")]
        policy = build_dropin_policy(
            DropinStrategy.LEARNED_FULL_RERANK, config=cfg,
            worthiness_model=_stub_model(0.5, 0.1),
        )
        tracker = env.tracker_with(reports)
        args = dict(
            initial_observations=(), reports=tuple(reports),
            asked={r.key for r in reports}, verified_report_indices=set(),
            verification_count=0,
        )
        a_none = policy.choose_action(tracker, oracle_states=None, **args)
        a_fake = policy.choose_action(
            tracker, oracle_states={FeatureKey("fever"): "absent"}, **args
        )
        self.assertEqual(
            (a_none.kind.value, a_none.report_index),
            (a_fake.kind.value, a_fake.report_index),
        )
        src = (SRC / "worthiness_dropin.py").read_text()
        self.assertNotIn("latent_states", src)
        self.assertNotIn("true_diagnosis", src)
        self.assertNotIn("oracle_correction", src)


class TestFrozenModels(unittest.TestCase):
    def test_12_frozen_model_load_predict_consistent(self):
        real = load_frozen_worthiness_model("real")
        none = load_frozen_worthiness_model("none")
        self.assertEqual(real.gain_kind, "hgb")
        self.assertEqual(real.harm_kind, "hgb")
        self.assertAlmostEqual(real.tau_harm, 0.4129, places=3)
        Xr = np.full((4, 9), 0.1)
        g1, h1 = real.predict(Xr)
        real2 = load_frozen_worthiness_model("real")
        g2, h2 = real2.predict(Xr)
        np.testing.assert_array_equal(g1, g2)
        np.testing.assert_array_equal(h1, h2)
        self.assertEqual(g1.shape, (4,))
        self.assertEqual(h1.shape, (4,))
        self.assertEqual(none.predict(np.zeros((3, 7)))[0].shape, (3,))


class TestDefaultStaysHeuristic(unittest.TestCase):
    def test_13_default_strategy_stays_heuristic(self):
        policy = build_dropin_policy(DropinStrategy.HEURISTIC_BASELINE)
        self.assertIs(type(policy), ReliabilityAwareActionPolicy)
        # decision.py must not import the drop-in module (default intact)
        decision_src = (SRC / "decision.py").read_text()
        self.assertNotIn("worthiness_dropin", decision_src)
        self.assertNotIn("worthiness_policy", decision_src)


if __name__ == "__main__":
    unittest.main()
