"""Phase 8A -- unified-Brier reliability-audit policy tests.

Covers the two fixes over ``_UnifiedBrierPolicy``: (1) AskNew enumerates every
unasked question on one Brier scale with per-question cost, and (2) Stop runs an
explicit VerifyOld audit gated by ``verification_audit_threshold``.  Also locks
in the red lines: the prediction path reads no true state, RAG never enters the
Brier value, the default stays heuristic, and the audit is bounded by
``maximum_verifications``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
from powerful_medrag.worthiness_policy import (
    UnifiedBrierReliabilityAuditPolicy,
    WorthinessStrategy,
    build_policy,
)

C_VERIFY = 0.03

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
        return Observation(FeatureKey(name), value, certainty=certainty)


class TestAskNewEnumeration(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()
        self.cfg = ReliabilityAwarePolicyConfig(retrieval_mode=RetrievalMode.NO_RAG)

    def _rank(self, policy, tracker, reports, asked):
        return policy.rank_actions(
            tracker,
            initial_observations=(),
            reports=tuple(reports),
            asked=asked,
            verified_report_indices=set(),
            verification_count=0,
        )

    def test_01_asknew_enumerates_all_unasked_not_just_eig_best(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        policy = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT, config=self.cfg
        )
        tracker = self.env.tracker_with(reports)
        asked = {r.key for r in reports}
        actions = self._rank(policy, tracker, reports, asked)
        news = [a for a in actions if a.kind is ActionKind.NEW]
        expected = NumpyQuestionSelector().rank(tracker, excluded=asked)
        self.assertEqual(
            len(news), len(expected),
            "AskNew must enumerate every still-unasked question, not just the EIG-best",
        )
        self.assertEqual({a.key for a in news}, {q.key for q in expected})

    def test_02_asknew_uses_per_question_cost_and_brier_best(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        policy = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT, config=self.cfg
        )
        tracker = self.env.tracker_with(reports)
        asked = {r.key for r in reports}
        actions = self._rank(policy, tracker, reports, asked)
        news = [a for a in actions if a.kind is ActionKind.NEW]
        self.assertEqual(actions[0].kind, ActionKind.NEW)
        # C_new,j = new_question_cost_weight * spec.cost_j (0.01 * 1.0 = 0.01)
        for a in news:
            spec = self.env.model.specs[a.key]
            c_new = self.cfg.new_question_cost_weight * spec.cost
            self.assertAlmostEqual(a.utility, asknew_value(tracker, a.key, C_new=c_new))
            self.assertAlmostEqual(a.burden, c_new)
        # top-ranked is the Brier-best (argmax V_new), not necessarily EIG-best
        brier_best = max(news, key=lambda a: a.utility)
        self.assertEqual(actions[0].key, brier_best.key)

    def test_03_eig_best_and_brier_best_are_recorded(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        policy = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT, config=self.cfg
        )
        tracker = self.env.tracker_with(reports)
        asked = {r.key for r in reports}
        self._rank(policy, tracker, reports, asked)
        eig_best = NumpyQuestionSelector().rank(tracker, excluded=asked)[0].key
        self.assertEqual(policy.last_eig_best_key, eig_best)
        # brier best == argmax of the logged v_new
        logged_brier_best = max(
            policy.last_asknew_log, key=lambda e: e["v_new"]
        )["key"]
        self.assertEqual(policy.last_brier_best_key.name, logged_brier_best)
        self.assertEqual(
            sum(e["is_eig_best"] for e in policy.last_asknew_log), 1,
        )
        self.assertEqual(
            sum(e["is_brier_best"] for e in policy.last_asknew_log), 1,
        )


class TestVerifyOldEnumeration(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()
        self.cfg = ReliabilityAwarePolicyConfig(retrieval_mode=RetrievalMode.NO_RAG)

    def _rank(self, policy, tracker, reports, verified=()):
        return policy.rank_actions(
            tracker,
            initial_observations=(),
            reports=tuple(reports),
            asked={r.key for r in reports},
            verified_report_indices=set(verified),
            verification_count=0,
        )

    def test_04_verifyold_enumerates_all_unverified_non_unknown(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        policy = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT, config=self.cfg
        )
        tracker = self.env.tracker_with(reports)
        actions = self._rank(policy, tracker, reports)
        verifies = [a for a in actions if a.kind is ActionKind.VERIFY]
        self.assertEqual(len(verifies), 2)
        self.assertEqual({a.report_index for a in verifies}, {0, 1})
        for a in verifies:
            expected = verify_value(
                tracker, tuple(reports), a.report_index, (),
                C_verify=self.cfg.verification_cost,
            )
            self.assertAlmostEqual(a.utility, expected)

    def test_05_unknown_reports_are_skipped(self):
        reports = [
            self.env.obs("fever"),
            self.env.obs("myalgia", value=UNKNOWN, certainty=CertaintyCue.NONE),
        ]
        policy = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT, config=self.cfg
        )
        tracker = self.env.tracker_with(reports)
        actions = self._rank(policy, tracker, reports)
        verifies = [a for a in actions if a.kind is ActionKind.VERIFY]
        self.assertEqual([a.report_index for a in verifies], [0])

    def test_06_g_verify_is_net_plus_cost_and_audit_state_is_max(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        policy = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT, config=self.cfg
        )
        tracker = self.env.tracker_with(reports)
        self._rank(policy, tracker, reports)
        self.assertIsNotNone(policy._last_audit_g_verify)
        self.assertIsNotNone(policy._last_audit_argmax)
        logged = policy.last_verification_log
        for entry in logged:
            self.assertAlmostEqual(
                entry["g_verify"], entry["v_verify"] + self.cfg.verification_cost,
                places=6,
            )
        raw_g = []
        for index in range(len(reports)):
            v = verify_value(
                tracker, tuple(reports), index, (),
                C_verify=self.cfg.verification_cost,
            )
            raw_g.append((index, v + self.cfg.verification_cost))
        self.assertAlmostEqual(
            policy._last_audit_g_verify, max(g for _, g in raw_g),
        )
        self.assertEqual(policy._last_audit_argmax, max(raw_g, key=lambda t: t[1])[0])


class TestAuditThreshold(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()
        self.policy = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT,
            config=ReliabilityAwarePolicyConfig(retrieval_mode=RetrievalMode.NO_RAG),
        )

    def test_07_audit_returns_verify_when_gross_exceeds_threshold(self):
        self.policy._last_audit_g_verify = 0.05
        self.policy._last_audit_argmax = 2
        action = self.policy._audit_verify_old(verification_count=0)
        self.assertIs(action.kind, ActionKind.VERIFY)
        self.assertEqual(action.report_index, 2)
        self.assertAlmostEqual(action.utility, 0.05 - C_VERIFY)

    def test_08_audit_none_when_below_threshold_or_at_max(self):
        self.policy._last_audit_g_verify = 0.01
        self.policy._last_audit_argmax = 0
        self.assertIsNone(self.policy._audit_verify_old(verification_count=0))
        self.policy._last_audit_g_verify = 0.05
        self.policy._last_audit_argmax = 0
        self.assertIsNone(
            self.policy._audit_verify_old(
                verification_count=self.policy.config.maximum_verifications
            )
        )
        self.policy._last_audit_g_verify = None
        self.policy._last_audit_argmax = None
        self.assertIsNone(self.policy._audit_verify_old(verification_count=0))


class TestPreStopAuditIntegration(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()
        self.cfg = ReliabilityAwarePolicyConfig(retrieval_mode=RetrievalMode.NO_RAG)

    def _choose(self, policy, tracker, reports, verification_count=0):
        return policy.choose_action(
            tracker,
            initial_observations=(),
            reports=tuple(reports),
            asked={r.key for r in reports},
            verified_report_indices=set(),
            verification_count=verification_count,
        )

    def test_09_audit_blocks_stop_when_gross_risk_remains(self):
        policy = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT, config=self.cfg
        )
        tracker = self.env.tracker_with([])
        reports = (self.env.obs("fever"),)
        with mock.patch.object(policy, "_asknew_brier_values", return_value=[]), \
             mock.patch.object(
                 policy, "_verifyold_brier_values", return_value=[(0, 0.0, 0.05)]
             ), \
             mock.patch.object(policy, "_verification_scores", return_value=[]), \
             mock.patch.object(
                 tracker, "ranked_diseases",
                 return_value=[("influenza", 0.92), ("common_cold", 0.08)],
             ):
            action = self._choose(policy, tracker, reports)
        self.assertIs(action.kind, ActionKind.VERIFY)
        self.assertEqual(action.report_index, 0)

    def test_10_audit_does_not_block_when_below_threshold(self):
        policy = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT, config=self.cfg
        )
        tracker = self.env.tracker_with([])
        reports = (self.env.obs("fever"),)
        with mock.patch.object(policy, "_asknew_brier_values", return_value=[]), \
             mock.patch.object(
                 policy, "_verifyold_brier_values", return_value=[(0, 0.0, 0.01)]
             ), \
             mock.patch.object(policy, "_verification_scores", return_value=[]), \
             mock.patch.object(
                 tracker, "ranked_diseases",
                 return_value=[("influenza", 0.92), ("common_cold", 0.08)],
             ):
            action = self._choose(policy, tracker, reports)
        self.assertIs(action.kind, ActionKind.STOP)

    def test_11_maximum_verifications_bounds_audit(self):
        policy = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT,
            config=ReliabilityAwarePolicyConfig(
                retrieval_mode=RetrievalMode.NO_RAG, maximum_verifications=1,
            ),
        )
        tracker = self.env.tracker_with([])
        reports = (self.env.obs("fever"),)
        with mock.patch.object(policy, "_asknew_brier_values", return_value=[]), \
             mock.patch.object(
                 policy, "_verifyold_brier_values", return_value=[(0, 0.0, 0.05)]
             ), \
             mock.patch.object(policy, "_verification_scores", return_value=[]), \
             mock.patch.object(
                 tracker, "ranked_diseases",
                 return_value=[("influenza", 0.92), ("common_cold", 0.08)],
             ):
            action = self._choose(policy, tracker, reports, verification_count=1)
        self.assertIs(action.kind, ActionKind.STOP)


class TestRedLines(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()
        self.cfg = ReliabilityAwarePolicyConfig(retrieval_mode=RetrievalMode.NO_RAG)

    def test_12_prediction_path_reads_no_true_state(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        policy = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT, config=self.cfg
        )
        tracker = self.env.tracker_with(reports)
        asked = {r.key for r in reports}
        args = dict(
            initial_observations=(), reports=tuple(reports), asked=asked,
            verified_report_indices=set(), verification_count=0,
        )
        a_none = policy.rank_actions(tracker, oracle_states=None, **args)
        a_fake = policy.rank_actions(
            tracker, oracle_states={FeatureKey("fever"): "absent"}, **args
        )
        self.assertEqual(
            [(a.kind.value, a.utility, a.report_index) for a in a_none],
            [(a.kind.value, a.utility, a.report_index) for a in a_fake],
        )

    def test_13_rag_does_not_enter_brier_value(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        tracker = self.env.tracker_with(reports)
        no_rag = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT, config=self.cfg
        )
        with tempfile.TemporaryDirectory() as tmp:
            retriever = MedicalRetriever(_write_corpus(tmp, INTEG_CORPUS))
            rag = build_policy(
                WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT,
                config=ReliabilityAwarePolicyConfig(
                    retrieval_mode=RetrievalMode.DYNAMIC_RAG,
                    retrieval_impact_weight=1.0,
                ),
                retriever=retriever,
            )
            def _verify_utils(policy):
                policy.rank_actions(
                    tracker, initial_observations=(), reports=tuple(reports),
                    asked={r.key for r in reports}, verified_report_indices=set(),
                    verification_count=0,
                )
                return sorted(
                    a.utility
                    for a in policy.rank_actions(
                        tracker, initial_observations=(), reports=tuple(reports),
                        asked={r.key for r in reports}, verified_report_indices=set(),
                        verification_count=0,
                    )
                    if a.kind is ActionKind.VERIFY
                )
            self.assertEqual(_verify_utils(no_rag), _verify_utils(rag))

    def test_14_default_stays_heuristic_new_name_enables_audit(self):
        self.assertIs(
            type(build_policy(WorthinessStrategy.HEURISTIC_VERIFY)),
            ReliabilityAwareActionPolicy,
        )
        audit = build_policy(WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT)
        self.assertIs(type(audit), UnifiedBrierReliabilityAuditPolicy)

    def test_15_config_validates_negative_threshold(self):
        with self.assertRaises(ValueError):
            ReliabilityAwarePolicyConfig(verification_audit_threshold=-0.01)


class TestDialogueSmoke(unittest.TestCase):
    def test_16_dialogue_terminates_and_respects_verification_budget(self):
        env = ToyEnv()
        cfg = ReliabilityAwarePolicyConfig(
            retrieval_mode=RetrievalMode.NO_RAG,
            maximum_verifications=1,
            max_total_turns=8,
        )
        policy = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT, config=cfg
        )
        case = next(c for c in env.cases if c.diagnosis == "influenza")
        patient = StructuredPatientSimulator(
            diagnosis=case.diagnosis, latent_states=case.states,
            model=env.model, profile=PatientProfile.from_noise_rate(0.3), seed=9,
        )
        result = run_reliability_aware_dialogue(
            patient, policy=policy, initial_observations=case.initial_observations
        )
        self.assertLessEqual(len(result.turns), cfg.max_total_turns)
        self.assertLessEqual(result.verification_questions, cfg.maximum_verifications)


if __name__ == "__main__":
    unittest.main()
