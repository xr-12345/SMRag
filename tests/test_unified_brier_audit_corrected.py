"""Prompt #18 -- corrected unified-Brier audit policy tests.

The Prompt #17 ``UnifiedBrierReliabilityAuditPolicy`` forced VerifyOld whenever
``G_t^verify >= verification_audit_threshold`` (``_audit_verify_old``).  Prompt #18
corrects this: gross verification gain only *blocks* Stop (strict ``>`` boundary),
and the AskNew/VerifyOld choice is a net-Brier comparison with a global
``verification_advantage_margin`` (tau_A).  The Prompt #17 forced-verify behaviour
is kept as an independent ablation strategy.

This file locks in: (A) the stop-audit-only-blocks-stop semantics, (B) the tau_A
global margin, (C) the unified action-selection fallbacks, (D) the per-turn
decision log, (E) the red lines (no true state, RAG-free Brier value, default
stays heuristic, old ablation preserved), and (F) a dialogue smoke.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from powerful_medrag.action_value import verify_value
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
from powerful_medrag.questioning import QuestionScore
from powerful_medrag.retrieval import MedicalRetriever
from powerful_medrag.schema import UNKNOWN, CertaintyCue, FeatureKey, Observation, VariableSpec
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator
from powerful_medrag.worthiness_policy import (
    CorrectedUnifiedBrierAuditPolicy,
    UnifiedBrierReliabilityAuditPolicy,
    WorthinessStrategy,
    build_policy,
)

C_VERIFY = 0.03

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


def _make_new_entry(key: str, v_new: float) -> tuple[QuestionScore, VariableSpec, float]:
    fk = FeatureKey(key)
    qs = QuestionScore(fk, expected_information_gain=0.5, cost=1.0, utility=0.5,
                       predicted_answers={"present": 0.5, "absent": 0.5})
    spec = VariableSpec(fk, ("present", "absent"), cost=1.0)
    return qs, spec, v_new


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


def _corrected(cfg=None):
    return build_policy(
        WorthinessStrategy.UNIFIED_BRIER_AUDIT_CORRECTED,
        config=cfg or ReliabilityAwarePolicyConfig(retrieval_mode=RetrievalMode.NO_RAG),
    )


class _StubChooser:
    """Patch the value enumeration and confidence for a controlled decision."""

    def __init__(self, policy, tracker, new_entries, verify_entries, top_diseases):
        self.policy = policy
        self.tracker = tracker
        self.patchers = [
            mock.patch.object(policy, "_asknew_brier_values", return_value=new_entries),
            mock.patch.object(policy, "_verifyold_brier_values", return_value=verify_entries),
            mock.patch.object(policy, "_verification_scores", return_value=[]),
            mock.patch.object(tracker, "ranked_diseases", return_value=top_diseases),
        ]

    def __enter__(self):
        for p in self.patchers:
            p.start()
        return self.policy

    def __exit__(self, *exc):
        for p in self.patchers:
            p.stop()


class TestStopAuditOnlyBlocksStop(unittest.TestCase):
    """A. The core fix: gross gain forbids Stop, never forces VerifyOld."""

    def setUp(self):
        self.env = ToyEnv()

    def _choose(self, policy, tracker, reports, new_entries, verify_entries):
        with _StubChooser(policy, tracker, new_entries, verify_entries,
                          [("influenza", 0.92), ("common_cold", 0.08)]):
            return policy.choose_action(
                tracker,
                initial_observations=(),
                reports=tuple(reports),
                asked={r.key for r in reports},
                verified_report_indices=set(),
                verification_count=0,
            )

    def test_01_gross_gain_blocks_stop_but_does_not_force_verify(self):
        # base_stop_ready True (both utilities <= 0.03) but gross gain 0.05 > tau_V.
        # Old code returned VerifyOld here; corrected returns AskNew (net comparison).
        policy = _corrected()
        tracker = self.env.tracker_with([])
        reports = (self.env.obs("fever"),)
        action = self._choose(
            policy, tracker, reports,
            new_entries=[_make_new_entry("dry_cough", 0.02)],
            verify_entries=[(0, 0.02, 0.05)],
        )
        self.assertIs(action.kind, ActionKind.NEW)
        self.assertTrue(policy.last_decision_log["stop_blocked_by_verify_audit"])
        self.assertTrue(policy.last_decision_log["base_stop_ready"])
        self.assertTrue(policy.last_decision_log["audit_blocks_stop"])

    def test_02_returns_stop_when_gross_gain_below_threshold(self):
        policy = _corrected()
        tracker = self.env.tracker_with([])
        reports = (self.env.obs("fever"),)
        action = self._choose(
            policy, tracker, reports,
            new_entries=[_make_new_entry("dry_cough", 0.02)],
            verify_entries=[(0, 0.0, 0.01)],
        )
        self.assertIs(action.kind, ActionKind.STOP)
        self.assertFalse(policy.last_decision_log["audit_blocks_stop"])

    def test_03_strict_greater_boundary_at_equality_does_not_block(self):
        # G_verify == tau_V exactly -> must NOT block stop (strict > boundary).
        policy = _corrected()
        tracker = self.env.tracker_with([])
        reports = (self.env.obs("fever"),)
        # v_verify = 0.0 -> gross = 0.0 + 0.03 == tau_V (0.03)
        action = self._choose(
            policy, tracker, reports,
            new_entries=[_make_new_entry("dry_cough", 0.02)],
            verify_entries=[(0, 0.0, 0.03)],
        )
        self.assertIs(action.kind, ActionKind.STOP)
        self.assertFalse(policy.last_decision_log["audit_blocks_stop"])


class TestGlobalAdvantageMargin(unittest.TestCase):
    """B. tau_A applies globally to every AskNew/VerifyOld comparison."""

    def setUp(self):
        self.env = ToyEnv()

    def _choose(self, policy, tracker, reports, new_entries, verify_entries):
        with _StubChooser(policy, tracker, new_entries, verify_entries,
                          [("influenza", 0.92), ("common_cold", 0.08)]):
            return policy.choose_action(
                tracker,
                initial_observations=(),
                reports=tuple(reports),
                asked={r.key for r in reports},
                verified_report_indices=set(),
                verification_count=0,
            )

    def test_04_verify_wins_only_above_margin(self):
        # v_verify 0.12 > v_new 0.05 + tau_A 0.03 = 0.08 -> VerifyOld.
        policy = _corrected()
        tracker = self.env.tracker_with([])
        reports = (self.env.obs("fever"),)
        action = self._choose(
            policy, tracker, reports,
            new_entries=[_make_new_entry("dry_cough", 0.05)],
            verify_entries=[(0, 0.12, 0.15)],
        )
        self.assertIs(action.kind, ActionKind.VERIFY)
        self.assertEqual(action.report_index, 0)

    def test_05_asknew_wins_within_margin(self):
        # v_verify 0.07 <= v_new 0.05 + tau_A 0.03 = 0.08 -> AskNew.
        policy = _corrected()
        tracker = self.env.tracker_with([])
        reports = (self.env.obs("fever"),)
        action = self._choose(
            policy, tracker, reports,
            new_entries=[_make_new_entry("dry_cough", 0.05)],
            verify_entries=[(0, 0.07, 0.10)],
        )
        self.assertIs(action.kind, ActionKind.NEW)

    def test_06_margin_applies_when_base_stop_not_ready(self):
        # base_stop_ready False (best_new.utility 0.05 > 0.03) but the margin still
        # decides AskNew (0.07 <= 0.05 + 0.03) -- the margin is global, not only
        # after a blocked stop.
        policy = _corrected()
        tracker = self.env.tracker_with([])
        reports = (self.env.obs("fever"),)
        action = self._choose(
            policy, tracker, reports,
            new_entries=[_make_new_entry("dry_cough", 0.05)],
            verify_entries=[(0, 0.07, 0.10)],
        )
        self.assertIs(action.kind, ActionKind.NEW)
        self.assertFalse(policy.last_decision_log["base_stop_ready"])


class TestActionSelectionFallbacks(unittest.TestCase):
    """C. Unified comparison fallbacks."""

    def setUp(self):
        self.env = ToyEnv()

    def _choose(self, policy, tracker, reports, new_entries, verify_entries):
        with _StubChooser(policy, tracker, new_entries, verify_entries,
                          [("influenza", 0.92), ("common_cold", 0.08)]):
            return policy.choose_action(
                tracker,
                initial_observations=(),
                reports=tuple(reports),
                asked={r.key for r in reports},
                verified_report_indices=set(),
                verification_count=0,
            )

    def test_07_asknew_returned_when_no_verify_candidate(self):
        policy = _corrected()
        tracker = self.env.tracker_with([])
        reports = ()
        action = self._choose(policy, tracker, reports,
                              new_entries=[_make_new_entry("dry_cough", 0.10)],
                              verify_entries=[])
        self.assertIs(action.kind, ActionKind.NEW)

    def test_08_verify_returned_when_no_asknew_and_positive(self):
        policy = _corrected()
        tracker = self.env.tracker_with([])
        reports = (self.env.obs("fever"),)
        action = self._choose(policy, tracker, reports, new_entries=[],
                              verify_entries=[(0, 0.10, 0.13)])
        self.assertIs(action.kind, ActionKind.VERIFY)

    def test_09_uncertainty_stop_when_neither_positive(self):
        policy = _corrected()
        tracker = self.env.tracker_with([])
        reports = ()
        action = self._choose(policy, tracker, reports, new_entries=[],
                              verify_entries=[])
        self.assertIs(action.kind, ActionKind.STOP)


class TestGrossGainAndBudget(unittest.TestCase):
    """D. max_gross_verify_gain + maximum_verifications bound."""

    def setUp(self):
        self.env = ToyEnv()

    def test_10_gross_gain_is_net_plus_cost(self):
        policy = _corrected()
        tracker = self.env.tracker_with([])
        reports = (self.env.obs("fever"),)
        with _StubChooser(policy, tracker, [_make_new_entry("dry_cough", 0.02)],
                          [(0, 0.08, 0.11)],
                          [("influenza", 0.92), ("common_cold", 0.08)]):
            policy.choose_action(
                tracker, initial_observations=(), reports=tuple(reports),
                asked={r.key for r in reports}, verified_report_indices=set(),
                verification_count=0,
            )
        self.assertAlmostEqual(policy.last_decision_log["best_verify_gross_gain"], 0.11)
        self.assertAlmostEqual(policy.last_decision_log["best_verify_net_value"], 0.08)

    def test_11_audit_does_not_block_at_max_verifications(self):
        policy = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_AUDIT_CORRECTED,
            config=ReliabilityAwarePolicyConfig(
                retrieval_mode=RetrievalMode.NO_RAG, maximum_verifications=1,
            ),
        )
        tracker = self.env.tracker_with([])
        reports = (self.env.obs("fever"),)
        # verify_values is empty at max (rank_actions gated), so no VERIFY candidate.
        with _StubChooser(policy, tracker, [_make_new_entry("dry_cough", 0.02)], [],
                          [("influenza", 0.92), ("common_cold", 0.08)]):
            action = policy.choose_action(
                tracker, initial_observations=(), reports=tuple(reports),
                asked={r.key for r in reports}, verified_report_indices=set(),
                verification_count=1,
            )
        self.assertIs(action.kind, ActionKind.STOP)
        self.assertFalse(policy.last_decision_log["audit_blocks_stop"])


class TestDecisionLog(unittest.TestCase):
    def test_12_decision_log_has_all_required_fields(self):
        env = ToyEnv()
        policy = _corrected()
        tracker = env.tracker_with([])
        reports = (env.obs("fever"),)
        with _StubChooser(policy, tracker, [_make_new_entry("dry_cough", 0.02)],
                          [(0, 0.02, 0.05)],
                          [("influenza", 0.92), ("common_cold", 0.08)]):
            policy.choose_action(
                tracker, initial_observations=(), reports=tuple(reports),
                asked={r.key for r in reports}, verified_report_indices=set(),
                verification_count=0,
            )
        required = {
            "best_eig_question", "best_brier_question", "eig_brier_agreement",
            "best_new_gross_gain", "best_new_net_value", "best_verify_report_index",
            "best_verify_gross_gain", "best_verify_net_value",
            "verification_audit_threshold", "verification_advantage_margin",
            "base_stop_ready", "audit_blocks_stop", "stop_blocked_by_verify_audit",
            "chosen_action", "retrieval_triggered",
        }
        self.assertEqual(set(policy.last_decision_log), required)
        self.assertIn(policy.last_decision_log["chosen_action"],
                      {ActionKind.NEW.value, ActionKind.VERIFY.value, ActionKind.STOP.value})


class TestRedLines(unittest.TestCase):
    def setUp(self):
        self.env = ToyEnv()

    def test_13_corrected_prediction_path_reads_no_true_state(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        policy = _corrected()
        tracker = self.env.tracker_with(reports)
        args = dict(
            initial_observations=(), reports=tuple(reports),
            asked={r.key for r in reports}, verified_report_indices=set(),
            verification_count=0,
        )
        a_none = policy.choose_action(tracker, oracle_states=None, **args)
        log_none = dict(policy.last_decision_log)
        a_fake = policy.choose_action(
            tracker, oracle_states={FeatureKey("fever"): "absent"}, **args
        )
        log_fake = dict(policy.last_decision_log)
        self.assertEqual(
            (a_none.kind.value, a_none.utility, a_none.report_index, a_none.key),
            (a_fake.kind.value, a_fake.utility, a_fake.report_index, a_fake.key),
        )
        self.assertEqual(log_none, log_fake)

    def test_14_rag_does_not_enter_corrected_brier_value(self):
        reports = [self.env.obs("fever"), self.env.obs("myalgia")]
        tracker = self.env.tracker_with(reports)
        no_rag = _corrected()
        with tempfile.TemporaryDirectory() as tmp:
            retriever = MedicalRetriever(_write_corpus(tmp, INTEG_CORPUS))
            rag = build_policy(
                WorthinessStrategy.UNIFIED_BRIER_AUDIT_CORRECTED,
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
                    a.utility for a in policy.rank_actions(
                        tracker, initial_observations=(), reports=tuple(reports),
                        asked={r.key for r in reports}, verified_report_indices=set(),
                        verification_count=0,
                    )
                    if a.kind is ActionKind.VERIFY
                )
            self.assertEqual(_verify_utils(no_rag), _verify_utils(rag))

    def test_15_forced_verify_ablation_is_preserved(self):
        # Prompt #17 forced-verify behaviour stays as an independent strategy and
        # still returns VerifyOld when gross gain exceeds the threshold.
        forced = build_policy(WorthinessStrategy.UNIFIED_BRIER_AUDIT_FORCED_VERIFY)
        self.assertIs(type(forced), UnifiedBrierReliabilityAuditPolicy)
        # legacy name maps to the same class (backwards compatible)
        legacy = build_policy(WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT)
        self.assertIs(type(legacy), UnifiedBrierReliabilityAuditPolicy)
        forced._last_audit_g_verify = 0.05
        forced._last_audit_argmax = 2
        audited = forced._audit_verify_old(verification_count=0)
        self.assertIs(audited.kind, ActionKind.VERIFY)
        self.assertEqual(audited.report_index, 2)

    def test_16_heuristic_baseline_byte_identical(self):
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

    def test_17_config_validates_negative_advantage_margin(self):
        with self.assertRaises(ValueError):
            ReliabilityAwarePolicyConfig(verification_advantage_margin=-0.01)
        cfg = ReliabilityAwarePolicyConfig()
        self.assertEqual(cfg.verification_advantage_margin, 0.03)
        self.assertEqual(cfg.verification_audit_threshold, 0.03)


class TestDialogueSmoke(unittest.TestCase):
    def test_18_corrected_dialogue_terminates_and_respects_budget(self):
        env = ToyEnv()
        cfg = ReliabilityAwarePolicyConfig(
            retrieval_mode=RetrievalMode.NO_RAG,
            maximum_verifications=1,
            max_total_turns=8,
        )
        policy = build_policy(
            WorthinessStrategy.UNIFIED_BRIER_AUDIT_CORRECTED, config=cfg
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
