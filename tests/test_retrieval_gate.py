"""Targeted tests for the retrieval joint gate (VeriMedRAG phase 2A).

Two modes of ``retrieval_gate_mode`` are exercised:

* ``RANK_ONLY`` (default) — retrieval impact only re-ranks VerifyOld candidates
  that the history/learned gate has already allowed; it never opens the gate.
* ``JOINT_GATE`` — a label-free activation value
  ``error_probability * normalized_retrieval_impact - verification_cost`` may
  open the gate when its maximum clears ``minimum_action_utility``.

All scenarios use the toy model plus a tiny in-memory corpus so the tests are
fast and independent of the 209 MB MedRAG Textbooks download.  The prediction
path never reads latent state, the true disease, or the noise label.
"""

import json
import tempfile
import unittest
from pathlib import Path

from powerful_medrag.belief import BeliefTracker
from powerful_medrag.decision import (
    ActionKind,
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    RetrievalGateMode,
    RetrievalMode,
    run_reliability_aware_dialogue,
)
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.gating import LEARNED_GATE_FEATURES, LearnedMisreportGate
from powerful_medrag.retrieval import MedicalRetriever
from powerful_medrag.schema import UNKNOWN, CertaintyCue, FeatureKey, Observation
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator

# Corpus supports "fever" and "myalgia" only; other toy features have no snippet,
# so deleting them leaves retrieval unchanged (impact 0).
INTEG_CORPUS = [
    {"id": "doc_fever", "title": "Fever", "content": "fever",
     "contents": "fever is elevated body temperature"},
    {"id": "doc_myalgia", "title": "Myalgia", "content": "myalgia",
     "contents": "myalgia is muscle pain"},
]

_EXPECTED_LOG_FIELDS = {
    "report_index", "evidence_code", "error_probability",
    "raw_retrieval_impact", "normalized_retrieval_impact",
    "retrieval_activation_value", "existing_verification_utility",
    "final_verification_utility", "old_history_gate_open",
    "opened_by_retrieval", "chosen_action", "best_asknew_utility",
}


def _write_corpus(tmpdir: str, docs: list[dict]) -> str:
    path = Path(tmpdir) / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc) + "\n")
    return tmpdir


def _constant_learned_gate() -> LearnedMisreportGate:
    return LearnedMisreportGate(
        intercept=0.0,
        coefficients={feature: 0.0 for feature in LEARNED_GATE_FEATURES},
    )


class RetrievalGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.retriever = MedicalRetriever(_write_corpus(self.tmp.name, INTEG_CORPUS))
        cases, specs = generate_toy_cases(cases_per_disease=30, seed=41)
        self.model = DiseaseStateModel.fit(cases, specs)
        self.cases = cases

    def tearDown(self):
        self.tmp.cleanup()

    def _policy(self, *, gate_mode=RetrievalGateMode.RANK_ONLY, weight=0.0,
                retrieval_mode=RetrievalMode.DYNAMIC_RAG, **cfg):
        return ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(
                retrieval_mode=retrieval_mode,
                retrieval_gate_mode=gate_mode,
                retrieval_impact_weight=weight,
                **cfg,
            ),
            retriever=self.retriever,
        )

    def _rank(self, policy, reports, *, initial=()):
        tracker = BeliefTracker(self.model)
        for obs in initial:
            tracker.update(obs)
        for obs in reports:
            tracker.update(obs)
        actions = policy.rank_actions(
            tracker,
            initial_observations=tuple(initial),
            reports=tuple(reports),
            asked={r.key for r in reports} | {r.key for r in initial},
            verified_report_indices=set(),
            verification_count=0,
        )
        return tracker, actions

    # --- 1. rank_only keeps the old gate behaviour ------------------------- #
    def test_rank_only_keeps_gate_closed_and_reranks(self):
        c = CertaintyCue.CERTAIN
        # (a) high-impact reports do NOT open the gate in rank_only
        reports = [
            Observation(FeatureKey("runny_nose"), "present", certainty=c),
            Observation(FeatureKey("itchy_eyes"), "present", certainty=c),
            Observation(FeatureKey("fever"), "present", certainty=c),
        ]
        policy = self._policy(gate_mode=RetrievalGateMode.RANK_ONLY, weight=2.0)
        _, actions = self._rank(policy, reports)
        self.assertFalse([a for a in actions if a.kind is ActionKind.VERIFY])
        self.assertTrue(
            all(not e["opened_by_retrieval"] for e in policy.last_verification_log)
        )

        # (b) with a history cue the gate opens, retrieval re-ranks the utility,
        # but the candidate is never flagged retrieval-triggered
        initial = (Observation(FeatureKey("sore_throat"), UNKNOWN,
                               certainty=CertaintyCue.UNCERTAIN),)
        reports2 = [Observation(FeatureKey("fever"), "present", certainty=c)]
        rag = self._policy(gate_mode=RetrievalGateMode.RANK_ONLY, weight=2.0)
        base = self._policy(gate_mode=RetrievalGateMode.RANK_ONLY, weight=0.0)
        _, rag_actions = self._rank(rag, reports2, initial=initial)
        _, base_actions = self._rank(base, reports2, initial=initial)
        rag_verify = {a.report_index: a for a in rag_actions if a.kind is ActionKind.VERIFY}
        base_verify = {a.report_index: a for a in base_actions if a.kind is ActionKind.VERIFY}
        self.assertTrue(rag_verify)
        self.assertGreater(rag_verify[0].utility, base_verify[0].utility)
        self.assertTrue(all(not a.retrieval_triggered for a in rag_verify.values()))

    # --- 2. joint gate opens on high risk + high impact -------------------- #
    def test_joint_gate_opens_on_high_risk_high_impact(self):
        c = CertaintyCue.CERTAIN
        # runny_nose + itchy_eyes push belief away from influenza; fever=present
        # is then both surprising (high error probability) and corpus-backed
        # (deleting it wipes retrieval) -> activation clears the bar.
        reports = [
            Observation(FeatureKey("runny_nose"), "present", certainty=c),
            Observation(FeatureKey("itchy_eyes"), "present", certainty=c),
            Observation(FeatureKey("fever"), "present", certainty=c),
        ]
        policy = self._policy(gate_mode=RetrievalGateMode.JOINT_GATE, weight=2.0)
        _, actions = self._rank(policy, reports)
        verify = [a for a in actions if a.kind is ActionKind.VERIFY]
        self.assertTrue(any(a.retrieval_triggered for a in verify))
        self.assertEqual(actions[0].kind, ActionKind.VERIFY)
        self.assertTrue(actions[0].retrieval_triggered)
        self.assertEqual(actions[0].report_index, 2)
        self.assertTrue(
            any(e["opened_by_retrieval"] for e in policy.last_verification_log)
        )

    # --- 3. high impact + low error probability -> stays below the bar ----- #
    def test_high_impact_low_error_probability_stays_below_bar(self):
        policy = self._policy(gate_mode=RetrievalGateMode.JOINT_GATE)
        # impact is maximal (1.0) but the error probability is tiny, so the
        # activation value 0.05 * 1.0 - cost falls below the minimum utility.
        activation = policy._retrieval_activation_value(0.05, 1.0)
        self.assertLess(activation, policy.config.minimum_action_utility)

    # --- 4. high error probability + low impact -> no open ----------------- #
    def test_high_error_probability_low_impact_does_not_open(self):
        c = CertaintyCue.CERTAIN
        # sore_throat has no corpus snippet, so impact is 0 and the activation is
        # -cost < 0 regardless of how high the error probability is.
        reports = [
            Observation(FeatureKey("runny_nose"), "present", certainty=c),
            Observation(FeatureKey("itchy_eyes"), "present", certainty=c),
            Observation(FeatureKey("sore_throat"), "present", certainty=c),
        ]
        policy = self._policy(gate_mode=RetrievalGateMode.JOINT_GATE, weight=2.0)
        _, actions = self._rank(policy, reports)
        self.assertFalse([a for a in actions if a.kind is ActionKind.VERIFY])
        self.assertLess(policy._retrieval_activation_value(0.9, 0.0), 0.0)

    # --- 5. activation below minimum_action_utility -> no open ------------- #
    def test_activation_below_minimum_utility_does_not_open(self):
        c = CertaintyCue.CERTAIN
        # fever alone clears the *default* bar but not a raised one.
        reports = [Observation(FeatureKey("fever"), "present", certainty=c)]
        policy = self._policy(
            gate_mode=RetrievalGateMode.JOINT_GATE, weight=2.0,
            minimum_action_utility=0.5,
        )
        _, actions = self._rank(policy, reports)
        self.assertFalse([a for a in actions if a.kind is ActionKind.VERIFY])
        self.assertLess(policy._retrieval_activation_value(0.5, 0.5), 0.5)

    # --- 6. VerifyOld still competes; AskNew can win ----------------------- #
    def test_asknew_wins_over_retrieval_opened_verify(self):
        c = CertaintyCue.CERTAIN
        # fever alone: the joint gate opens (activation ~0.04 >= 0.03), but with
        # retrieval_impact_weight=0 the verify utility stays low, so AskNew wins.
        reports = [Observation(FeatureKey("fever"), "present", certainty=c)]
        policy = self._policy(gate_mode=RetrievalGateMode.JOINT_GATE, weight=0.0)
        _, actions = self._rank(policy, reports)
        verify = [a for a in actions if a.kind is ActionKind.VERIFY]
        self.assertTrue(any(a.retrieval_triggered for a in verify))  # gate opened
        self.assertEqual(actions[0].kind, ActionKind.NEW)  # ...but AskNew wins

    # --- 7. no_rag safely degenerates to the old strategy ------------------ #
    def test_no_rag_degenerates_to_baseline(self):
        c = CertaintyCue.CERTAIN
        reports = [
            Observation(FeatureKey("runny_nose"), "present", certainty=c),
            Observation(FeatureKey("itchy_eyes"), "present", certainty=c),
            Observation(FeatureKey("fever"), "present", certainty=c),
        ]
        joint_no_retriever = ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(
                retrieval_mode=RetrievalMode.NO_RAG,
                retrieval_gate_mode=RetrievalGateMode.JOINT_GATE,
                retrieval_impact_weight=2.0,
            ),
            retriever=None,
        )
        baseline = ReliabilityAwareActionPolicy()
        _, joint_actions = self._rank(joint_no_retriever, reports)
        _, base_actions = self._rank(baseline, reports)
        self.assertEqual(
            [a.utility for a in joint_actions],
            [a.utility for a in base_actions],
        )
        self.assertFalse([a for a in joint_actions if a.kind is ActionKind.VERIFY])

    # --- 8. static_rag retrieves only the presenting complaint -------------- #
    def test_static_rag_does_not_refresh_retrieval_mid_dialogue(self):
        c = CertaintyCue.CERTAIN
        case = next(case for case in self.cases if case.diagnosis == "influenza")

        def run(mode):
            patient = StructuredPatientSimulator(
                diagnosis=case.diagnosis,
                latent_states=case.states,
                model=self.model,
                profile=PatientProfile.from_noise_rate(0.3),
                seed=9,
            )
            policy = self._policy(
                gate_mode=RetrievalGateMode.JOINT_GATE, weight=2.0,
                retrieval_mode=mode,
            )
            return run_reliability_aware_dialogue(
                patient, policy=policy,
                initial_observations=(
                    Observation(FeatureKey("fever"), "present", certainty=c),
                ),
            )

        static = run(RetrievalMode.STATIC_RAG)
        dynamic = run(RetrievalMode.DYNAMIC_RAG)
        # static retrieves exactly once, from the presenting complaint only
        self.assertEqual(len(static.retrieval_log), 1)
        self.assertEqual(static.retrieval_log[0]["mode"], "static_rag")
        self.assertEqual(static.retrieval_log[0]["turn"], 0)
        self.assertIn("fever", static.retrieval_log[0]["query"])
        # dynamic refreshes the query across the dialogue
        self.assertGreater(len(dynamic.retrieval_log), 1)

    # --- 9. report risks must align one-to-one with reports ---------------- #
    def test_retrieval_gate_rejects_misaligned_report_risks(self):
        tracker = BeliefTracker(self.model)
        policy = ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(
                retrieval_mode=RetrievalMode.DYNAMIC_RAG,
                retrieval_gate_mode=RetrievalGateMode.JOINT_GATE,
            ),
            verification_risk_gate=_constant_learned_gate(),
            retriever=self.retriever,
        )
        report = Observation(
            FeatureKey("fever"), "present", certainty=CertaintyCue.CERTAIN
        )
        tracker.update(report)
        with self.assertRaisesRegex(ValueError, "align one-to-one"):
            policy.rank_actions(
                tracker,
                initial_observations=(),
                reports=(report,),
                asked={report.key},
                verified_report_indices=set(),
                verification_count=0,
                report_risks=(),
            )

    # --- 10. the prediction path reads no latent state --------------------- #
    def test_prediction_path_reads_no_latent_state(self):
        c = CertaintyCue.CERTAIN
        reports = [
            Observation(FeatureKey("runny_nose"), "present", certainty=c),
            Observation(FeatureKey("itchy_eyes"), "present", certainty=c),
            Observation(FeatureKey("fever"), "present", certainty=c),
        ]
        policy = self._policy(gate_mode=RetrievalGateMode.JOINT_GATE, weight=2.0)
        # rank_actions is called with no oracle state and still opens the gate,
        # proving the activation value never reads latent/trueness/noise labels.
        _, actions = self._rank(policy, reports)
        verify = [a for a in actions if a.kind is ActionKind.VERIFY]
        self.assertTrue(any(a.retrieval_triggered for a in verify))
        # the per-candidate log carries exactly the label-free fields
        self.assertTrue(policy.last_verification_log)
        for entry in policy.last_verification_log:
            self.assertEqual(set(entry.keys()), _EXPECTED_LOG_FIELDS)
        # the activation value is a pure function of its two arguments
        self.assertEqual(
            policy._retrieval_activation_value(0.5, 0.4),
            0.5 * 0.4 - policy.config.verification_cost,
        )


if __name__ == "__main__":
    unittest.main()
