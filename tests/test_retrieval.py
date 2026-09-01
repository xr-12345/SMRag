"""Targeted tests for minimal BM25 medical retrieval (VeriMedRAG phase 1).

Covers the two behaviours the phase must guarantee:

1. Patient-answer changes (flip / delete / weaken) actually change the query and
   the retrieved snippet ranking — requirement 9.
2. Retrieval impact enters the VerifyOld ranking in ``decision.py`` — requirement 7.

A tiny in-memory corpus (a few ``.jsonl`` snippets written to a temp dir) keeps
the tests fast and independent of the 209 MB MedRAG Textbooks download.
"""

import json
import tempfile
import unittest
from pathlib import Path

from powerful_medrag.belief import BeliefTracker
from powerful_medrag.decision import (
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    RetrievalMode,
    run_reliability_aware_dialogue,
)
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.retrieval import (
    RetrievalHit,
    MedicalRetriever,
    apply_counterfactual,
    build_retrieval_query,
    counterfactual_retrieval,
    jaccard,
    ndcg,
    rbo,
)
from powerful_medrag.schema import UNKNOWN, CertaintyCue, FeatureKey, Observation
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator

# Distinct single-token terms so a flip/delete visibly reorders the ranking.
UNIT_CORPUS = [
    {"id": "doc_fever", "title": "Fever", "content": "fever", "contents": "fever is elevated body temperature"},
    {"id": "doc_rash", "title": "Rash", "content": "rash", "contents": "rash is a skin lesion"},
    {"id": "doc_neuro", "title": "Neuro", "content": "headache", "contents": "headache is cranial pain"},
]

# Toy-model features (fever, myalgia) with corpus support; sore_throat has none.
INTEG_CORPUS = [
    {"id": "doc_fever", "title": "Fever", "content": "fever", "contents": "fever is elevated body temperature"},
    {"id": "doc_myalgia", "title": "Myalgia", "content": "myalgia", "contents": "myalgia is muscle pain"},
]


def _write_corpus(tmpdir: str, docs: list[dict]) -> str:
    path = Path(tmpdir) / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc) + "\n")
    return tmpdir


def _toy_model():
    cases, specs = generate_toy_cases(cases_per_disease=30, seed=41)
    return DiseaseStateModel.fit(cases, specs), cases


class RetrievalQueryAndCounterfactualTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.retriever = MedicalRetriever(_write_corpus(self.tmp.name, UNIT_CORPUS))

    def tearDown(self):
        self.tmp.cleanup()

    def _reports(self):
        return [
            Observation(FeatureKey("fever"), "present"),
            Observation(FeatureKey("rash"), "present"),
        ]

    def test_present_symptom_enters_query_absent_does_not(self):
        reports = [
            Observation(FeatureKey("fever"), "present"),
            Observation(FeatureKey("rash"), "absent"),
        ]
        query = build_retrieval_query(reports, [("influenza", 0.6)], top_k=5)
        self.assertIn("fever", query.evidence_terms)
        self.assertNotIn("rash", query.evidence_terms)

    def test_flip_changes_query_and_retrieval(self):
        reports = self._reports()
        ranked = [("influenza", 0.6)]
        before = build_retrieval_query(reports, ranked, top_k=5)
        flipped = apply_counterfactual(reports, 0, "flip")  # fever present -> absent
        after = build_retrieval_query(flipped, ranked, top_k=5)
        self.assertNotEqual(before.text, after.text)
        self.assertIn("fever", before.evidence_terms)
        self.assertNotIn("fever", after.evidence_terms)
        result = counterfactual_retrieval(self.retriever, reports, ranked, 0, "flip")
        self.assertLess(result["change"]["topk_jaccard"], 1.0)

    def test_feature_names_mapping_replaces_evidence_codes(self):
        reports = [Observation(FeatureKey("E_146"), "present")]
        mapping = {FeatureKey("E_146"): "Do you have a fever?"}
        query = build_retrieval_query(
            reports, [("influenza", 0.6)], feature_names=mapping
        )
        self.assertIn("Do you have a fever?", query.evidence_terms)
        self.assertNotIn("E_146", query.evidence_terms)

    def test_flip_absent_to_present_adds_term(self):
        reports = [Observation(FeatureKey("fever"), "absent")]
        ranked = [("influenza", 0.6)]
        self.assertNotIn("fever", build_retrieval_query(reports, ranked).evidence_terms)
        flipped = apply_counterfactual(reports, 0, "flip")
        self.assertIn("fever", build_retrieval_query(flipped, ranked).evidence_terms)

    def test_weaken_removes_term(self):
        reports = [Observation(FeatureKey("fever"), "present")]
        weakened = apply_counterfactual(reports, 0, "weaken")
        self.assertEqual(weakened[0].value, UNKNOWN)
        query = build_retrieval_query(weakened, [("influenza", 0.6)])
        self.assertNotIn("fever", query.evidence_terms)

    def test_delete_removes_term_and_report(self):
        reports = self._reports()
        deleted = apply_counterfactual(reports, 0, "delete")
        self.assertEqual(len(deleted), 1)
        query = build_retrieval_query(deleted, [("influenza", 0.6)])
        self.assertNotIn("fever", query.evidence_terms)

    def test_retriever_retrieves_and_caches(self):
        query = "influenza fever"
        first = self.retriever.retrieve(query)
        second = self.retriever.retrieve(query)
        self.assertTrue(first)
        self.assertIs(first, second)  # cache returns the same object


class RetrievalMetricsTests(unittest.TestCase):
    def test_jaccard_rbo_ndcg(self):
        a = [RetrievalHit("x", 2.0), RetrievalHit("y", 1.0)]
        self.assertEqual(jaccard(a, a), 1.0)
        self.assertEqual(rbo(a, a), 1.0)
        self.assertEqual(jaccard(a, [RetrievalHit("z", 1.0)]), 0.0)
        self.assertGreaterEqual(ndcg(a), 0.0)
        self.assertLessEqual(ndcg(a), 1.0)


class RetrievalImpactIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.model, self.cases = _toy_model()
        self.tmp = tempfile.TemporaryDirectory()
        self.retriever = MedicalRetriever(_write_corpus(self.tmp.name, INTEG_CORPUS))

    def tearDown(self):
        self.tmp.cleanup()

    def _tracker_and_reports(self):
        tracker = BeliefTracker(self.model)
        reports = (
            Observation(FeatureKey("fever"), "present", certainty=CertaintyCue.CERTAIN),
            Observation(FeatureKey("sore_throat"), "present", certainty=CertaintyCue.CERTAIN),
        )
        for report in reports:
            tracker.update(report)
        return tracker, reports

    def test_retrieval_impact_enters_verification_scores(self):
        tracker, reports = self._tracker_and_reports()
        rag_policy = ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(
                retrieval_mode=RetrievalMode.DYNAMIC_RAG,
                retrieval_impact_weight=2.0,
            ),
            retriever=self.retriever,
        )
        base_policy = ReliabilityAwareActionPolicy()

        impacts = rag_policy._retrieval_impacts(tracker, reports, set())
        self.assertGreater(impacts[0], 0.0)  # fever matches the corpus
        self.assertEqual(impacts[1], 0.0)  # sore_throat has no corpus snippet

        base = {
            score.report_index: score
            for score in base_policy._verification_scores(
                tracker,
                initial_observations=(),
                reports=reports,
                verified_report_indices=set(),
                report_risks=(),
            )
        }
        rag = {
            score.report_index: score
            for score in rag_policy._verification_scores(
                tracker,
                initial_observations=(),
                reports=reports,
                verified_report_indices=set(),
                report_risks=(),
            )
        }
        # retrieval impact now lives in its own field, not folded into score.score
        self.assertGreater(rag[0].retrieval_impact, 0.0)  # fever matches the corpus
        self.assertEqual(rag[1].retrieval_impact, 0.0)  # sore_throat has no snippet
        self.assertAlmostEqual(rag[0].score, base[0].score)  # score.score unchanged
        self.assertAlmostEqual(rag[1].score, base[1].score)
        # ...and it re-ranks the VerifyOld utility (not just the sort key)
        base_util = {
            i: base_policy._verification_action(s).utility for i, s in base.items()
        }
        rag_util = {
            i: rag_policy._verification_action(s).utility for i, s in rag.items()
        }
        self.assertGreater(rag_util[0], base_util[0])  # fever boosted via utility
        self.assertAlmostEqual(rag_util[1], base_util[1])  # sore_throat unchanged

    def test_dialogue_retrieval_log_follows_mode(self):
        case = next(c for c in self.cases if c.diagnosis == "influenza")
        patient = StructuredPatientSimulator(
            diagnosis=case.diagnosis,
            latent_states=case.states,
            model=self.model,
            profile=PatientProfile.from_noise_rate(0.3),
            seed=9,
        )

        dynamic = ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(
                max_total_turns=4,
                retrieval_mode=RetrievalMode.DYNAMIC_RAG,
            ),
            retriever=self.retriever,
        )
        dynamic_result = run_reliability_aware_dialogue(patient, policy=dynamic)
        self.assertTrue(dynamic_result.retrieval_log)
        self.assertTrue(
            all(entry["mode"] == "dynamic_rag" for entry in dynamic_result.retrieval_log)
        )

        no_rag = ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(max_total_turns=4)
        )
        no_result = run_reliability_aware_dialogue(patient, policy=no_rag)
        self.assertEqual(no_result.retrieval_log, ())

    def test_static_mode_logs_presenting_complaint_once(self):
        case = next(c for c in self.cases if c.diagnosis == "influenza")
        patient = StructuredPatientSimulator(
            diagnosis=case.diagnosis,
            latent_states=case.states,
            model=self.model,
            profile=PatientProfile.from_noise_rate(0.3),
            seed=9,
        )
        static = ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(
                max_total_turns=4,
                retrieval_mode=RetrievalMode.STATIC_RAG,
            ),
            retriever=self.retriever,
        )
        initial = (
            Observation(FeatureKey("fever"), "present", certainty=CertaintyCue.CERTAIN),
        )
        result = run_reliability_aware_dialogue(
            patient, policy=static, initial_observations=initial
        )
        self.assertEqual(len(result.retrieval_log), 1)
        self.assertEqual(result.retrieval_log[0]["mode"], "static_rag")
        self.assertEqual(result.retrieval_log[0]["turn"], 0)


if __name__ == "__main__":
    unittest.main()
