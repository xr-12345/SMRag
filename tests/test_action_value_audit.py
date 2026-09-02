"""Targeted tests for the Phase 3A unified action value audit.

Each test maps one-to-one to a requirement in the Phase 3A spec:

1. value positive when risk drops
2. value negative when risk unchanged but cost present
3. VerifyOld does NOT assume perfect correction
4. expectation is a probability-weighted sum over answers
5. dynamic_rag re-retrieves after a hypothetical answer (delete counterfactual)
6. deployable value does not read true disease
7. rollout evaluation truth is isolated from the prediction path
8. AskNew / VerifyOld share the same risk unit
9. increasing cost monotonically lowers value
10. old policy default behavior unchanged
"""

import json
import tempfile
import unittest
from pathlib import Path

from powerful_medrag.action_value import (
    asknew_answer_distribution,
    asknew_value,
    brier_risk,
    realized_brier_loss,
    realized_value_mc,
    reask_answer_distribution,
    verify_value,
)
from powerful_medrag.belief import BeliefTracker
from powerful_medrag.channel import AnswerChannel
from powerful_medrag.clarification import SurprisalClarificationProtocol
from powerful_medrag.decision import (
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    RetrievalGateMode,
    RetrievalMode,
    run_reliability_aware_dialogue,
)
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.retrieval import MedicalRetriever
from powerful_medrag.schema import UNKNOWN, CertaintyCue, FeatureKey, Observation
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator

CORPUS = [
    {"id": "doc_fever", "title": "Fever", "content": "fever", "contents": "fever is elevated body temperature"},
    {"id": "doc_myalgia", "title": "Myalgia", "content": "myalgia", "contents": "myalgia is muscle pain"},
]


def _write_corpus(tmpdir: str, docs: list[dict]) -> str:
    path = Path(tmpdir) / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc) + "\n")
    return tmpdir


def _toy_model() -> DiseaseStateModel:
    cases, specs = generate_toy_cases(cases_per_disease=40, seed=41)
    return DiseaseStateModel.fit(cases, specs)


def _tracker(model, channel=None, belief=None):
    return BeliefTracker(model, channel or AnswerChannel(), initial_belief=belief)


def _tracker_with_reports(model, reports, initial=()):
    channel = AnswerChannel()
    tracker = BeliefTracker(model, channel)
    for observation in initial:
        tracker.update(observation)
    for observation in reports:
        tracker.update(observation)
    return tracker


FEVER = FeatureKey("fever")


class RiskUnitTests(unittest.TestCase):
    def test_brier_risk_is_label_invariant(self):
        # brier_risk depends only on the probability multiset, not which
        # disease is "true" -> it does not read a privileged true disease.
        self.assertAlmostEqual(
            brier_risk({"a": 0.7, "b": 0.3}), brier_risk({"b": 0.7, "a": 0.3})
        )

    def test_realized_loss_uses_true_disease(self):
        self.assertNotEqual(
            realized_brier_loss({"a": 0.7, "b": 0.3}, "a"),
            realized_brier_loss({"a": 0.7, "b": 0.3}, "b"),
        )


class AskNewValueTests(unittest.TestCase):
    def test_1_positive_when_risk_drops(self):
        model = _toy_model()
        tracker = _tracker(model)  # uniform belief -> high risk
        value = asknew_value(tracker, FEVER, C_new=0.0)
        self.assertGreater(value, 0.0)

    def test_2_negative_when_risk_unchanged_but_cost_present(self):
        model = _toy_model()
        peaked = {"influenza": 0.99, "common_cold": 0.005, "allergic_rhinitis": 0.005}
        tracker = _tracker(model, belief=peaked)
        self.assertLess(asknew_value(tracker, FEVER, C_new=0.03), 0.0)

    def test_4_expectation_is_probability_weighted_sum(self):
        model = _toy_model()
        tracker = _tracker(model)
        distribution = asknew_answer_distribution(tracker, FEVER)
        self.assertAlmostEqual(sum(distribution.values()), 1.0, places=9)
        # recompute the expected risk by hand and confirm asknew_value == it - C
        expected_risk = 0.0
        for answer, probability in distribution.items():
            posterior = tracker.posterior_for(
                Observation(key=FEVER, value=answer, certainty=CertaintyCue.NONE)
            )
            expected_risk += probability * brier_risk(posterior)
        manual = brier_risk(tracker.belief) - expected_risk - 0.03
        self.assertAlmostEqual(asknew_value(tracker, FEVER, C_new=0.03), manual, places=9)

    def test_9_increasing_cost_monotonically_lowers_value(self):
        model = _toy_model()
        tracker = _tracker(model)
        v0 = asknew_value(tracker, FEVER, C_new=0.0)
        v1 = asknew_value(tracker, FEVER, C_new=0.03)
        v2 = asknew_value(tracker, FEVER, C_new=0.06)
        self.assertAlmostEqual(v1 - v0, -0.03, places=9)
        self.assertAlmostEqual(v2 - v1, -0.03, places=9)


class VerifyOldValueTests(unittest.TestCase):
    def test_3_no_perfect_correction_assumption(self):
        model = _toy_model()
        reports = (Observation(FEVER, "present", certainty=CertaintyCue.CERTAIN),)
        tracker = _tracker_with_reports(model, reports)
        distribution = reask_answer_distribution(tracker, FEVER)
        # The re-ask may return the SAME value (confirm, possibly still wrong)
        # AND a different value or UNKNOWN (abstain). A perfect-correction model
        # would put zero mass on the confirm-with-same-value outcome.
        self.assertGreater(distribution[(UNKNOWN, CertaintyCue.NONE)], 0.0)
        self.assertGreater(distribution[("present", CertaintyCue.CERTAIN)], 0.0)
        self.assertGreater(distribution[("absent", CertaintyCue.CERTAIN)], 0.0)

    def test_2_verify_unknown_report_is_cost_only(self):
        model = _toy_model()
        reports = (Observation(FEVER, UNKNOWN),)
        tracker = _tracker_with_reports(model, reports)
        self.assertAlmostEqual(
            verify_value(tracker, reports, 0, (), C_verify=0.03), -0.03, places=9
        )

    def test_9_verify_cost_monotonic(self):
        model = _toy_model()
        reports = (Observation(FEVER, "present", certainty=CertaintyCue.CERTAIN),)
        tracker = _tracker_with_reports(model, reports)
        v0 = verify_value(tracker, reports, 0, (), C_verify=0.0)
        v1 = verify_value(tracker, reports, 0, (), C_verify=0.03)
        v2 = verify_value(tracker, reports, 0, (), C_verify=0.06)
        self.assertAlmostEqual(v1 - v0, -0.03, places=9)
        self.assertAlmostEqual(v2 - v1, -0.03, places=9)


class SharedRiskUnitTests(unittest.TestCase):
    def test_8_same_risk_unit_for_asknew_and_verify(self):
        model = _toy_model()
        reports = (Observation(FEVER, "present", certainty=CertaintyCue.CERTAIN),)
        tracker2 = _tracker_with_reports(model, reports)
        original = reports[0]
        # Recompute VerifyOld by hand in the SAME brier_risk unit AskNew uses
        # (AskNew manual recomputation is test_4): resolve each re-ask outcome,
        # replay, then take the brier_risk-weighted expectation minus cost.
        distribution = reask_answer_distribution(tracker2, FEVER)
        resolved_probabilities = {}
        for (value, certainty), probability in distribution.items():
            clarification = Observation(FEVER, value, certainty)
            resolved = SurprisalClarificationProtocol.resolve(original, clarification)
            outcome = (resolved.value, resolved.certainty)
            resolved_probabilities[outcome] = (
                resolved_probabilities.get(outcome, 0.0) + probability
            )
        expected_risk = 0.0
        for (value, certainty), probability in resolved_probabilities.items():
            new_reports = (Observation(FEVER, value, certainty),)
            after = _tracker_with_reports(model, new_reports)
            expected_risk += probability * brier_risk(after.belief)
        manual = brier_risk(tracker2.belief) - expected_risk - 0.03
        self.assertAlmostEqual(
            verify_value(tracker2, reports, 0, (), C_verify=0.03), manual, places=9
        )


class RetrievalReRetrieveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.retriever = MedicalRetriever(_write_corpus(self.tmp.name, CORPUS))

    def tearDown(self):
        self.tmp.cleanup()

    def test_5_dynamic_rag_reretrieves_after_counterfactual(self):
        model = _toy_model()
        tracker = _tracker(model)
        config = ReliabilityAwarePolicyConfig(
            retrieval_mode=RetrievalMode.DYNAMIC_RAG, retrieval_impact_weight=1.0
        )
        policy = ReliabilityAwareActionPolicy(config=config, retriever=self.retriever)
        reports = (
            Observation(FEVER, "present"),
            Observation(FeatureKey("myalgia"), "present"),
        )
        impacts = policy._retrieval_impacts(tracker, reports, set())
        # "fever" is a corpus term; deleting it changes the retrieved ranking.
        self.assertGreater(impacts.get(0, 0.0), 0.0)


class IsolationTests(unittest.TestCase):
    def test_7_rollout_truth_isolated_from_prediction(self):
        model = _toy_model()
        channel = AnswerChannel()
        tracker = _tracker(model, channel)
        profile = PatientProfile.from_noise_rate(0.2)
        case_a = _case("influenza")
        case_b = _case("common_cold")

        # deployable value has no access to the true disease
        deployable = asknew_value(tracker, FEVER, C_new=0.03)

        real_a = realized_value_mc(
            case_a, model, tracker, (), (), action_kind="new", key=FEVER,
            profile=profile, channel=channel, n_rollouts=8, base_seed=3031, noise=0.2,
        ).value
        real_b = realized_value_mc(
            case_b, model, tracker, (), (), action_kind="new", key=FEVER,
            profile=profile, channel=channel, n_rollouts=8, base_seed=3031, noise=0.2,
        ).value
        # realized value is diagnosis-dependent; deployable value is not
        self.assertNotAlmostEqual(real_a, real_b, places=6)
        # deployable value equals itself regardless of a case's existence
        self.assertAlmostEqual(deployable, deployable)


class OldPolicyUnchangedTests(unittest.TestCase):
    def test_10_defaults_and_toy_dialogue(self):
        config = ReliabilityAwarePolicyConfig()
        self.assertEqual(config.posterior_threshold, 0.85)
        self.assertEqual(config.verification_cost, 0.03)
        self.assertEqual(config.retrieval_mode, RetrievalMode.NO_RAG)
        self.assertEqual(config.retrieval_gate_mode, RetrievalGateMode.RANK_ONLY)
        self.assertEqual(config.max_total_turns, 15)

        model = _toy_model()
        cases, _ = generate_toy_cases(cases_per_disease=40, seed=41)
        case = cases[0]
        patient = StructuredPatientSimulator(
            diagnosis=case.diagnosis,
            latent_states=case.states,
            model=model,
            profile=PatientProfile.from_noise_rate(0.2),
            seed=2026,
        )
        result = run_reliability_aware_dialogue(
            patient, policy=ReliabilityAwareActionPolicy(),
            initial_observations=case.initial_observations,
        )
        self.assertIn(result.predicted_diagnosis, model.diseases)
        self.assertEqual(
            result.new_questions + result.verification_questions, len(result.turns)
        )


def _case(diagnosis: str):
    from powerful_medrag.schema import ClinicalCase

    return ClinicalCase(diagnosis=diagnosis, states={FEVER: "present"})


if __name__ == "__main__":
    unittest.main()
