"""Phase 7 (Prompt #16) -- multi-step verification-value audit tests.

These 14 tests pin the *label-generation* machinery of ``multistep_value``:
reconstruction, forced first action, heuristic continuation, common random
numbers, real termination, the terminal-loss formula, per-question cost, the
Q/V sign convention, and the red lines (no true state / no oracle label in the
prediction path, dynamic RAG still refreshes, reproducibility, default policy
unchanged).

The module only *generates and audits* rollout labels; it never retrains a
model or wires anything into the online policy.  These tests assert exactly
that separation.
"""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from powerful_medrag import multistep_value as mv
from powerful_medrag.belief import BeliefTracker
from powerful_medrag.channel import AnswerChannel, ReportMode
from powerful_medrag.decision import (
    ActionKind,
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    RetrievalMode,
)
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.retrieval import MedicalRetriever
from powerful_medrag.schema import UNKNOWN, CertaintyCue, FeatureKey, Observation
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator

SRC = Path("src/powerful_medrag")


class ToyEnv:
    def __init__(self):
        cases, specs = generate_toy_cases(cases_per_disease=30, seed=41)
        self.model = DiseaseStateModel.fit(cases, specs)
        self.cases = cases

    @staticmethod
    def obs(name, value="present", certainty=CertaintyCue.CERTAIN):
        return Observation(FeatureKey(name), value, certainty)


class _FakePatient:
    """Deterministic patient stub: every asked key answers ``present``/CERTAIN
    (unless overridden), so no randomness enters the continuation under test."""

    def __init__(self, model, answers=None):
        self.model = model
        self.answers = answers or {}

    def answer(self, key):
        if key in self.answers:
            return self.answers[key], ReportMode.CERTAIN
        return (
            Observation(key=key, value="present", certainty=CertaintyCue.CERTAIN),
            ReportMode.CERTAIN,
        )


class _GuardPatient(_FakePatient):
    """Raises if the policy path touches privileged simulator state."""

    @property
    def latent_states(self):
        raise AssertionError("policy path read latent_states")

    @property
    def diagnosis(self):
        raise AssertionError("policy path read diagnosis")


class _RecordingPolicy:
    """Wraps a real policy and records every decision call's keyword args and
    result, so tests can assert the oracle stays off and actions are counted."""

    def __init__(self, inner):
        self.inner = inner
        self.choose_calls = []
        self.choose_results = []
        self.rank_calls = []

    @property
    def config(self):
        return self.inner.config

    def choose_action(self, tracker, **kwargs):
        self.choose_calls.append(dict(kwargs))
        action = self.inner.choose_action(tracker, **kwargs)
        self.choose_results.append(action)
        return action

    def rank_actions(self, tracker, **kwargs):
        self.rank_calls.append(dict(kwargs))
        return self.inner.rank_actions(tracker, **kwargs)


class _RecordingRetriever:
    def __init__(self, inner):
        self.inner = inner
        self.queries = []

    def retrieve(self, query, k=10, use_cache=True):
        self.queries.append(query)
        return self.inner.retrieve(query, k=k, use_cache=use_cache)


def _write_corpus(tmp: str) -> str:
    snippets = [
        ("s0", "influenza", "fever and dry cough suggest influenza"),
        ("s1", "allergy", "runny nose and itchy eyes suggest allergic rhinitis"),
        ("s2", "cold", "sore throat and myalgia suggest common cold"),
    ]
    path = Path(tmp) / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for sid, title, content in snippets:
            fh.write(
                json.dumps(
                    {"id": sid, "title": title, "content": content,
                     "contents": content}
                )
                + "\n"
            )
    return tmp


def _no_rag_config(**overrides):
    kwargs = dict(retrieval_mode=RetrievalMode.NO_RAG, max_total_turns=8)
    kwargs.update(overrides)
    return ReliabilityAwarePolicyConfig(**kwargs)


def _frozen(reports=(), asked=(), verification_count=0, verified=()):
    return mv.FrozenState(
        initial_observations=(),
        reports=tuple(reports),
        asked=frozenset(asked),
        verified=frozenset(verified),
        verification_count=verification_count,
    )


class TestReconstruction(unittest.TestCase):
    def test_01_all_branches_reconstruct_same_start_state(self):
        env = ToyEnv()
        channel = AnswerChannel()
        reports = (env.obs("fever"), env.obs("myalgia"))
        direct = BeliefTracker(env.model, channel)
        for obs in reports:
            direct.update(obs)
        recon = mv.reconstruct_tracker(env.model, channel, (), reports)
        recon_again = mv.reconstruct_tracker(env.model, channel, (), reports)
        self.assertEqual(recon.belief, direct.belief)
        # Deterministic: every branch (AskNew / VerifyOld) rebuilds the *same*
        # starting belief from the same frozen state.
        self.assertEqual(recon.belief, recon_again.belief)


class TestFirstAction(unittest.TestCase):
    def test_02_first_action_new_is_forced(self):
        env = ToyEnv()
        policy = _RecordingPolicy(
            ReliabilityAwareActionPolicy(config=_no_rag_config())
        )
        fake = _FakePatient(env.model)
        mv.run_continuation(
            fake, policy,
            initial_observations=(), reports=(), asked=frozenset(),
            verified=frozenset(), verification_count=0,
            first_kind="new", first_key=FeatureKey("fever"), first_report_index=None,
            channel=AnswerChannel(), max_total_turns=8,
        )
        # The first heuristic decision is made *after* the forced question, so
        # its state already contains the forced key.
        self.assertIn(FeatureKey("fever"), policy.choose_calls[0]["asked"])
        self.assertEqual(len(policy.choose_calls[0]["reports"]), 1)

    def test_03_first_action_verify_is_forced(self):
        env = ToyEnv()
        policy = _RecordingPolicy(
            ReliabilityAwareActionPolicy(config=_no_rag_config())
        )
        fake = _FakePatient(env.model)
        reports = (env.obs("fever"), env.obs("myalgia"))
        mv.run_continuation(
            fake, policy,
            initial_observations=(), reports=reports,
            asked=frozenset({FeatureKey("fever"), FeatureKey("myalgia")}),
            verified=frozenset(), verification_count=0,
            first_kind="verify", first_key=None, first_report_index=0,
            channel=AnswerChannel(), max_total_turns=8,
        )
        call = policy.choose_calls[0]
        self.assertIn(0, call["verified_report_indices"])
        self.assertEqual(call["verification_count"], 1)


class TestContinuationHeuristic(unittest.TestCase):
    def test_04_continuation_uses_heuristic_not_oracle(self):
        env = ToyEnv()
        inner = ReliabilityAwareActionPolicy(config=_no_rag_config())
        self.assertFalse(inner.oracle_selection)
        self.assertFalse(inner.oracle_correction)
        policy = _RecordingPolicy(inner)
        fake = _FakePatient(env.model)
        mv.run_continuation(
            fake, policy,
            initial_observations=(), reports=(env.obs("fever"),),
            asked=frozenset({FeatureKey("fever")}), verified=frozenset(),
            verification_count=0,
            first_kind="new", first_key=FeatureKey("dry_cough"),
            first_report_index=None,
            channel=AnswerChannel(), max_total_turns=8,
        )
        self.assertGreater(len(policy.choose_calls), 0)
        for call in policy.choose_calls:
            self.assertIsNone(call["oracle_states"])
            self.assertEqual(call["report_risks"], ())


class TestCommonRandomNumbers(unittest.TestCase):
    def test_05_common_random_numbers_across_actions(self):
        env = ToyEnv()
        case = next(c for c in env.cases if c.diagnosis == "influenza")
        frozen = _frozen(
            reports=(env.obs("fever"), env.obs("myalgia")),
            asked=(FeatureKey("fever"), FeatureKey("myalgia")),
        )
        profile = PatientProfile.from_noise_rate(0.3)
        channel = AnswerChannel()
        policy = ReliabilityAwareActionPolicy(config=_no_rag_config())
        seeds = []
        orig = mv.StructuredPatientSimulator

        class _Rec(orig):
            def __init__(self, **kw):
                seeds.append(kw["seed"])
                super().__init__(**kw)

        def run(kind, key, report_index):
            return mv.multistep_rollout(
                case, env.model, policy=policy, profile=profile, channel=channel,
                frozen=frozen, first_kind=kind, first_key=key,
                first_report_index=report_index,
                base_seed=4041, noise=0.3, state_index=0, n_rollouts=4,
                max_total_turns=8,
            )

        with mock.patch.object(mv, "StructuredPatientSimulator", _Rec):
            run("new", FeatureKey("dry_cough"), None)
            new_seeds = list(seeds)
            seeds.clear()
            run("verify", None, 0)
            verify_seeds = list(seeds)
        # The per-rollout patient seed depends only on (base_seed, case, noise,
        # state_index, rollout_index) -- never on the candidate action -- so
        # AskNew and VerifyOld share the same random stream (variance reduction).
        self.assertEqual(len(new_seeds), 4)
        self.assertEqual(new_seeds, verify_seeds)


class TestTermination(unittest.TestCase):
    def test_06_rollout_runs_to_real_termination(self):
        env = ToyEnv()
        case = next(c for c in env.cases if c.diagnosis == "influenza")
        patient = StructuredPatientSimulator(
            diagnosis=case.diagnosis, latent_states=case.states, model=env.model,
            profile=PatientProfile.from_noise_rate(0.3), seed=123,
        )
        policy = ReliabilityAwareActionPolicy(config=_no_rag_config())
        outcome = mv.run_continuation(
            patient, policy,
            initial_observations=(), reports=(), asked=frozenset(),
            verified=frozenset(), verification_count=0,
            first_kind="new", first_key=FeatureKey("fever"), first_report_index=None,
            channel=AnswerChannel(), max_total_turns=8,
        )
        # The continuation always executes the forced question and then runs the
        # heuristic until Stop or the remaining budget -- never overshooting.
        self.assertGreaterEqual(outcome.future_questions, 1)
        self.assertLessEqual(outcome.future_questions, 8)
        self.assertAlmostEqual(sum(outcome.terminal_belief.values()), 1.0, places=9)
        recognized = {
            "total_turn_budget",
            "confidence, marginal utility, and report-risk checks passed",
            "no positive-utility acquisition action or a reliability/safety check remains; "
            "return uncertainty",
        }
        self.assertIn(outcome.stop_reason, recognized)


class TestTerminalLoss(unittest.TestCase):
    def test_07_terminal_objective_formula(self):
        self.assertEqual(mv.terminal_objective(0.5, 5, c_q=0.03), 0.5 + 0.03 * 5)
        self.assertEqual(mv.terminal_objective(0.0, 0, c_q=mv.C_QUESTION), 0.0)

    def test_08_cost_counted_once_per_atomic_question(self):
        env = ToyEnv()
        case = next(c for c in env.cases if c.diagnosis == "influenza")
        patient = StructuredPatientSimulator(
            diagnosis=case.diagnosis, latent_states=case.states, model=env.model,
            profile=PatientProfile.from_noise_rate(0.3), seed=7,
        )
        policy = _RecordingPolicy(ReliabilityAwareActionPolicy(config=_no_rag_config()))
        frozen = _frozen(
            reports=(env.obs("fever"), env.obs("myalgia")),
            asked=(FeatureKey("fever"), FeatureKey("myalgia")),
        )
        outcome = mv.run_continuation(
            patient, policy,
            initial_observations=frozen.initial_observations,
            reports=frozen.reports, asked=frozen.asked,
            verified=frozen.verified, verification_count=frozen.verification_count,
            first_kind="verify", first_key=None, first_report_index=0,
            channel=AnswerChannel(), max_total_turns=8,
        )
        # future_questions = 1 forced action + every non-Stop heuristic action.
        non_stop = [a for a in policy.choose_results if a.kind is not ActionKind.STOP]
        self.assertEqual(outcome.future_questions, 1 + len(non_stop))
        # Each action (AskNew or VerifyOld) therefore contributes exactly one
        # atomic question to the c_q * N_future cost term.
        j = mv.terminal_objective(0.5, outcome.future_questions, c_q=mv.C_QUESTION)
        self.assertEqual(j, 0.5 + mv.C_QUESTION * outcome.future_questions)


class TestQMultiVMulti(unittest.TestCase):
    def test_09_q_multi_and_v_multi_sign(self):
        env = ToyEnv()
        case = next(c for c in env.cases if c.diagnosis == "influenza")
        frozen = _frozen(
            reports=(env.obs("fever"), env.obs("myalgia")),
            asked=(FeatureKey("fever"), FeatureKey("myalgia")),
        )
        profile = PatientProfile.from_noise_rate(0.3)
        channel = AnswerChannel()
        policy = ReliabilityAwareActionPolicy(config=_no_rag_config())
        common = dict(
            model=env.model, policy=policy, profile=profile, channel=channel,
            frozen=frozen, base_seed=4041, noise=0.3, state_index=0,
            n_rollouts=4, max_total_turns=8,
        )
        r_new = mv.multistep_rollout(
            case, first_kind="new", first_key=FeatureKey("dry_cough"),
            first_report_index=None, **common,
        )
        r_verify = mv.multistep_rollout(
            case, first_kind="verify", first_key=None, first_report_index=0,
            **common,
        )
        self.assertAlmostEqual(r_new.q_multi, -r_new.j_value, places=12)
        self.assertAlmostEqual(r_verify.q_multi, -r_verify.j_value, places=12)
        # V_multi(0) = J(AskNew_best) - J(VerifyOld(0)) = -Q(new) + Q(verify).
        v_multi = r_new.j_value - r_verify.j_value
        self.assertAlmostEqual(v_multi, -r_new.q_multi + r_verify.q_multi, places=12)


class TestNoTrueState(unittest.TestCase):
    def test_10_deployable_policy_reads_no_true_state(self):
        env = ToyEnv()
        policy = _RecordingPolicy(ReliabilityAwareActionPolicy(config=_no_rag_config()))
        guard = _GuardPatient(env.model)
        # probe exercises rank_actions (snapshot) + choose_action (decision)
        mv.run_heuristic_probe(
            guard, policy, initial_observations=(), channel=AnswerChannel(),
            max_total_turns=8,
        )
        for call in policy.rank_calls + policy.choose_calls:
            self.assertIsNone(call["oracle_states"])
            self.assertEqual(call["report_risks"], ())
        src = (SRC / "multistep_value.py").read_text(encoding="utf-8")
        self.assertNotIn("oracle_selection", src)
        self.assertNotIn("oracle_correction", src)

    def test_11_oracle_labels_do_not_feed_policy(self):
        # run_continuation exposes no oracle / true-disease / latent-state input.
        params = inspect.signature(mv.run_continuation).parameters
        for banned in (
            "oracle_states", "report_risks", "diagnosis", "latent_states",
            "oracle_correction", "true_diagnosis",
        ):
            self.assertNotIn(banned, params)
        # multistep_rollout never smuggles the true disease into the continuation.
        env = ToyEnv()
        case = next(c for c in env.cases if c.diagnosis == "influenza")
        frozen = _frozen(reports=(env.obs("fever"),), asked=(FeatureKey("fever"),))
        profile = PatientProfile.from_noise_rate(0.3)
        policy = ReliabilityAwareActionPolicy(config=_no_rag_config())
        with mock.patch.object(mv, "run_continuation", wraps=mv.run_continuation) as rc:
            mv.multistep_rollout(
                case, env.model, policy=policy, profile=profile,
                channel=AnswerChannel(), frozen=frozen,
                first_kind="new", first_key=FeatureKey("dry_cough"),
                first_report_index=None, base_seed=4041, noise=0.3,
                state_index=0, n_rollouts=2, max_total_turns=8,
            )
        self.assertEqual(rc.call_count, 2)
        for banned in ("oracle_states", "report_risks", "diagnosis", "latent_states"):
            self.assertNotIn(banned, rc.call_args.kwargs)


class TestDynamicRag(unittest.TestCase):
    def test_12_dynamic_rag_updates_each_turn(self):
        env = ToyEnv()
        with tempfile.TemporaryDirectory() as tmp:
            retriever = _RecordingRetriever(
                MedicalRetriever(_write_corpus(tmp), k1=1.5, b=0.75)
            )
            policy = ReliabilityAwareActionPolicy(
                config=ReliabilityAwarePolicyConfig(
                    retrieval_mode=RetrievalMode.DYNAMIC_RAG,
                    retrieval_impact_weight=1.0,
                    max_total_turns=6,
                ),
                retriever=retriever,
            )
            # An UNKNOWN report opens the verification gate, so retrieval runs.
            frozen = _frozen(
                reports=(Observation(FeatureKey("fever"), UNKNOWN, CertaintyCue.NONE),),
                asked=(FeatureKey("fever"),),
            )
            fake = _FakePatient(env.model, {
                FeatureKey("dry_cough"): Observation(
                    FeatureKey("dry_cough"), "present", CertaintyCue.CERTAIN
                ),
            })
            mv.run_continuation(
                fake, policy,
                initial_observations=frozen.initial_observations,
                reports=frozen.reports, asked=frozen.asked,
                verified=frozen.verified, verification_count=frozen.verification_count,
                first_kind="new", first_key=FeatureKey("dry_cough"),
                first_report_index=None,
                channel=AnswerChannel(), max_total_turns=6,
            )
        # Retrieval participated, and the query reflects the evidence added on the
        # current turn (the forced "dry_cough" answer), i.e. it is rebuilt each
        # turn from the latest reports rather than frozen at the initial state.
        self.assertGreaterEqual(len(retriever.queries), 1)
        self.assertTrue(any("dry_cough" in q for q in retriever.queries))


class TestReproducibility(unittest.TestCase):
    def test_13_same_seed_reproducible(self):
        env = ToyEnv()
        case = next(c for c in env.cases if c.diagnosis == "influenza")
        frozen = _frozen(
            reports=(env.obs("fever"), env.obs("myalgia")),
            asked=(FeatureKey("fever"), FeatureKey("myalgia")),
        )
        profile = PatientProfile.from_noise_rate(0.3)
        channel = AnswerChannel()
        policy = ReliabilityAwareActionPolicy(config=_no_rag_config())
        kwargs = dict(
            model=env.model, policy=policy, profile=profile, channel=channel,
            frozen=frozen, first_kind="new", first_key=FeatureKey("dry_cough"),
            first_report_index=None, base_seed=4041, noise=0.3, state_index=0,
            n_rollouts=4, max_total_turns=8,
        )
        r1 = mv.multistep_rollout(case, **kwargs)
        r2 = mv.multistep_rollout(case, **kwargs)
        self.assertEqual(r1.j_value, r2.j_value)
        self.assertEqual(r1.terminal_brier, r2.terminal_brier)
        self.assertEqual(r1.terminal_brier_list, r2.terminal_brier_list)
        self.assertEqual(r1.future_questions_list, r2.future_questions_list)


class TestDefaultPolicyUnchanged(unittest.TestCase):
    def test_14_default_policy_unchanged(self):
        # decision.py must not import the audit module (default policy intact).
        decision_src = (SRC / "decision.py").read_text(encoding="utf-8")
        self.assertNotIn("multistep_value", decision_src)
        cfg = ReliabilityAwarePolicyConfig()
        self.assertIs(cfg.retrieval_mode, RetrievalMode.NO_RAG)
        self.assertEqual(cfg.maximum_verifications, 1)
        self.assertEqual(cfg.max_total_turns, 15)
        policy = ReliabilityAwareActionPolicy()
        self.assertFalse(policy.oracle_selection)
        self.assertFalse(policy.oracle_correction)
        self.assertIsNone(policy.verification_risk_gate)


if __name__ == "__main__":
    unittest.main()
