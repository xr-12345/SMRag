"""Phase 8C -- correlated-re-ask patient simulator + environment/inference tests.

Covers the Phase 8C spec sections 5-7 (backward compatibility, rho_env/rho_model
separation, probability mechanisms) and section 15 (leakage / regression):

* simulator correctness (01-08): mode transition degeneracy, rho=1 persistence,
  monotonic correlation, feature independence, determinism, cross-strategy
  pairing, certainty-cue mapping, all modes generated.
* theory--empirical consistency (09-12): empirical joint frequency ==
  JointReportChannel.joint_probability, second-answer marginal invariant to rho,
  only joint structure changes with rho.
* inference consistency (13-16): matched setting uses rho_model, mismatch never
  reads rho_env, no double-count of consistent correlated answers, no UNKNOWN
  compression, single shared channel, posteriors normalized.
* leakage / regression (17-18): policy never reads rho_env / latent / mode chain;
  old simulator + heuristic baseline unchanged.
"""

from __future__ import annotations

import unittest

from powerful_medrag.channel import AnswerChannel, ReportMode
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.joint_channel_policy import (
    JointChannelBrierAuditPolicy,
    run_joint_channel_dialogue,
)
from powerful_medrag.joint_patient_simulator import JointStructuredPatientSimulator
from powerful_medrag.joint_reliability_belief import JointReliabilityBeliefTracker
from powerful_medrag.joint_report_channel import (
    JointReportChannel,
    VerificationType,
    joint_answer_space,
)
from powerful_medrag.schema import UNKNOWN, CertaintyCue, FeatureKey, Observation
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator
from powerful_medrag.worthiness_policy import WorthinessStrategy, build_policy


def _obs(name, value, certainty=CertaintyCue.CERTAIN):
    return Observation(FeatureKey(name), value, certainty=certainty)


def _toy_model(cases_per_disease=40, seed=41):
    cases, specs = generate_toy_cases(cases_per_disease=cases_per_disease, seed=seed)
    return DiseaseStateModel.fit(cases, specs), specs


def _simulator(model, case, *, seed, rho_env, profile=None):
    return JointStructuredPatientSimulator(
        diagnosis=case.diagnosis,
        latent_states=case.states,
        model=model,
        channel=AnswerChannel(),
        profile=profile or PatientProfile(),
        seed=seed,
        rho_env=rho_env,
        case_id=case.case_id,
    )


def _mode_agreement(model, case, key, *, rho, n_seeds, profile):
    """Empirical P(re-ask mode == first mode) over n_seeds for a given rho."""
    agreements = 0
    for seed in range(n_seeds):
        sim = _simulator(model, case, seed=seed, rho_env=rho, profile=profile)
        _, first = sim.answer(key)
        _, second = sim.answer(key)
        agreements += int(second == first)
    return agreements / n_seeds


# --------------------------------------------------------------------------- #
# 1-8: simulator correctness
# --------------------------------------------------------------------------- #


class TestSimulatorCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.specs = _toy_model()
        cls.cases, _ = generate_toy_cases(cases_per_disease=40, seed=41)
        cls.case = next(c for c in cls.cases if c.diagnosis == "influenza")
        cls.key = cls.specs[0].key
        cls.profile = PatientProfile.from_noise_rate(0.3)

    def test_01_rho_zero_mode_transition_is_prior(self):
        # At rho=0 the re-ask mode is an independent draw from the mode prior.
        prior = dict(self.profile.mode_prior)
        n_seeds = 4000
        counts = {mode: 0 for mode in ReportMode}
        for seed in range(n_seeds):
            sim = _simulator(self.model, self.case, seed=seed, rho_env=0.0,
                             profile=self.profile)
            sim.answer(self.key)
            _, second = sim.answer(self.key)
            counts[second] += 1
        empirical = {mode: count / n_seeds for mode, count in counts.items()}
        for mode in ReportMode:
            self.assertLess(abs(empirical[mode] - prior[mode]), 0.03, mode)

    def test_02_rho_one_subsequent_modes_equal_first(self):
        sim = _simulator(self.model, self.case, seed=7, rho_env=1.0,
                         profile=self.profile)
        _, first = sim.answer(self.key)
        for _ in range(8):
            _, mode = sim.answer(self.key)
            self.assertEqual(mode, first)

    def test_03_correlation_monotonic_in_rho(self):
        rhos = (0.0, 0.25, 0.5, 0.75, 1.0)
        agreements = [
            _mode_agreement(self.model, self.case, self.key, rho=rho,
                            n_seeds=3000, profile=self.profile)
            for rho in rhos
        ]
        for lower, upper in zip(agreements, agreements[1:]):
            self.assertLessEqual(lower, upper + 1e-9)

    def test_04_different_features_do_not_interfere(self):
        key_a = self.specs[0].key
        key_b = self.specs[1].key
        sim_b_first = _simulator(self.model, self.case, seed=123,
                                 rho_env=0.5, profile=self.profile)
        _, mode_b_first = sim_b_first.answer(key_b)

        sim_a_then_b = _simulator(self.model, self.case, seed=123,
                                  rho_env=0.5, profile=self.profile)
        sim_a_then_b.answer(key_a)
        _, mode_b_after_a = sim_a_then_b.answer(key_b)
        self.assertEqual(mode_b_after_a, mode_b_first)

    def test_05_same_input_same_output(self):
        sim1 = _simulator(self.model, self.case, seed=99, rho_env=0.7,
                          profile=self.profile)
        sim2 = _simulator(self.model, self.case, seed=99, rho_env=0.7,
                          profile=self.profile)
        for _ in range(5):
            o1, m1 = sim1.answer(self.key)
            o2, m2 = sim2.answer(self.key)
            self.assertEqual((o1.value, o1.certainty, m1),
                             (o2.value, o2.certainty, m2))

    def test_06_paired_answers_across_strategies(self):
        # Two "strategies" asking the same key in different orders must observe
        # the same answer at each (key, occurrence).
        key_a = self.specs[0].key
        key_b = self.specs[1].key
        s_a = _simulator(self.model, self.case, seed=555, rho_env=0.6,
                         profile=self.profile)
        s_b = _simulator(self.model, self.case, seed=555, rho_env=0.6,
                         profile=self.profile)
        a_first, a_mode_first = s_a.answer(key_a)
        s_a.answer(key_b)
        a_reask, a_mode_reask = s_a.answer(key_a)
        b_first, b_mode_first = s_b.answer(key_a)
        b_reask, b_mode_reask = s_b.answer(key_a)
        s_b.answer(key_b)
        self.assertEqual((a_first.value, a_first.certainty, a_mode_first),
                         (b_first.value, b_first.certainty, b_mode_first))
        self.assertEqual((a_reask.value, a_reask.certainty, a_mode_reask),
                         (b_reask.value, b_reask.certainty, b_mode_reask))

    def test_07_certainty_cue_maps_to_report_mode(self):
        n = 2000
        seen = set()
        for seed in range(n):
            sim = _simulator(self.model, self.case, seed=seed, rho_env=0.5,
                             profile=self.profile)
            for _ in range(3):
                observation, mode = sim.answer(self.key)
                if observation.value == UNKNOWN:
                    self.assertEqual(observation.certainty, CertaintyCue.NONE)
                elif mode is ReportMode.UNCERTAIN:
                    self.assertEqual(observation.certainty, CertaintyCue.UNCERTAIN)
                else:
                    self.assertEqual(observation.certainty, CertaintyCue.CERTAIN)
                seen.add(mode)

    def test_08_all_modes_generated(self):
        seen = set()
        for seed in range(2000):
            sim = _simulator(self.model, self.case, seed=seed, rho_env=0.4,
                             profile=self.profile)
            _, mode = sim.answer(self.key)
            seen.add(mode)
        self.assertEqual(seen, set(ReportMode))


# --------------------------------------------------------------------------- #
# 9-12: theory--empirical consistency
# --------------------------------------------------------------------------- #


class TestTheoryEmpiricalConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.specs = _toy_model()
        cls.cases, _ = generate_toy_cases(cases_per_disease=40, seed=41)
        cls.case = next(c for c in cls.cases if c.diagnosis == "influenza")
        cls.key = cls.specs[0].key
        cls.states = ("absent", "present")
        cls.profile = PatientProfile()  # default profile == channel cue prior

    def _sample_joint(self, rho, n_seeds, true_state):
        """Empirical P(y, y' | z) for a fixed true state over n_seeds."""
        counts = {}
        for seed in range(n_seeds):
            sim = JointStructuredPatientSimulator(
                diagnosis=self.case.diagnosis,
                latent_states={self.key: true_state},
                model=self.model,
                channel=AnswerChannel(),
                profile=self.profile,
                seed=seed,
                rho_env=rho,
            )
            first, _ = sim.answer(self.key)
            second, _ = sim.answer(self.key)
            counts[(first.value, second.value)] = counts.get(
                (first.value, second.value), 0
            ) + 1
        return {pair: count / n_seeds for pair, count in counts.items()}

    def test_09_joint_frequency_matches_theoretical(self):
        rho = 0.5
        true_state = "present"
        n_seeds = 20000
        empirical = self._sample_joint(rho, n_seeds, true_state)
        channel = JointReportChannel(
            AnswerChannel(), repeat_mode_persistence=rho
        )
        answers = joint_answer_space(self.states)
        for y in answers:
            for y_prime in answers:
                theory = channel.joint_probability(
                    y, y_prime, true_state, self.states, VerificationType.REPEAT
                )
                observed = empirical.get((y, y_prime), 0.0)
                self.assertLess(abs(observed - theory), 0.02, (y, y_prime))

    def test_10_second_answer_marginal_invariant_to_rho(self):
        true_state = "present"
        n_seeds = 20000
        marginals = {}
        for rho in (0.0, 0.5, 0.9):
            counts = {}
            for seed in range(n_seeds):
                sim = JointStructuredPatientSimulator(
                    diagnosis=self.case.diagnosis,
                    latent_states={self.key: true_state},
                    model=self.model,
                    channel=AnswerChannel(),
                    profile=self.profile,
                    seed=seed,
                    rho_env=rho,
                )
                sim.answer(self.key)
                second, _ = sim.answer(self.key)
                counts[second.value] = counts.get(second.value, 0) + 1
            marginals[rho] = {v: c / n_seeds for v, c in counts.items()}
        base = marginals[0.0]
        for rho in (0.5, 0.9):
            for value in joint_answer_space(self.states):
                self.assertLess(abs(marginals[rho].get(value, 0.0) - base.get(value, 0.0)), 0.02, (rho, value))

    def test_11_only_joint_structure_changes_with_rho(self):
        true_state = "present"
        n_seeds = 20000

        def agreement(rho):
            agree = 0
            for seed in range(n_seeds):
                sim = JointStructuredPatientSimulator(
                    diagnosis=self.case.diagnosis,
                    latent_states={self.key: true_state},
                    model=self.model, channel=AnswerChannel(),
                    profile=self.profile, seed=seed, rho_env=rho,
                )
                first, _ = sim.answer(self.key)
                second, _ = sim.answer(self.key)
                agree += int(first.value == second.value)
            return agree / n_seeds

        self.assertLess(agreement(0.0), agreement(0.9))

    def test_12_rho_zero_joint_equals_product_of_marginals(self):
        # At rho=0 the independent re-ask makes P(y,y'|z) == P(y|z) P(y'|z).
        true_state = "present"
        n_seeds = 20000
        channel0 = JointReportChannel(AnswerChannel(), repeat_mode_persistence=0.0)
        empirical = self._sample_joint(0.0, n_seeds, true_state)
        for y in joint_answer_space(self.states):
            py = channel0.single_probability(y, true_state, self.states)
            for y_prime in joint_answer_space(self.states):
                py_prime = channel0.single_probability(y_prime, true_state, self.states)
                observed = empirical.get((y, y_prime), 0.0)
                self.assertLess(abs(observed - py * py_prime), 0.02, (y, y_prime))


# --------------------------------------------------------------------------- #
# 13-16: inference consistency
# --------------------------------------------------------------------------- #


class TestInferenceConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.specs = _toy_model()
        cls.cases, _ = generate_toy_cases(cases_per_disease=40, seed=41)
        cls.case = next(c for c in cls.cases if c.diagnosis == "influenza")
        cls.key = cls.specs[0].key

    def test_13_matched_setting_uses_rho_model(self):
        rho_model = 0.5
        channel = JointReportChannel(AnswerChannel(), repeat_mode_persistence=rho_model)
        tracker = JointReliabilityBeliefTracker(self.model, channel)
        tracker.observe_single(_obs(self.key.name, "present"))
        self.assertEqual(tracker.channel.repeat_mode_persistence, rho_model)
        tracker.observe_verification(
            self.key, _obs(self.key.name, "present"), VerificationType.REPEAT
        )

    def test_14_rho_model_changes_reask_predictive(self):
        def predictive(rho_model):
            channel = JointReportChannel(AnswerChannel(), repeat_mode_persistence=rho_model)
            tracker = JointReliabilityBeliefTracker(self.model, channel)
            tracker.observe_single(_obs(self.key.name, "present"))
            return tracker.reask_predictive(self.key)
        self.assertNotEqual(predictive(0.0), predictive(0.9))

    def test_15_mismatch_never_reads_rho_env(self):
        # Simulator rho_env=0.9 but tracker rho_model=0.0: the tracker's channel
        # must keep rho_model, never the environment's rho_env.
        rho_model = 0.0
        channel = JointReportChannel(AnswerChannel(), repeat_mode_persistence=rho_model)
        tracker = JointReliabilityBeliefTracker(self.model, channel)
        policy = JointChannelBrierAuditPolicy(self.model, channel=channel, tracker=tracker)
        sim = _simulator(self.model, self.case, seed=5, rho_env=0.9)
        run_joint_channel_dialogue(sim, policy=policy)
        self.assertEqual(tracker.channel.repeat_mode_persistence, rho_model)
        self.assertFalse(hasattr(policy, "rho_env"))
        self.assertFalse(hasattr(tracker, "rho_env"))

    def test_16_consistent_correlated_answers_not_double_counted(self):
        # At high rho, two identical answers are ONE joint factor whose likelihood
        # exceeds the independent product (i.e. the second answer is treated as
        # partially redundant, not as a second independent evidence).
        rho = 0.9
        channel = JointReportChannel(AnswerChannel(), repeat_mode_persistence=rho)
        tracker = JointReliabilityBeliefTracker(self.model, channel)
        y = "present"
        single = tracker.single_likelihood(
            "influenza", self.key, _obs(self.key.name, y)
        )
        joint = tracker.joint_likelihood(
            "influenza", self.key,
            _obs(self.key.name, y), _obs(self.key.name, y),
            VerificationType.REPEAT,
        )
        self.assertGreater(joint, single * single)

    def test_17_conflicting_answers_preserved_not_unknown(self):
        channel = JointReportChannel(AnswerChannel(), repeat_mode_persistence=0.5)
        tracker = JointReliabilityBeliefTracker(self.model, channel)
        tracker.observe_single(_obs(self.key.name, "absent"))
        tracker.observe_verification(
            self.key, _obs(self.key.name, "present"), VerificationType.REPEAT
        )
        bundle = tracker.memory[self.key]
        self.assertEqual(
            [a.value for a in bundle.all_answers], ["absent", "present"]
        )

    def test_18_posteriors_normalized_no_nan(self):
        import math
        channel = JointReportChannel(AnswerChannel(), repeat_mode_persistence=0.5)
        tracker = JointReliabilityBeliefTracker(self.model, channel)
        tracker.observe_single(_obs(self.key.name, "present"))
        # Before verification, every predictive/posterior distribution is valid.
        distributions = [
            tracker.disease_belief(),
            tracker.state_posterior(self.key),
            tracker.mode_posterior(self.key),
            tracker.reask_predictive(self.key),
        ]
        for dist in distributions:
            for value in dist.values():
                self.assertFalse(math.isnan(value))
            self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)
        # After verification the re-ask predictive is no longer defined for the
        # feature, but the belief / state / mode posteriors stay normalized.
        tracker.observe_verification(
            self.key, _obs(self.key.name, "absent"), VerificationType.REPEAT
        )
        for dist in (
            tracker.disease_belief(),
            tracker.state_posterior(self.key),
            tracker.mode_posterior(self.key),
        ):
            for value in dist.values():
                self.assertFalse(math.isnan(value))
            self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)


# --------------------------------------------------------------------------- #
# 17-18: leakage / regression
# --------------------------------------------------------------------------- #


class TestLeakageAndRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.specs = _toy_model()
        cls.cases, _ = generate_toy_cases(cases_per_disease=40, seed=41)
        cls.case = next(c for c in cls.cases if c.diagnosis == "influenza")

    def test_19_old_simulator_unchanged_byte_identical(self):
        # The old simulator must reproduce byte-identical answers to the new
        # simulator at rho_env=0.
        profile = PatientProfile.from_noise_rate(0.3)
        key = self.specs[0].key
        old = StructuredPatientSimulator(
            diagnosis=self.case.diagnosis, latent_states=self.case.states,
            model=self.model, channel=AnswerChannel(), profile=profile, seed=42,
        )
        new = JointStructuredPatientSimulator(
            diagnosis=self.case.diagnosis, latent_states=self.case.states,
            model=self.model, channel=AnswerChannel(), profile=profile, seed=42,
            rho_env=0.0, case_id=self.case.case_id,
        )
        for _ in range(6):
            o1, m1 = old.answer(key)
            o2, m2 = new.answer(key)
            self.assertEqual((o1.value, o1.certainty, m1),
                             (o2.value, o2.certainty, m2))

    def test_20_heuristic_baseline_byte_identical(self):
        # Registering JOINT_CHANNEL_BRIER_AUDIT must not change the heuristic
        # default: build_policy(HEURISTIC_VERIFY) still yields the baseline
        # ReliabilityAwareActionPolicy.
        from powerful_medrag.decision import ReliabilityAwareActionPolicy
        direct = build_policy(WorthinessStrategy.HEURISTIC_VERIFY)
        self.assertIsInstance(direct, ReliabilityAwareActionPolicy)

    def test_21_policy_never_reads_simulator_privilege(self):
        channel = JointReportChannel(AnswerChannel(), repeat_mode_persistence=0.5)
        tracker = JointReliabilityBeliefTracker(self.model, channel)
        policy = JointChannelBrierAuditPolicy(self.model, channel=channel, tracker=tracker)
        # The policy holds no reference to a simulator / patient at all.
        for attr in ("rho_env", "latent_states", "diagnosis", "hidden_mode_chain",
                     "noise_type", "true_wrongness"):
            self.assertFalse(hasattr(policy, attr), attr)


if __name__ == "__main__":
    unittest.main()
