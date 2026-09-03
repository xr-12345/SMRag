"""Correlated-re-ask patient simulator (Phase 8C).

The Phase 8B joint inference model assumed a re-ask mode correlation
``rho_model`` (``JointReportChannel.repeat_mode_persistence``), but the patient
simulator still sampled every answer independently.  This module closes that
environment/inference gap with a simulator that samples a *correlated* re-ask:

    E_i^(1) ~ P(E)
    Y_i^(1) ~ P(Y | Z_i, E_i^(1))
    E_i^(k) ~ T_v(E_i^(k) | E_i^(k-1))          (k >= 2)
    Y_i^(k) ~ P(Y | Z_i, E_i^(k))

    T_v(e' | e) = rho_env * 1[e' = e] + (1 - rho_env) * P(e')

Only ``VerificationType.REPEAT`` is formally supported (the others remain an
extensible interface and raise ``NotImplementedError``).  ``rho_env`` is the
*environment* correlation -- the simulator's real answer mechanism -- and must
never be confused with ``rho_model`` (the inference assumption).  Both are logged
separately by the experiment harness.

Backward compatibility
----------------------

* The old ``StructuredPatientSimulator`` is untouched (red line).
* At ``rho_env = 0`` the mode transition degenerates to ``T_v(e'|e) = P(e')`` and
  this simulator samples every occurrence independently, **byte-identical** to
  ``StructuredPatientSimulator.answer`` for the same ``(seed, key, occurrence)``.
* The first answer (occurrence 0) is always independent and byte-identical to the
  old simulator regardless of ``rho_env``.

Determinism
-----------

An answer is a pure function of ``(case -> latent_states, seed, key.token,
occurrence, verification_type, rho_env)``.  The per-answer RNG stream is derived
from ``blake2b(seed | key.token | occurrence)`` for the independent path (exactly
the old simulator's derivation) and from ``blake2b(seed | key.token | occurrence
| verification_type | rho_env)`` for the correlated re-ask path.  ``case_id``
enters via the case's latent states (as in the old simulator), so the full input
tuple ``(case_id, noise_seed, feature_key, answer_occurrence, verification_type,
rho_env)`` uniquely determines every answer.  No global RNG state is used.

Red lines: the hidden mode chain is recorded only for evaluation; it never enters
any policy decision.  ``rho_env`` is never read by the inference model.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Mapping

from .channel import AnswerChannel, ReportMode, _weighted_choice
from .estimation import DiseaseStateModel
from .joint_report_channel import VerificationType
from .schema import UNKNOWN, CertaintyCue, FeatureKey, Observation
from .simulator import PatientProfile


@dataclass(frozen=True)
class CorrelatedAnswer:
    """One sampled answer plus its hidden report mode (evaluation-only metadata)."""

    occurrence: int
    mode: ReportMode
    observation: Observation


class JointStructuredPatientSimulator:
    """Samples reports from fixed latent truth with an optional correlated re-ask.

    Drop-in compatible with :class:`StructuredPatientSimulator` (same ``answer``
    return type, ``latent_states`` / ``diagnosis`` / ``model`` / ``channel`` /
    ``profile`` / ``seed`` attributes).  Adds ``rho_env`` (environment re-ask
    correlation) and a per-feature hidden mode chain used only by evaluation.
    """

    def __init__(
        self,
        *,
        diagnosis: str,
        latent_states: Mapping[FeatureKey, str],
        model: DiseaseStateModel,
        channel: AnswerChannel | None = None,
        profile: PatientProfile | None = None,
        seed: int = 0,
        rho_env: float = 0.0,
        case_id: str = "",
    ) -> None:
        if diagnosis not in model.diseases:
            raise ValueError(f"unknown simulated diagnosis: {diagnosis}")
        if not 0.0 <= rho_env <= 1.0:
            raise ValueError("rho_env must be in [0, 1]")
        self.diagnosis = diagnosis
        self.latent_states = dict(latent_states)
        self.model = model
        self.channel = channel or AnswerChannel()
        self.profile = profile or PatientProfile()
        self.seed = seed
        self.rho_env = float(rho_env)
        self.case_id = case_id
        self._answer_counts: dict[FeatureKey, int] = {}
        self._mode_chain: dict[FeatureKey, list[ReportMode]] = {}
        self._answer_chain: dict[FeatureKey, list[Observation]] = {}

    # -- construction -------------------------------------------------------- #

    @classmethod
    def sample_case(
        cls,
        model: DiseaseStateModel,
        diagnosis: str,
        *,
        channel: AnswerChannel | None = None,
        profile: PatientProfile | None = None,
        seed: int = 0,
        rho_env: float = 0.0,
        case_id: str = "",
    ) -> "JointStructuredPatientSimulator":
        """Sample latent states from ``(diagnosis, seed)`` and build a simulator."""
        rng = random.Random(seed)
        latent_states: dict[FeatureKey, str] = {}
        for key in model.specs:
            distribution = model.state_distribution(diagnosis, key)
            target = rng.random()
            cumulative = 0.0
            selected = next(iter(distribution))
            for value, probability in distribution.items():
                selected = value
                cumulative += probability
                if target <= cumulative:
                    break
            latent_states[key] = selected
        return cls(
            diagnosis=diagnosis,
            latent_states=latent_states,
            model=model,
            channel=channel,
            profile=profile,
            seed=seed,
            rho_env=rho_env,
            case_id=case_id,
        )

    # -- hidden mode chain (evaluation-only) --------------------------------- #

    @property
    def hidden_mode_chain(self) -> dict[FeatureKey, tuple[ReportMode, ...]]:
        """Frozen copy of each feature's sampled report-mode chain."""
        return {key: tuple(modes) for key, modes in self._mode_chain.items()}

    def mode_chain_for(self, key: FeatureKey) -> tuple[ReportMode, ...]:
        return tuple(self._mode_chain.get(key, ()))

    def answer_chain_for(self, key: FeatureKey) -> tuple[Observation, ...]:
        return tuple(self._answer_chain.get(key, ()))

    # -- sampling ------------------------------------------------------------ #

    def answer(
        self,
        key: FeatureKey,
        verification_type: VerificationType = VerificationType.REPEAT,
    ) -> tuple[Observation, ReportMode]:
        """Sample one answer for ``key``, correlated with prior occurrences.

        Returns ``(Observation, ReportMode)`` exactly like the old simulator, so
        this class is drop-in for both dialogue runners.  The hidden mode is also
        appended to ``hidden_mode_chain`` for evaluation.
        """
        spec = self.model.specs[key]
        if key not in self.latent_states:
            observation = Observation(
                key=key,
                value=UNKNOWN,
                oracle_report_mode=ReportMode.UNKNOWN.value,
            )
            self._mode_chain.setdefault(key, []).append(ReportMode.UNKNOWN)
            self._answer_chain.setdefault(key, []).append(observation)
            return observation, ReportMode.UNKNOWN

        occurrence = self._answer_counts.get(key, 0)
        self._answer_counts[key] = occurrence + 1

        if verification_type is not VerificationType.REPEAT:
            raise NotImplementedError(
                f"verification type {verification_type.value!r} has no simulator "
                "sampling yet; only REPEAT is implemented"
            )

        if occurrence == 0 or self.rho_env == 0.0:
            # Independent path: byte-identical to StructuredPatientSimulator.answer.
            digest = hashlib.blake2b(
                f"{self.seed}|{key.token}|{occurrence}".encode("utf-8"),
                digest_size=8,
            ).digest()
            answer_rng = random.Random(int.from_bytes(digest, "big"))
            observed_value, mode = self.channel.sample(
                self.latent_states[key],
                spec.values,
                rng=answer_rng,
                mode_prior=self.profile.mode_prior,
            )
        else:
            # Correlated re-ask: e' ~ T_v(. | e_prev), y' ~ P(Y | z, e').
            prev_mode = self._mode_chain[key][-1]
            digest = hashlib.blake2b(
                f"{self.seed}|{key.token}|{occurrence}"
                f"|{verification_type.value}|{self.rho_env:g}".encode("utf-8"),
                digest_size=8,
            ).digest()
            answer_rng = random.Random(int.from_bytes(digest, "big"))
            mode = self._sample_transition_mode(prev_mode, answer_rng)
            observed_value = self._sample_answer_given_mode(
                self.latent_states[key], spec.values, mode, answer_rng
            )

        if observed_value == UNKNOWN:
            certainty = CertaintyCue.NONE
        elif mode == ReportMode.UNCERTAIN:
            certainty = CertaintyCue.UNCERTAIN
        else:
            certainty = CertaintyCue.CERTAIN

        observation = Observation(
            key=key,
            value=observed_value,
            certainty=certainty,
            oracle_report_mode=mode.value,
        )
        self._mode_chain.setdefault(key, []).append(mode)
        self._answer_chain.setdefault(key, []).append(observation)
        return observation, mode

    def _sample_transition_mode(
        self, prev_mode: ReportMode, rng: random.Random
    ) -> ReportMode:
        """Sample e' ~ rho * 1[e' = e] + (1 - rho) * P(e')."""
        if rng.random() < self.rho_env:
            return prev_mode
        return _weighted_choice(self.profile.mode_prior, rng)

    def _sample_answer_given_mode(
        self,
        true_state: str,
        states: tuple[str, ...],
        mode: ReportMode,
        rng: random.Random,
    ) -> str:
        answer_probabilities = {
            value: self.channel.probability(value, true_state, states, mode)
            for value in (*states, UNKNOWN)
        }
        return _weighted_choice(answer_probabilities, rng)
