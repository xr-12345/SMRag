"""Joint report channel P(y_i, y_i' | z_i, verification).

Phase 8B: the disease posterior, per-answer reliability, and the VerifyOld
prediction previously came from *different* answer channels.  This module gives
them one shared generative model for a feature asked twice (once originally,
once under verification):

    P(y_i, y_i' | z_i, v)
        = sum_{e_i, e_i'}  P(e_i) P(y_i | z_i, e_i)
                           T_v(e_i' | e_i) P(y_i' | z_i, e_i')

where

* ``z_i`` is the feature's true clinical state,
* ``e_i``  is the first answer's latent report mode,
* ``e_i'`` is the re-ask's latent report mode,
* ``v``    is the verification type (only ``REPEAT`` is implemented here).

The mode transition ``T_v`` models the dependence between the two answers::

    T(e' | e) = rho * 1[e' = e] + (1 - rho) * P_reask(e')

with ``rho = repeat_mode_persistence``.  ``rho = 0`` is an independent re-ask,
``rho = 1`` is a fully persistent report mode.  ``rho`` is a **simulation
assumption**: DDXPlus contains no repeated-answer records, so it must never be
described as an estimate of real patient behaviour.

Red lines: this channel only reads ``AnswerChannel`` rates/cue-priors and the
observable answers.  It never reads the true disease, latent state, noise label,
or true wrongness.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping, Sequence

from .channel import AnswerChannel, ReportMode
from .schema import UNKNOWN, CertaintyCue


class VerificationType(str, Enum):
    """How a previous answer is verified (only REPEAT is formally modelled)."""

    REPEAT = "repeat"
    REPHRASE = "rephrase"
    DEFINITION = "definition"
    TEMPORAL_ANCHOR = "temporal_anchor"
    CONTRASTIVE = "contrastive"


class JointReportChannel:
    """One shared generative model for single and repeated answers.

    Reuses the frozen ``AnswerChannel`` confusion rates and cue priors, so the
    single-answer marginal equals ``AnswerChannel.marginal_probability`` exactly
    (the independent re-ask is a special case of this joint model).
    """

    def __init__(
        self,
        channel: AnswerChannel | None = None,
        *,
        repeat_mode_persistence: float = 0.5,
    ) -> None:
        self.channel = channel or AnswerChannel()
        if not 0.0 <= repeat_mode_persistence <= 1.0:
            raise ValueError("repeat_mode_persistence must be in [0, 1]")
        self.repeat_mode_persistence = float(repeat_mode_persistence)

    # -- mode priors -------------------------------------------------------- #

    def mode_prior(self, cue: CertaintyCue = CertaintyCue.NONE) -> dict[ReportMode, float]:
        """P(e) for an answer with the given observable certainty cue."""
        return dict(self.channel.parameters.cue_priors[cue])

    def reask_mode_prior(self) -> dict[ReportMode, float]:
        """Unconditional P(e') used by the ``(1 - rho)`` transition term."""
        return dict(self.channel.parameters.cue_priors[CertaintyCue.NONE])

    # -- transition T_v(e' | e) -------------------------------------------- #

    def mode_transition(
        self,
        e_prime: ReportMode,
        e: ReportMode,
        verification_type: VerificationType = VerificationType.REPEAT,
    ) -> float:
        """T_v(e' | e): the re-ask mode distribution given the first mode.

        Only ``REPEAT`` is implemented; the other verification types are part of
        the extensibility interface and raise until they are specified.
        """
        if verification_type is not VerificationType.REPEAT:
            raise NotImplementedError(
                f"verification type {verification_type.value!r} has no mode "
                "transition yet; only REPEAT is implemented"
            )
        rho = self.repeat_mode_persistence
        prior = self.reask_mode_prior()
        if e_prime == e:
            return rho + (1.0 - rho) * prior[e]
        return (1.0 - rho) * prior[e_prime]

    # -- single-answer marginal -------------------------------------------- #

    def single_probability(
        self,
        observed_value: str,
        true_state: str,
        states: Sequence[str],
        cue: CertaintyCue = CertaintyCue.NONE,
    ) -> float:
        """P(y | z) = sum_e P(e | cue) P(y | z, e).

        Exactly ``AnswerChannel.marginal_probability`` -- the shared channel's
        single-answer marginal is the joint model's one-answer special case.
        """
        return self.channel.marginal_probability(
            observed_value, true_state, states, cue
        )

    # -- joint probability -------------------------------------------------- #

    def joint_probability(
        self,
        y: str,
        y_prime: str,
        true_state: str,
        states: Sequence[str],
        verification_type: VerificationType = VerificationType.REPEAT,
        first_cue: CertaintyCue = CertaintyCue.NONE,
    ) -> float:
        """P(y, y' | z, v) = sum_{e,e'} P(e|first_cue) P(y|z,e) T_v(e'|e) P(y'|z,e')."""
        if true_state not in states:
            raise ValueError(f"true state {true_state!r} is outside the variable states")
        answer_space = (*states, UNKNOWN)
        if y not in answer_space:
            raise ValueError(f"first answer {y!r} is outside the answer space")
        if y_prime not in answer_space:
            raise ValueError(f"re-ask answer {y_prime!r} is outside the answer space")

        total = 0.0
        for e, e_weight in self.mode_prior(first_cue).items():
            if e_weight <= 0:
                continue
            first = self.channel.probability(y, true_state, states, e)
            if first <= 0:
                continue
            second = 0.0
            for e_prime in ReportMode:
                transition = self.mode_transition(e_prime, e, verification_type)
                if transition <= 0:
                    continue
                second += transition * self.channel.probability(
                    y_prime, true_state, states, e_prime
                )
            total += e_weight * first * second
        return total

    def reask_conditional_probability(
        self,
        y_prime: str,
        true_state: str,
        e: ReportMode,
        states: Sequence[str],
        verification_type: VerificationType = VerificationType.REPEAT,
    ) -> float:
        """sum_{e'} T_v(e' | e) P(y' | z, e') -- the re-ask density given mode ``e``."""
        total = 0.0
        for e_prime in ReportMode:
            transition = self.mode_transition(e_prime, e, verification_type)
            if transition <= 0:
                continue
            total += transition * self.channel.probability(
                y_prime, true_state, states, e_prime
            )
        return total


def joint_answer_space(states: Sequence[str]) -> tuple[str, ...]:
    """The full answer space used by a feature: ``(*values, UNKNOWN)``."""
    return (*states, UNKNOWN)
