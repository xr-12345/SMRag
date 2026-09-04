"""Observation channel P(reported answer | true state, latent report mode)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from .schema import UNKNOWN, CertaintyCue, Observation


class ReportMode(str, Enum):
    CERTAIN = "certain"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"
    MISREPORTED = "misreported"


@dataclass(frozen=True)
class ModeRates:
    correct: float
    unknown: float
    wrong: float

    def __post_init__(self) -> None:
        if min(self.correct, self.unknown, self.wrong) < 0:
            raise ValueError("channel rates cannot be negative")
        if abs(self.correct + self.unknown + self.wrong - 1.0) > 1e-9:
            raise ValueError("channel rates must sum to one")


@dataclass(frozen=True)
class ChannelParameters:
    """Default confusion behavior for each latent reporting mode."""

    rates: Mapping[ReportMode, ModeRates] = field(
        default_factory=lambda: {
            ReportMode.CERTAIN: ModeRates(correct=0.98, unknown=0.01, wrong=0.01),
            ReportMode.UNCERTAIN: ModeRates(correct=0.55, unknown=0.35, wrong=0.10),
            ReportMode.UNKNOWN: ModeRates(correct=0.04, unknown=0.94, wrong=0.02),
            ReportMode.MISREPORTED: ModeRates(correct=0.08, unknown=0.07, wrong=0.85),
        }
    )
    cue_priors: Mapping[CertaintyCue, Mapping[ReportMode, float]] = field(
        default_factory=lambda: {
            CertaintyCue.NONE: {
                ReportMode.CERTAIN: 0.82,
                ReportMode.UNCERTAIN: 0.10,
                ReportMode.UNKNOWN: 0.05,
                ReportMode.MISREPORTED: 0.03,
            },
            CertaintyCue.CERTAIN: {
                ReportMode.CERTAIN: 0.93,
                ReportMode.UNCERTAIN: 0.03,
                ReportMode.UNKNOWN: 0.01,
                ReportMode.MISREPORTED: 0.03,
            },
            CertaintyCue.UNCERTAIN: {
                ReportMode.CERTAIN: 0.18,
                ReportMode.UNCERTAIN: 0.70,
                ReportMode.UNKNOWN: 0.09,
                ReportMode.MISREPORTED: 0.03,
            },
        }
    )

    def __post_init__(self) -> None:
        if set(self.rates) != set(ReportMode):
            raise ValueError("rates must cover every report mode")
        for cue in CertaintyCue:
            weights = self.cue_priors.get(cue)
            if weights is None or set(weights) != set(ReportMode):
                raise ValueError(f"cue prior for {cue.value} must cover every report mode")
            if min(weights.values()) < 0 or abs(sum(weights.values()) - 1.0) > 1e-9:
                raise ValueError("mode prior weights must be nonnegative and sum to one")


class AnswerChannel:
    """Marginalizes the unobserved reporting mode instead of hard-labeling it."""

    def __init__(self, parameters: ChannelParameters | None = None) -> None:
        self.parameters = parameters or ChannelParameters()

    def probability(
        self,
        observed_value: str,
        true_state: str,
        states: Sequence[str],
        mode: ReportMode,
    ) -> float:
        if true_state not in states:
            raise ValueError(f"true state {true_state!r} is outside the variable states")
        if observed_value != UNKNOWN and observed_value not in states:
            raise ValueError(f"reported value {observed_value!r} is outside the answer space")
        rates = self.parameters.rates[mode]
        if observed_value == UNKNOWN:
            return rates.unknown
        if observed_value == true_state:
            return rates.correct
        return rates.wrong / (len(states) - 1)

    def marginal_probability(
        self,
        observed_value: str,
        true_state: str,
        states: Sequence[str],
        cue: CertaintyCue = CertaintyCue.NONE,
        mode_prior: Mapping[ReportMode, float] | None = None,
    ) -> float:
        prior = mode_prior or self.parameters.cue_priors[cue]
        return sum(
            mode_weight
            * self.probability(observed_value, true_state, states, mode)
            for mode, mode_weight in prior.items()
        )

    def mode_posterior(
        self,
        observation: Observation,
        true_state_distribution: Mapping[str, float],
        mode_prior: Mapping[ReportMode, float] | None = None,
    ) -> dict[ReportMode, float]:
        states = tuple(true_state_distribution)
        prior = mode_prior or self.parameters.cue_priors[observation.certainty]
        unnormalized = {
            mode: prior[mode]
            * sum(
                state_probability
                * self.probability(observation.value, state, states, mode)
                for state, state_probability in true_state_distribution.items()
            )
            for mode in ReportMode
        }
        denominator = sum(unnormalized.values())
        if denominator <= 0:
            return dict(prior)
        return {mode: value / denominator for mode, value in unnormalized.items()}

    def sample(
        self,
        true_state: str,
        states: Sequence[str],
        *,
        rng: random.Random,
        mode_prior: Mapping[ReportMode, float] | None = None,
    ) -> tuple[str, ReportMode]:
        prior = mode_prior or self.parameters.cue_priors[CertaintyCue.NONE]
        mode = _weighted_choice(prior, rng)
        answer_probabilities = {
            value: self.probability(value, true_state, states, mode)
            for value in (*states, UNKNOWN)
        }
        return _weighted_choice(answer_probabilities, rng), mode


def reliable_answer_channel(*, epsilon: float = 1e-6) -> AnswerChannel:
    """Single-layer ablation: known reports are treated as virtually exact.

    A tiny non-zero mass keeps contradictory synthetic reports numerically
    possible.  ``UNKNOWN`` has the same likelihood under every true state and
    is therefore uninformative rather than being silently treated as negative.
    """

    if not 0 < epsilon < 1 / 3:
        raise ValueError("epsilon must be in (0, 1/3)")
    rates = {
        mode: ModeRates(
            correct=1.0 - 2.0 * epsilon,
            unknown=epsilon,
            wrong=epsilon,
        )
        for mode in ReportMode
    }
    cue_priors = {
        cue: {
            mode: 1.0 if mode == ReportMode.CERTAIN else 0.0
            for mode in ReportMode
        }
        for cue in CertaintyCue
    }
    return AnswerChannel(ChannelParameters(rates=rates, cue_priors=cue_priors))


def answer_channel_without_misreport() -> AnswerChannel:
    """Two-layer ablation whose inference prior excludes MISREPORTED."""

    base = ChannelParameters()
    cue_priors: dict[CertaintyCue, dict[ReportMode, float]] = {}
    for cue, weights in base.cue_priors.items():
        retained_mass = 1.0 - weights[ReportMode.MISREPORTED]
        cue_priors[cue] = {
            mode: (
                0.0
                if mode == ReportMode.MISREPORTED
                else weight / retained_mass
            )
            for mode, weight in weights.items()
        }
    return AnswerChannel(
        ChannelParameters(rates=base.rates, cue_priors=cue_priors)
    )


def protocol_fixed_prior() -> dict[ReportMode, float]:
    """Cross-noise-average report-mode prior from the protocol's noise levels.

    This is the Phase 8D "train_fixed" prior, renamed ``protocol_fixed_prior``
    because it is a **computed design-level average** -- the arithmetic mean of
    ``PatientProfile.from_noise_rate(n)`` over the protocol noise set {0.2, 0.3}
    with the frozen mechanism shares (uncertain 0.40 / unknown 0.30 /
    misreported 0.30) -- and is NOT learned from any training/validation/test
    label.  It is not cue-conditioned: every certainty cue shares the same prior.

    Numerically: CERTAIN 0.75, UNCERTAIN 0.10, UNKNOWN 0.075, MISREPORTED 0.075.
    """
    noise_levels = (0.2, 0.3)
    uncertain_share, unknown_share, misreported_share = 0.40, 0.30, 0.30
    n = len(noise_levels)
    return {
        ReportMode.CERTAIN: sum(1.0 - r for r in noise_levels) / n,
        ReportMode.UNCERTAIN: sum(r * uncertain_share for r in noise_levels) / n,
        ReportMode.UNKNOWN: sum(r * unknown_share for r in noise_levels) / n,
        ReportMode.MISREPORTED: sum(r * misreported_share for r in noise_levels) / n,
    }


def protocol_fixed_channel_parameters() -> ChannelParameters:
    """``ChannelParameters`` whose cue priors are the ``protocol_fixed_prior``.

    Keeps the default confusion ``rates`` and replaces every cue prior with the
    single non-cue-conditioned protocol prior.  The legacy default
    ``ChannelParameters()`` is left untouched; this is an explicit opt-in config
    (``legacy_prior`` vs ``train_fixed_prior``) so old experiments stay
    reproducible.
    """
    base = ChannelParameters()
    prior = protocol_fixed_prior()
    cue_priors = {cue: dict(prior) for cue in CertaintyCue}
    return ChannelParameters(rates=base.rates, cue_priors=cue_priors)


def _weighted_choice(weights: Mapping[object, float], rng: random.Random):
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("sampling weights must have positive mass")
    target = rng.random() * total
    cumulative = 0.0
    last = None
    for value, weight in weights.items():
        last = value
        cumulative += weight
        if target <= cumulative:
            return value
    return last
