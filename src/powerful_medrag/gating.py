"""History-dependent gates for activating the latent misreport state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .schema import UNKNOWN, CertaintyCue, Observation

if TYPE_CHECKING:
    from .belief import BeliefTracker


class MisreportGate(Protocol):
    def probability(
        self, tracker: "BeliefTracker", observation: Observation
    ) -> float: ...


@dataclass(frozen=True)
class HeuristicMisreportGate:
    """Conservative, label-free gate based only on observable dialogue state.

    A single surprising answer can activate at most ``single_answer_cap``.
    Direct contradiction with a previous report permits the larger
    ``conflict_cap``.  This limits confirmation bias when a surprising answer
    is actually correcting an incorrect current diagnosis.
    """

    baseline_probability: float = 0.005
    surprise_threshold: float = 2.30
    surprise_weight: float = 1.25
    confident_surprise_weight: float = 0.75
    conflict_weight: float = 3.0
    uncertainty_discount: float = 0.75
    unknown_discount: float = 1.50
    single_answer_cap: float = 0.15
    conflict_cap: float = 0.40
    activation_threshold: float = 0.01

    def __post_init__(self) -> None:
        probabilities = (
            self.baseline_probability,
            self.single_answer_cap,
            self.conflict_cap,
            self.activation_threshold,
        )
        if min(probabilities) < 0 or max(probabilities) >= 1:
            raise ValueError("gate probabilities must be in [0, 1)")
        if self.baseline_probability > self.single_answer_cap:
            raise ValueError("baseline probability cannot exceed single-answer cap")
        if self.single_answer_cap > self.conflict_cap:
            raise ValueError("single-answer cap cannot exceed conflict cap")
        if min(
            self.surprise_threshold,
            self.surprise_weight,
            self.confident_surprise_weight,
            self.conflict_weight,
            self.uncertainty_discount,
            self.unknown_discount,
        ) < 0:
            raise ValueError("gate weights and thresholds cannot be negative")

    def probability(
        self, tracker: "BeliefTracker", observation: Observation
    ) -> float:
        predictive_probability = self.predictive_probability(tracker, observation)
        conflicts = sum(
            previous.observation.key == observation.key
            and previous.observation.value != UNKNOWN
            and observation.value != UNKNOWN
            and previous.observation.value != observation.value
            for previous in tracker.history
        )
        return self.probability_from_signals(
            predictive_probability=predictive_probability,
            certainty=observation.certainty,
            is_unknown=observation.value == UNKNOWN,
            direct_conflicts=conflicts,
        )

    def probability_from_signals(
        self,
        *,
        predictive_probability: float,
        certainty: CertaintyCue,
        is_unknown: bool,
        direct_conflicts: int = 0,
    ) -> float:
        probability = min(max(predictive_probability, 1e-12), 1.0)
        surprise = -math.log(probability)
        excess_surprise = max(0.0, surprise - self.surprise_threshold)
        baseline = min(max(self.baseline_probability, 1e-12), 1.0 - 1e-12)
        logit = math.log(baseline / (1.0 - baseline))
        logit += self.surprise_weight * excess_surprise
        if certainty == CertaintyCue.CERTAIN:
            logit += self.confident_surprise_weight * excess_surprise
        elif certainty == CertaintyCue.UNCERTAIN:
            logit -= self.uncertainty_discount
        if is_unknown:
            logit -= self.unknown_discount
        logit += self.conflict_weight * min(max(direct_conflicts, 0), 2)
        if logit >= 0:
            raw_probability = 1.0 / (1.0 + math.exp(-logit))
        else:
            exponential = math.exp(logit)
            raw_probability = exponential / (1.0 + exponential)
        cap = self.conflict_cap if direct_conflicts else self.single_answer_cap
        return min(cap, max(self.baseline_probability, raw_probability))

    @staticmethod
    def predictive_probability(
        tracker: "BeliefTracker", observation: Observation
    ) -> float:
        """Return P(observed answer | dialogue so far) under the base channel."""

        spec = tracker.model.specs[observation.key]
        return sum(
            tracker.belief[disease]
            * sum(
                state_probability
                * tracker.channel.marginal_probability(
                    observation.value,
                    state,
                    spec.values,
                    observation.certainty,
                )
                for state, state_probability in tracker.state_beliefs[disease][
                    observation.key
                ].items()
            )
            for disease in tracker.model.diseases
        )


@dataclass(frozen=True)
class SparseSurprisalMisreportGate:
    """Activate only for confident answers that are extreme under the base model.

    The three probability levels are deliberately conservative relative to the
    empirical validation-bin frequencies.  In particular, the highest prior is
    kept below 0.5 so that one unusual answer is attenuated rather than treated
    as evidence for its opposite.
    """

    low_surprisal: float = 3.0
    medium_surprisal: float = 3.5
    high_surprisal: float = 4.0
    low_probability: float = 0.20
    medium_probability: float = 0.35
    high_probability: float = 0.49
    conflict_probability: float = 0.49
    activation_threshold: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.low_surprisal < self.medium_surprisal < self.high_surprisal:
            raise ValueError("surprisal cutoffs must be nonnegative and increasing")
        probabilities = (
            self.low_probability,
            self.medium_probability,
            self.high_probability,
            self.conflict_probability,
        )
        if min(probabilities) < 0 or max(probabilities) >= 0.5:
            raise ValueError("sparse gate probabilities must be in [0, 0.5)")
        if not (
            self.low_probability
            <= self.medium_probability
            <= self.high_probability
        ):
            raise ValueError("sparse gate probabilities must be nondecreasing")

    def probability(
        self, tracker: "BeliefTracker", observation: Observation
    ) -> float:
        conflicts = sum(
            previous.observation.key == observation.key
            and previous.observation.value != UNKNOWN
            and observation.value != UNKNOWN
            and previous.observation.value != observation.value
            for previous in tracker.history
        )
        if conflicts:
            return self.conflict_probability
        if (
            observation.certainty != CertaintyCue.CERTAIN
            or observation.value == UNKNOWN
        ):
            return 0.0
        predictive_probability = HeuristicMisreportGate.predictive_probability(
            tracker, observation
        )
        surprisal = -math.log(min(max(predictive_probability, 1e-12), 1.0))
        if surprisal < self.low_surprisal:
            return 0.0
        if surprisal < self.medium_surprisal:
            return self.low_probability
        if surprisal < self.high_surprisal:
            return self.medium_probability
        return self.high_probability


@dataclass(frozen=True)
class HistoryReliabilityMisreportGate:
    """Estimate patient-level unreliability from earlier observable cues.

    Unlike answer-surprisal gates, this gate never distrusts an answer merely
    because it disagrees with the current diagnostic belief.  Previous unknown
    or explicitly uncertain answers provide independent evidence that this
    patient's subsequent confident reports may also be noisy.
    """

    unreliable_cue_scale: float = 0.55
    pseudo_turns: float = 3.0
    misreport_share: float = 0.30
    maximum_probability: float = 0.12
    minimum_unreliable_cues: int = 1
    conflict_probability: float = 0.30
    activation_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.unreliable_cue_scale <= 0 or self.pseudo_turns < 0:
            raise ValueError("history-gate scale must be positive and smoothing nonnegative")
        probabilities = (
            self.misreport_share,
            self.maximum_probability,
            self.conflict_probability,
        )
        if min(probabilities) < 0 or max(probabilities) >= 1:
            raise ValueError("history-gate probabilities must be in [0, 1)")
        if self.minimum_unreliable_cues <= 0:
            raise ValueError("minimum_unreliable_cues must be positive")

    def probability(
        self, tracker: "BeliefTracker", observation: Observation
    ) -> float:
        conflicts = sum(
            previous.observation.key == observation.key
            and previous.observation.value != UNKNOWN
            and observation.value != UNKNOWN
            and previous.observation.value != observation.value
            for previous in tracker.history
        )
        if conflicts:
            return self.conflict_probability
        if (
            observation.certainty != CertaintyCue.CERTAIN
            or observation.value == UNKNOWN
        ):
            return 0.0
        unreliable_cues = sum(
            previous.observation.value == UNKNOWN
            or previous.observation.certainty == CertaintyCue.UNCERTAIN
            for previous in tracker.history
        )
        if unreliable_cues < self.minimum_unreliable_cues:
            return 0.0
        estimated_noise_rate = unreliable_cues / (
            self.unreliable_cue_scale
            * (len(tracker.history) + self.pseudo_turns)
        )
        return min(
            self.maximum_probability,
            self.misreport_share * estimated_noise_rate,
        )
