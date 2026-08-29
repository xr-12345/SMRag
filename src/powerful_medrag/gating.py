"""History-dependent gates for activating the latent misreport state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Protocol

from .schema import UNKNOWN, CertaintyCue, Observation

if TYPE_CHECKING:
    from .belief import BeliefTracker


class MisreportGate(Protocol):
    def probability(
        self, tracker: "BeliefTracker", observation: Observation
    ) -> float: ...


LEARNED_GATE_FEATURES = (
    "surprisal",
    "direct_conflicts",
    "is_uncertain",
    "is_unknown",
    "history_unreliable_fraction",
    "top_probability",
    "normalized_belief_entropy",
    "diagnostic_impact",
    "extraction_uncertainty",
)


@dataclass(frozen=True)
class ObservableGateSignals:
    """Signals available after receiving an answer but before accepting it."""

    surprisal: float
    direct_conflicts: float
    is_uncertain: float
    is_unknown: float
    history_unreliable_fraction: float
    top_probability: float
    normalized_belief_entropy: float
    diagnostic_impact: float
    extraction_uncertainty: float

    def as_mapping(self) -> dict[str, float]:
        return {
            feature: float(getattr(self, feature))
            for feature in LEARNED_GATE_FEATURES
        }


def observable_gate_signals(
    tracker: "BeliefTracker", observation: Observation
) -> ObservableGateSignals:
    """Extract label-free gate inputs without using future or simulator truth."""

    predictive_probability = HeuristicMisreportGate.predictive_probability(
        tracker, observation
    )
    direct_conflicts = sum(
        previous.observation.key == observation.key
        and previous.observation.value != UNKNOWN
        and observation.value != UNKNOWN
        and previous.observation.value != observation.value
        for previous in tracker.history
    )
    unreliable_history = sum(
        previous.observation.value == UNKNOWN
        or previous.observation.certainty == CertaintyCue.UNCERTAIN
        for previous in tracker.history
    )
    history_fraction = unreliable_history / max(1, len(tracker.history))
    top_probability = max(tracker.belief.values())
    disease_count = len(tracker.belief)
    belief_entropy = -sum(
        probability * math.log2(probability)
        for probability in tracker.belief.values()
        if probability > 0
    )
    normalized_entropy = (
        belief_entropy / math.log2(disease_count) if disease_count > 1 else 0.0
    )

    # Compare the ordinary Bayesian-update explanation with retaining the
    # current belief.  The gate has not been invoked here: the fixed cue prior
    # is supplied explicitly, preventing recursion and privileged-label use.
    base_mode_prior = tracker.channel.parameters.cue_priors[observation.certainty]
    likelihoods = tracker.observation_likelihoods(observation, base_mode_prior)
    unnormalized = {
        disease: tracker.belief[disease] * likelihoods[disease]
        for disease in tracker.model.diseases
    }
    denominator = sum(unnormalized.values())
    ordinary_posterior = {
        disease: value / denominator for disease, value in unnormalized.items()
    }
    diagnostic_impact = 0.5 * sum(
        abs(tracker.belief[disease] - ordinary_posterior[disease])
        for disease in tracker.model.diseases
    )
    extraction_uncertainty = (
        0.0
        if observation.extraction_confidence is None
        else 1.0 - observation.extraction_confidence
    )
    return ObservableGateSignals(
        surprisal=-math.log(min(max(predictive_probability, 1e-12), 1.0)),
        direct_conflicts=float(min(direct_conflicts, 2)),
        is_uncertain=float(observation.certainty == CertaintyCue.UNCERTAIN),
        is_unknown=float(observation.value == UNKNOWN),
        history_unreliable_fraction=history_fraction,
        top_probability=top_probability,
        normalized_belief_entropy=normalized_entropy,
        diagnostic_impact=diagnostic_impact,
        extraction_uncertainty=extraction_uncertainty,
    )


@dataclass(frozen=True)
class LearnedMisreportGate:
    """Serializable calibrated logistic gate over observable online signals."""

    intercept: float
    coefficients: Mapping[str, float]
    feature_means: Mapping[str, float] = field(default_factory=dict)
    feature_scales: Mapping[str, float] = field(default_factory=dict)
    calibration_intercept: float = 0.0
    calibration_slope: float = 1.0
    minimum_probability: float = 0.005
    single_answer_cap: float = 0.15
    conflict_cap: float = 0.40
    activation_threshold: float = 0.01

    def __post_init__(self) -> None:
        unknown = set(self.coefficients) - set(LEARNED_GATE_FEATURES)
        if unknown:
            raise ValueError(f"unknown learned-gate features: {sorted(unknown)}")
        if self.calibration_slope < 0:
            raise ValueError("calibration slope cannot be negative")
        if not (
            0 <= self.minimum_probability
            <= self.single_answer_cap
            <= self.conflict_cap
            < 1
        ):
            raise ValueError("invalid learned-gate probability limits")
        if any(scale <= 0 for scale in self.feature_scales.values()):
            raise ValueError("feature scales must be positive")

    def probability(
        self, tracker: "BeliefTracker", observation: Observation
    ) -> float:
        signals = observable_gate_signals(tracker, observation).as_mapping()
        direct_conflicts = signals["direct_conflicts"]
        return self.probability_from_features(signals, has_conflict=bool(direct_conflicts))

    def probability_from_features(
        self, features: Mapping[str, float], *, has_conflict: bool | None = None
    ) -> float:
        missing = set(self.coefficients) - set(features)
        if missing:
            raise ValueError(f"missing learned-gate features: {sorted(missing)}")
        logit = self.intercept
        for feature, coefficient in self.coefficients.items():
            value = float(features[feature])
            mean = self.feature_means.get(feature, 0.0)
            scale = self.feature_scales.get(feature, 1.0)
            logit += coefficient * (value - mean) / scale
        calibrated_logit = self.calibration_intercept + self.calibration_slope * logit
        probability = _sigmoid(calibrated_logit)
        conflict = (
            bool(float(features.get("direct_conflicts", 0.0)))
            if has_conflict is None
            else has_conflict
        )
        cap = self.conflict_cap if conflict else self.single_answer_cap
        return min(cap, max(self.minimum_probability, probability))

    def to_dict(self) -> dict[str, object]:
        return {
            "intercept": self.intercept,
            "coefficients": dict(self.coefficients),
            "feature_means": dict(self.feature_means),
            "feature_scales": dict(self.feature_scales),
            "calibration_intercept": self.calibration_intercept,
            "calibration_slope": self.calibration_slope,
            "minimum_probability": self.minimum_probability,
            "single_answer_cap": self.single_answer_cap,
            "conflict_cap": self.conflict_cap,
            "activation_threshold": self.activation_threshold,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "LearnedMisreportGate":
        def numeric_mapping(name: str) -> dict[str, float]:
            raw = data.get(name, {})
            if not isinstance(raw, Mapping):
                raise ValueError(f"{name} must be a mapping")
            return {str(key): float(value) for key, value in raw.items()}

        return cls(
            intercept=float(data["intercept"]),
            coefficients=numeric_mapping("coefficients"),
            feature_means=numeric_mapping("feature_means"),
            feature_scales=numeric_mapping("feature_scales"),
            calibration_intercept=float(data.get("calibration_intercept", 0.0)),
            calibration_slope=float(data.get("calibration_slope", 1.0)),
            minimum_probability=float(data.get("minimum_probability", 0.005)),
            single_answer_cap=float(data.get("single_answer_cap", 0.15)),
            conflict_cap=float(data.get("conflict_cap", 0.40)),
            activation_threshold=float(data.get("activation_threshold", 0.01)),
        )


@dataclass(frozen=True)
class OracleMisreportGate:
    """Simulation-only ceiling that reads explicitly privileged metadata."""

    positive_probability: float = 0.49
    negative_probability: float = 0.0
    activation_threshold: float = 0.01

    def __post_init__(self) -> None:
        if not 0 <= self.negative_probability <= self.positive_probability < 0.5:
            raise ValueError("oracle probabilities must satisfy 0 <= negative <= positive < 0.5")

    def probability(
        self, tracker: "BeliefTracker", observation: Observation
    ) -> float:
        del tracker
        if not observation.oracle_report_mode:
            raise ValueError("oracle gate requires simulation-only oracle_report_mode")
        return (
            self.positive_probability
            if observation.oracle_report_mode == "misreported"
            else self.negative_probability
        )


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


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
