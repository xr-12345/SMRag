"""Expected-information-gain question selection."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from collections import Counter
from typing import Mapping, Iterable

from .belief import BeliefTracker, entropy
from .schema import UNKNOWN, CertaintyCue, FeatureKey, Observation


@dataclass(frozen=True)
class QuestionScore:
    key: FeatureKey
    expected_information_gain: float
    cost: float
    utility: float
    predicted_answers: dict[str, float]


class QuestionSelector:
    """Ranks questions by expected disease-entropy reduction per unit cost."""

    def __init__(self, *, cost_power: float = 1.0) -> None:
        if cost_power < 0:
            raise ValueError("cost_power cannot be negative")
        self.cost_power = cost_power
        self._likelihood_cache: dict[
            tuple[int, int, FeatureKey, str], dict[str, float]
        ] = {}

    def _cached_likelihoods(
        self,
        tracker: BeliefTracker,
        key: FeatureKey,
        answer: str,
    ) -> dict[str, float]:
        if tracker.misreport_gate is not None:
            return tracker.observation_likelihoods(
                Observation(
                    key=key,
                    value=answer,
                    certainty=CertaintyCue.NONE,
                )
            )
        cache_key = (id(tracker.model), id(tracker.channel.parameters), key, answer)
        if cache_key not in self._likelihood_cache:
            observation = Observation(
                key=key,
                value=answer,
                certainty=CertaintyCue.NONE,
            )
            self._likelihood_cache[cache_key] = tracker.observation_likelihoods(
                observation
            )
        return self._likelihood_cache[cache_key]

    def score(self, tracker: BeliefTracker, key: FeatureKey) -> QuestionScore:
        spec = tracker.model.specs[key]
        current_entropy = entropy(tracker.belief)
        answer_space = (*spec.values, UNKNOWN)

        predicted_answers: dict[str, float] = {}
        posteriors: dict[str, dict[str, float]] = {}
        for answer in answer_space:
            likelihoods = self._cached_likelihoods(tracker, key, answer)
            answer_probability = sum(
                tracker.belief[disease] * likelihoods[disease]
                for disease in tracker.model.diseases
            )
            predicted_answers[answer] = answer_probability
            if answer_probability > 0:
                posteriors[answer] = {
                    disease: tracker.belief[disease]
                    * likelihoods[disease]
                    / answer_probability
                    for disease in tracker.model.diseases
                }

        total_answer_probability = sum(predicted_answers.values())
        if total_answer_probability <= 0:
            expected_information_gain = 0.0
        else:
            predicted_answers = {
                answer: probability / total_answer_probability
                for answer, probability in predicted_answers.items()
            }
            expected_entropy = sum(
                predicted_answers[answer] * entropy(posteriors[answer])
                for answer in posteriors
            )
            expected_information_gain = max(0.0, current_entropy - expected_entropy)

        utility = expected_information_gain / (spec.cost**self.cost_power)
        return QuestionScore(
            key=key,
            expected_information_gain=expected_information_gain,
            cost=spec.cost,
            utility=utility,
            predicted_answers=predicted_answers,
        )

    def rank(
        self,
        tracker: BeliefTracker,
        *,
        excluded: set[FeatureKey] | None = None,
    ) -> list[QuestionScore]:
        excluded_keys = excluded or set()
        scores = [
            self.score(tracker, key)
            for key in tracker.model.specs
            if key not in excluded_keys and _question_is_available(tracker, key)
        ]
        return sorted(
            scores,
            key=lambda score: (
                score.utility,
                score.expected_information_gain,
                score.key.token,
            ),
            reverse=True,
        )


class FixedOrderQuestionSelector(QuestionSelector):
    """Non-adaptive baseline that asks variables in a predefined order."""

    def __init__(self, order: tuple[FeatureKey, ...]) -> None:
        super().__init__()
        self.order = order

    def rank(
        self,
        tracker: BeliefTracker,
        *,
        excluded: set[FeatureKey] | None = None,
    ) -> list[QuestionScore]:
        excluded_keys = excluded or set()
        position = {key: index for index, key in enumerate(self.order)}
        keys = [
            key
            for key in tracker.model.specs
            if key not in excluded_keys and _question_is_available(tracker, key)
        ]
        keys.sort(key=lambda key: (position.get(key, len(position)), key.token))
        # Keep the true EIG for analysis, but make the baseline's continuation
        # independent of the EIG-specific minimum-utility stopping rule.
        return [replace(self.score(tracker, key), utility=1.0) for key in keys]


class PrevalenceQuestionSelector(QuestionSelector):
    """A static disease-graph-degree proxy baseline.

    For binary variables, degree is approximated by mean symptom prevalence over
    diseases.  It is intentionally case-independent, unlike EIG.
    """

    def rank(
        self,
        tracker: BeliefTracker,
        *,
        excluded: set[FeatureKey] | None = None,
    ) -> list[QuestionScore]:
        excluded_keys = excluded or set()
        scored: list[tuple[float, QuestionScore]] = []
        for key, spec in tracker.model.specs.items():
            if key in excluded_keys or not _question_is_available(tracker, key):
                continue
            baseline_state = "absent" if "absent" in spec.values else spec.values[0]
            prevalence = sum(
                1.0
                - tracker.model.state_distribution(disease, key)[baseline_state]
                for disease in tracker.model.diseases
            ) / len(tracker.model.diseases)
            scored.append(
                (prevalence, replace(self.score(tracker, key), utility=max(prevalence, 1e-3)))
            )
        scored.sort(key=lambda item: (item[0], item[1].key.token), reverse=True)
        return [score for _, score in scored]


class NumpyQuestionSelector(QuestionSelector):
    """Numerically equivalent EIG selector optimized for cohort experiments."""

    def __init__(self, *, cost_power: float = 1.0) -> None:
        super().__init__(cost_power=cost_power)
        self._numpy_model_id: int | None = None
        self._numpy_channel_id: int | None = None
        self._numpy_cache: dict[FeatureKey, tuple[tuple[str, ...], object]] = {}

    def _ensure_numpy_cache(self, tracker: BeliefTracker) -> None:
        if (
            self._numpy_model_id == id(tracker.model)
            and self._numpy_channel_id == id(tracker.channel.parameters)
        ):
            return
        import numpy as np

        self._numpy_cache = {}
        for key, spec in tracker.model.specs.items():
            if not spec.askable:
                continue
            answer_space = (*spec.values, UNKNOWN)
            state_probabilities = np.array(
                [
                    [
                        tracker.model.state_distribution(disease, key)[state]
                        for state in spec.values
                    ]
                    for disease in tracker.model.diseases
                ],
                dtype=float,
            )
            channel_matrix = np.array(
                [
                    [
                        tracker.channel.marginal_probability(
                            answer,
                            state,
                            spec.values,
                            CertaintyCue.NONE,
                        )
                        for answer in answer_space
                    ]
                    for state in spec.values
                ],
                dtype=float,
            )
            self._numpy_cache[key] = (answer_space, state_probabilities @ channel_matrix)
        self._numpy_model_id = id(tracker.model)
        self._numpy_channel_id = id(tracker.channel.parameters)

    def rank(
        self,
        tracker: BeliefTracker,
        *,
        excluded: set[FeatureKey] | None = None,
    ) -> list[QuestionScore]:
        import numpy as np

        self._ensure_numpy_cache(tracker)
        excluded_keys = excluded or set()
        available = [
            key
            for key in tracker.model.specs
            if key not in excluded_keys and _question_is_available(tracker, key)
        ]
        if not available:
            return []
        belief = np.array(
            [tracker.belief[disease] for disease in tracker.model.diseases], dtype=float
        )
        current_entropy = entropy(tracker.belief)
        groups: dict[int, list[FeatureKey]] = {}
        for key in available:
            answer_space, _ = self._numpy_cache[key]
            groups.setdefault(len(answer_space), []).append(key)

        scores: list[QuestionScore] = []
        for keys in groups.values():
            likelihoods = np.stack([self._numpy_cache[key][1] for key in keys], axis=0)
            answer_probabilities = np.einsum("d,fdo->fo", belief, likelihoods)
            weighted = belief[None, :, None] * likelihoods
            with np.errstate(divide="ignore", invalid="ignore"):
                posteriors = np.divide(
                    weighted,
                    answer_probabilities[:, None, :],
                    out=np.zeros_like(weighted),
                    where=answer_probabilities[:, None, :] > 0,
                )
                logs = np.log2(
                    posteriors,
                    out=np.zeros_like(posteriors),
                    where=posteriors > 0,
                )
            posterior_entropies = -(posteriors * logs).sum(axis=1)
            information_gains = np.maximum(
                0.0,
                current_entropy - (answer_probabilities * posterior_entropies).sum(axis=1),
            )
            for index, key in enumerate(keys):
                spec = tracker.model.specs[key]
                answer_space, _ = self._numpy_cache[key]
                information_gain = float(information_gains[index])
                scores.append(
                    QuestionScore(
                        key=key,
                        expected_information_gain=information_gain,
                        cost=spec.cost,
                        utility=information_gain / (spec.cost**self.cost_power),
                        predicted_answers={
                            answer: float(probability)
                            for answer, probability in zip(
                                answer_space, answer_probabilities[index]
                            )
                        },
                    )
                )
        return sorted(
            scores,
            key=lambda score: (
                score.utility,
                score.expected_information_gain,
                score.key.token,
            ),
            reverse=True,
        )

    def score(self, tracker: BeliefTracker, key: FeatureKey) -> QuestionScore:
        import numpy as np

        self._ensure_numpy_cache(tracker)
        spec = tracker.model.specs[key]
        answer_space, raw_likelihoods = self._numpy_cache[key]
        likelihoods = raw_likelihoods
        belief = np.array(
            [tracker.belief[disease] for disease in tracker.model.diseases], dtype=float
        )
        answer_probabilities = belief @ likelihoods
        weighted = belief[:, None] * likelihoods
        with np.errstate(divide="ignore", invalid="ignore"):
            posteriors = np.divide(
                weighted,
                answer_probabilities[None, :],
                out=np.zeros_like(weighted),
                where=answer_probabilities[None, :] > 0,
            )
            log_posteriors = np.log2(
                posteriors,
                out=np.zeros_like(posteriors),
                where=posteriors > 0,
            )
        posterior_entropies = -(posteriors * log_posteriors).sum(axis=0)
        current_entropy = entropy(tracker.belief)
        expected_entropy = float(answer_probabilities @ posterior_entropies)
        information_gain = max(0.0, current_entropy - expected_entropy)
        utility = information_gain / (spec.cost**self.cost_power)
        return QuestionScore(
            key=key,
            expected_information_gain=information_gain,
            cost=spec.cost,
            utility=utility,
            predicted_answers={
                answer: float(probability)
                for answer, probability in zip(answer_space, answer_probabilities)
            },
        )


class NumpyPrevalenceQuestionSelector(NumpyQuestionSelector):
    """Static prevalence baseline with vectorized diagnostic EIG reporting."""

    def rank(
        self,
        tracker: BeliefTracker,
        *,
        excluded: set[FeatureKey] | None = None,
    ) -> list[QuestionScore]:
        excluded_keys = excluded or set()
        candidates: list[tuple[float, FeatureKey]] = []
        for key, spec in tracker.model.specs.items():
            if key in excluded_keys or not _question_is_available(tracker, key):
                continue
            baseline_state = "absent" if "absent" in spec.values else spec.values[0]
            prevalence = sum(
                1.0
                - tracker.model.state_distribution(disease, key)[baseline_state]
                for disease in tracker.model.diseases
            ) / len(tracker.model.diseases)
            candidates.append((prevalence, key))
        if not candidates:
            return []
        prevalence, key = max(candidates, key=lambda item: (item[0], item[1].token))
        return [replace(self.score(tracker, key), utility=max(prevalence, 1e-3))]


class MedRAGReciprocalDegreeSelector(NumpyQuestionSelector):
    """Structured adaptation of MedRAG Eq. 15 reciprocal degree centrality.

    Candidate manifestations are the union of features linked to the current
    top-k disease hypotheses.  Their discriminability is exactly
    ``(n - 1) / degree(feature)`` on the supplied disease--feature graph.
    """

    def __init__(
        self,
        disease_to_features: Mapping[str, Iterable[FeatureKey]],
        *,
        candidate_disease_count: int = 5,
    ) -> None:
        super().__init__()
        if candidate_disease_count <= 0:
            raise ValueError("candidate_disease_count must be positive")
        self.disease_to_features = {
            disease: set(features) for disease, features in disease_to_features.items()
        }
        self.candidate_disease_count = candidate_disease_count
        self.feature_degrees: Counter[FeatureKey] = Counter(
            feature
            for features in self.disease_to_features.values()
            for feature in features
        )
        self.feature_node_count = len(self.feature_degrees)
        if self.feature_node_count == 0:
            raise ValueError("disease--feature graph has no feature nodes")

    def rank(
        self,
        tracker: BeliefTracker,
        *,
        excluded: set[FeatureKey] | None = None,
    ) -> list[QuestionScore]:
        excluded_keys = excluded or set()
        candidate_diseases = [
            disease
            for disease, _ in tracker.ranked_diseases()[: self.candidate_disease_count]
        ]
        candidate_features = set().union(
            *(self.disease_to_features.get(disease, set()) for disease in candidate_diseases)
        )
        candidates: list[tuple[float, FeatureKey]] = []
        for key in candidate_features:
            if (
                key in tracker.model.specs
                and key not in excluded_keys
                and _question_is_available(tracker, key)
            ):
                degree = self.feature_degrees[key]
                discriminability = (self.feature_node_count - 1) / degree
                candidates.append((discriminability, key))
        if not candidates:
            return []
        discriminability, key = max(
            candidates, key=lambda item: (item[0], item[1].token)
        )
        return [replace(self.score(tracker, key), utility=discriminability)]


def _question_is_available(tracker: BeliefTracker, key: FeatureKey) -> bool:
    spec = tracker.model.specs[key]
    if not spec.askable:
        return False
    prerequisite = spec.prerequisite
    if prerequisite is None:
        return True
    for update in reversed(tracker.history):
        observation = update.observation
        if observation.key == prerequisite:
            return observation.value == "present"
    return False
