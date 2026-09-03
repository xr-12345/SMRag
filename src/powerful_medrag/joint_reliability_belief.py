"""Joint disease + reliability belief under the shared joint report channel.

Phase 8B.  Unlike ``BeliefTracker`` (which updates the disease posterior one
report at a time and models verification through a separate re-ask channel),
this tracker holds one ``ReliabilityMemory`` of per-feature ``ReportBundle``s
and derives **every** quantity from the single ``JointReportChannel``:

    b_t(d)            = P(D = d | H_t)
    p_i^mode          = P(E_i = MISREPORTED | H_t)
    p_i^wrong         = P(Z_i != Y_i | H_t)
    P(Y_i' | H_t, VerifyOld(i))

so the disease posterior, the reliability posterior, and the verification
prediction can no longer disagree with one another.

A feature asked twice contributes ONE joint likelihood factor::

    L_i(d) = sum_z P(z | d) P(y_i, y_i' | z, v)          (verified)
    L_i(d) = sum_z P(z | d) P(y_i | z)                    (single)

and the disease posterior is ``b(d) ∝ P(d) ∏_i L_i(d)``.  Repeated identical
answers are therefore NOT double-counted (a feature is one factor regardless of
how many times it was asked), and conflicting answers are never collapsed to
``UNKNOWN`` -- they are resolved probabilistically by the joint likelihood.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .channel import ReportMode
from .estimation import DiseaseStateModel
from .joint_report_channel import JointReportChannel, VerificationType, joint_answer_space
from .reliability_memory import ReliabilityMemory, ReportBundle
from .schema import UNKNOWN, CertaintyCue, FeatureKey, Observation


class JointReliabilityBeliefTracker:
    """Disease belief + per-feature reliability from one joint channel."""

    def __init__(
        self,
        model: DiseaseStateModel,
        channel: JointReportChannel | None = None,
        initial_belief: Mapping[str, float] | None = None,
        memory: ReliabilityMemory | None = None,
    ) -> None:
        self.model = model
        self.channel = channel or JointReportChannel()
        self.memory = memory or ReliabilityMemory()
        self.belief = self._normalize(initial_belief or model.disease_priors)
        if set(self.belief) != set(model.diseases):
            raise ValueError("initial belief keys must match model diseases")
        # Pure per-bundle likelihood memo.  ``_feature_likelihood`` is a pure
        # function of (disease, bundle) and the frozen channel/model, so it can
        # be cached across the many leave-one-out products computed in one
        # ranking step.  Cleared on every memory mutation.  ReportBundle is a
        # frozen dataclass (hashable), so it is a valid cache key.
        self._feature_lh_cache: dict[tuple[str, ReportBundle], float] = {}
        # ``single_likelihood`` / ``joint_likelihood`` are pure functions of
        # (disease, key, answer value, certainty) and the frozen channel/model.
        # They are recomputed many times per ranking step (once for the AskNew
        # predictive and again for each hypothetical posterior), so caching them
        # removes that redundancy.  Never cleared: no memory dependence.
        self._single_lh_cache: dict[tuple[str, str, str, str], float] = {}

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _normalize(values: Mapping[str, float]) -> dict[str, float]:
        if min(values.values(), default=0.0) < 0:
            raise ValueError("probabilities cannot be negative")
        denominator = sum(values.values())
        if denominator <= 0:
            raise ValueError("probabilities need positive mass")
        return {key: value / denominator for key, value in values.items()}

    def _states(self, key: FeatureKey) -> tuple[str, ...]:
        return self.model.specs[key].values

    # -- per-feature likelihood factors ------------------------------------- #

    def single_likelihood(
        self, disease: str, key: FeatureKey, observation: Observation
    ) -> float:
        """L_i(d) for a feature with a single answer."""
        cache_key = (disease, key.token, observation.value, observation.certainty.value)
        cached = self._single_lh_cache.get(cache_key)
        if cached is not None:
            return cached
        states = self._states(key)
        result = sum(
            state_probability
            * self.channel.single_probability(
                observation.value, state, states, observation.certainty
            )
            for state, state_probability in self.model.state_distribution(
                disease, key
            ).items()
        )
        self._single_lh_cache[cache_key] = result
        return result

    def joint_likelihood(
        self,
        disease: str,
        key: FeatureKey,
        original: Observation,
        verification: Observation,
        verification_type: VerificationType,
    ) -> float:
        """L_i(d) for a feature with a first answer and one re-ask."""
        states = self._states(key)
        return sum(
            state_probability
            * self.channel.joint_probability(
                original.value,
                verification.value,
                state,
                states,
                verification_type,
                original.certainty,
            )
            for state, state_probability in self.model.state_distribution(
                disease, key
            ).items()
        )

    def _feature_likelihood(self, disease: str, bundle: ReportBundle) -> float:
        cache_key = (disease, bundle)
        cached = self._feature_lh_cache.get(cache_key)
        if cached is not None:
            return cached
        if len(bundle.verifications) > 1:
            raise NotImplementedError(
                "joint model supports a single re-ask per feature (maximum_verifications=1)"
            )
        if not bundle.is_verified:
            result = self.single_likelihood(disease, bundle.key, bundle.original)
        else:
            result = self.joint_likelihood(
                disease,
                bundle.key,
                bundle.original,
                bundle.verifications[0],
                bundle.verification_types[0],
            )
        self._feature_lh_cache[cache_key] = result
        return result

    def _feature_contribution(
        self, disease: str, bundle: ReportBundle, mode: ReportMode
    ) -> float:
        """C_i(d, m): the mode-``m`` slice of feature i's likelihood factor."""
        key = bundle.key
        states = self._states(key)
        mode_weight = self.channel.mode_prior(bundle.original.certainty).get(mode, 0.0)
        if mode_weight <= 0:
            return 0.0
        total = 0.0
        for state, state_probability in self.model.state_distribution(
            disease, key
        ).items():
            first = self.channel.channel.probability(
                bundle.original.value, state, states, mode
            )
            if first <= 0:
                continue
            reask = 1.0
            if bundle.is_verified:
                reask = self.channel.reask_conditional_probability(
                    bundle.verifications[0].value,
                    state,
                    mode,
                    states,
                    bundle.verification_types[0],
                )
            total += state_probability * mode_weight * first * reask
        return total

    # -- disease belief ----------------------------------------------------- #

    def _unnormalized_belief(
        self, override: Mapping[FeatureKey, Mapping[str, float]] | None = None
    ) -> dict[str, float]:
        """P(d) ∏_i L_i(d), with optional per-feature likelihood override."""
        override = override or {}
        result = {disease: float(self.model.disease_priors[disease])
                  for disease in self.model.diseases}
        for key, bundle in self.memory.items():
            if key in override:
                likelihoods = override[key]
            else:
                likelihoods = {
                    disease: self._feature_likelihood(disease, bundle)
                    for disease in self.model.diseases
                }
            for disease in self.model.diseases:
                result[disease] *= likelihoods[disease]
        for key, likelihoods in override.items():
            if key not in self.memory:
                for disease in self.model.diseases:
                    result[disease] *= likelihoods[disease]
        return result

    def disease_belief(self) -> dict[str, float]:
        """Recompute ``b_t(d)`` from the current memory and cache it."""
        self.belief = self._normalize(self._unnormalized_belief())
        return self.belief

    def ranked_diseases(self) -> list[tuple[str, float]]:
        self.disease_belief()
        return sorted(self.belief.items(), key=lambda item: item[1], reverse=True)

    def _leave_one_out(self, key: FeatureKey) -> dict[str, float]:
        """∏_{j != key} L_j(d) for each disease."""
        leave = {disease: 1.0 for disease in self.model.diseases}
        for other, bundle in self.memory.items():
            if other == key:
                continue
            for disease in self.model.diseases:
                leave[disease] *= self._feature_likelihood(disease, bundle)
        return leave

    # -- reliability posteriors --------------------------------------------- #

    def state_posterior(self, key: FeatureKey) -> dict[str, float]:
        """P(Z_i = z | H_t) for the feature's true clinical state."""
        bundle = self.memory.get(key)
        if bundle is None:
            return dict(
                (state, sum(
                    self.belief[disease]
                    * self.model.state_distribution(disease, key)[state]
                    for disease in self.model.diseases
                ))
                for state in self._states(key)
            )
        leave = self._leave_one_out(key)
        states = self._states(key)
        unnormalized: dict[str, float] = {}
        for state in states:
            if bundle.is_verified:
                phi = self.channel.joint_probability(
                    bundle.original.value,
                    bundle.verifications[0].value,
                    state,
                    states,
                    bundle.verification_types[0],
                    bundle.original.certainty,
                )
            else:
                phi = self.channel.single_probability(
                    bundle.original.value, state, states, bundle.original.certainty
                )
            unnormalized[state] = phi * sum(
                self.model.disease_priors[disease]
                * self.model.state_distribution(disease, key)[state]
                * leave[disease]
                for disease in self.model.diseases
            )
        return self._normalize(unnormalized)

    def mode_posterior(self, key: FeatureKey) -> dict[ReportMode, float]:
        """P(E_i = e | H_t) for the first answer's latent report mode."""
        bundle = self.memory.get(key)
        if bundle is None:
            return self.channel.mode_prior(CertaintyCue.NONE)
        leave = self._leave_one_out(key)
        unnormalized: dict[ReportMode, float] = {}
        for mode in ReportMode:
            unnormalized[mode] = sum(
                self.model.disease_priors[disease]
                * self._feature_contribution(disease, bundle, mode)
                * leave[disease]
                for disease in self.model.diseases
            )
        return self._normalize(unnormalized)

    def p_mode_misreported(self, key: FeatureKey) -> float:
        """p_i^mode = P(E_i = MISREPORTED | H_t)."""
        return self.mode_posterior(key)[ReportMode.MISREPORTED]

    def p_wrong(self, key: FeatureKey) -> float:
        """p_i^wrong = P(Z_i != Y_i | H_t) for the first answer ``Y_i``."""
        bundle = self.memory.get(key)
        if bundle is None:
            return 0.0
        original_value = bundle.original.value
        if original_value == UNKNOWN:
            return 1.0  # UNKNOWN is never the true state, so it is always "wrong"
        return 1.0 - self.state_posterior(key).get(original_value, 0.0)

    # -- predictive distributions ------------------------------------------- #

    def single_answer_predictive(self, key: FeatureKey) -> dict[str, float]:
        """P(y_j | b_t) over ``(*values, UNKNOWN)`` for a not-yet-asked feature."""
        states = self._states(key)
        answer_space = joint_answer_space(states)
        distribution: dict[str, float] = {}
        for answer in answer_space:
            distribution[answer] = sum(
                self.belief[disease]
                * self.single_likelihood(
                    disease, key, Observation(key, answer, CertaintyCue.NONE)
                )
                for disease in self.model.diseases
            )
        return self._normalize(distribution)

    def reask_predictive(self, key: FeatureKey) -> dict[str, float]:
        """P(y' | H_t, VerifyOld(i)) = P(y' | H_t, Y_i, v) over the answer space.

        The re-ask distribution conditions on the observed first answer ``Y_i``
        through the posterior over (d, z, e), then transitions e -> e' with
        ``T_v`` before sampling y'.  This is the shared channel's VerifyOld
        prediction, so it can no longer disagree with the disease posterior.
        """
        bundle = self.memory.get(key)
        if bundle is None:
            raise KeyError(f"cannot predict a re-ask for unseen feature {key.token}")
        if bundle.is_verified:
            raise ValueError(f"feature {key.token} is already verified")
        states = self._states(key)
        leave = self._leave_one_out(key)
        answer_space = joint_answer_space(states)

        unnormalized: dict[str, float] = {}
        for y_prime in answer_space:
            total = 0.0
            for disease in self.model.diseases:
                base = self.model.disease_priors[disease] * leave[disease]
                if base <= 0:
                    continue
                for state, state_probability in self.model.state_distribution(
                    disease, key
                ).items():
                    for mode, mode_weight in self.channel.mode_prior(
                        bundle.original.certainty
                    ).items():
                        first = self.channel.channel.probability(
                            bundle.original.value, state, states, mode
                        )
                        if first <= 0:
                            continue
                        reask = self.channel.reask_conditional_probability(
                            y_prime, state, mode, states,
                            VerificationType.REPEAT,
                        )
                        total += (
                            base
                            * state_probability
                            * mode_weight
                            * first
                            * reask
                        )
            unnormalized[y_prime] = total
        return self._normalize(unnormalized)

    # -- hypothetical posterior updates ------------------------------------- #

    def posterior_after_single(
        self, key: FeatureKey, observation: Observation
    ) -> dict[str, float]:
        """Disease posterior after observing ``observation`` as a first answer."""
        likelihoods = {
            disease: self.single_likelihood(disease, key, observation)
            for disease in self.model.diseases
        }
        return self._normalize(self._unnormalized_belief({key: likelihoods}))

    def posterior_after_verification(
        self, key: FeatureKey, verification: Observation
    ) -> dict[str, float]:
        """Disease posterior after a re-ask ``verification`` for feature ``key``."""
        bundle = self.memory.get(key)
        if bundle is None:
            raise KeyError(f"cannot verify unseen feature {key.token}")
        likelihoods = {
            disease: self.joint_likelihood(
                disease, key, bundle.original, verification, VerificationType.REPEAT
            )
            for disease in self.model.diseases
        }
        return self._normalize(self._unnormalized_belief({key: likelihoods}))

    # -- dialogue updates --------------------------------------------------- #

    def observe_single(self, observation: Observation) -> None:
        """Record a first answer and refresh the disease belief."""
        self._feature_lh_cache.clear()
        self.memory.observe_single(observation)
        self.disease_belief()

    def observe_verification(
        self,
        key: FeatureKey,
        observation: Observation,
        verification_type: VerificationType = VerificationType.REPEAT,
    ) -> None:
        """Record a re-ask (no UNKNOWN compression) and refresh the belief."""
        self._feature_lh_cache.clear()
        self.memory.observe_verification(key, observation, verification_type)
        self.disease_belief()
