"""Bayesian disease belief updates under the latent answer channel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .channel import AnswerChannel, ReportMode
from .estimation import DiseaseStateModel
from .gating import MisreportGate
from .schema import UNKNOWN, FeatureKey, Observation


def entropy(distribution: Mapping[str, float]) -> float:
    return -sum(probability * math.log2(probability) for probability in distribution.values() if probability > 0)


@dataclass(frozen=True)
class UpdateResult:
    observation: Observation
    prior: Mapping[str, float]
    posterior: Mapping[str, float]
    disease_likelihoods: Mapping[str, float]
    report_mode_prior: Mapping[ReportMode, float]
    report_mode_posterior: Mapping[ReportMode, float]
    conflicting_turns: tuple[int, ...]

    @property
    def misreport_probability(self) -> float:
        return self.report_mode_posterior[ReportMode.MISREPORTED]

    @property
    def misreport_prior(self) -> float:
        return self.report_mode_prior[ReportMode.MISREPORTED]

    @property
    def information_gain(self) -> float:
        return entropy(self.prior) - entropy(self.posterior)


class BeliefTracker:
    def __init__(
        self,
        model: DiseaseStateModel,
        channel: AnswerChannel | None = None,
        initial_belief: Mapping[str, float] | None = None,
        misreport_gate: MisreportGate | None = None,
    ) -> None:
        self.model = model
        self.channel = channel or AnswerChannel()
        self.misreport_gate = misreport_gate
        self.belief = self._normalize(initial_belief or model.disease_priors)
        if set(self.belief) != set(model.diseases):
            raise ValueError("initial belief keys must match model diseases")
        # P(Z_j | D, reports about j).  The same latent state is retained across
        # repeated questions; it is not re-sampled at every turn.
        self.state_beliefs: dict[str, dict[FeatureKey, dict[str, float]]] = {
            disease: {
                key: model.state_distribution(disease, key) for key in model.specs
            }
            for disease in model.diseases
        }
        self.history: list[UpdateResult] = []

    @staticmethod
    def _normalize(values: Mapping[str, float]) -> dict[str, float]:
        if min(values.values(), default=0.0) < 0:
            raise ValueError("probabilities cannot be negative")
        denominator = sum(values.values())
        if denominator <= 0:
            raise ValueError("probabilities need positive mass")
        return {key: value / denominator for key, value in values.items()}

    def predictive_state_distribution(
        self,
        key: FeatureKey,
        belief: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        disease_belief = belief or self.belief
        spec = self.model.specs[key]
        return {
            state: sum(
                disease_belief[disease]
                * self.state_beliefs[disease][key][state]
                for disease in self.model.diseases
            )
            for state in spec.values
        }

    def observation_likelihoods(
        self,
        observation: Observation,
        mode_prior: Mapping[ReportMode, float] | None = None,
    ) -> dict[str, float]:
        spec = self.model.specs.get(observation.key)
        if spec is None:
            raise KeyError(f"unknown feature: {observation.key.token}")
        if observation.value != UNKNOWN and observation.value not in spec.values:
            raise ValueError(
                f"invalid reported value {observation.value!r} for {observation.key.token}"
            )
        effective_prior = mode_prior or self.report_mode_prior(observation)
        return {
            disease: sum(
                state_probability
                * self.channel.marginal_probability(
                    observation.value,
                    state,
                    spec.values,
                    observation.certainty,
                    effective_prior,
                )
                for state, state_probability in self.state_beliefs[disease][
                    observation.key
                ].items()
            )
            for disease in self.model.diseases
        }

    def posterior_for(
        self,
        observation: Observation,
        prior: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        disease_prior = prior or self.belief
        likelihoods = self.observation_likelihoods(observation)
        return self._normalize(
            {
                disease: disease_prior[disease] * likelihoods[disease]
                for disease in self.model.diseases
            }
        )

    def update(self, observation: Observation) -> UpdateResult:
        prior = dict(self.belief)
        mode_prior = self.report_mode_prior(observation)
        predictive_states = self.predictive_state_distribution(observation.key, prior)
        mode_posterior = self.channel.mode_posterior(
            observation, predictive_states, mode_prior
        )
        likelihoods = self.observation_likelihoods(observation, mode_prior)
        posterior = self._normalize(
            {
                disease: prior[disease] * likelihoods[disease]
                for disease in self.model.diseases
            }
        )
        conflicts = tuple(
            index
            for index, previous in enumerate(self.history)
            if self._is_direct_conflict(previous.observation, observation)
        )
        result = UpdateResult(
            observation=observation,
            prior=prior,
            posterior=posterior,
            disease_likelihoods=likelihoods,
            report_mode_prior=mode_prior,
            report_mode_posterior=mode_posterior,
            conflicting_turns=conflicts,
        )
        spec = self.model.specs[observation.key]
        for disease in self.model.diseases:
            previous_states = self.state_beliefs[disease][observation.key]
            state_weights = {
                state: probability
                * self.channel.marginal_probability(
                    observation.value,
                    state,
                    spec.values,
                    observation.certainty,
                    mode_prior,
                )
                for state, probability in previous_states.items()
            }
            self.state_beliefs[disease][observation.key] = self._normalize(state_weights)
        self.belief = posterior
        self.history.append(result)
        return result

    def report_mode_prior(
        self, observation: Observation
    ) -> dict[ReportMode, float]:
        base = dict(self.channel.parameters.cue_priors[observation.certainty])
        if self.misreport_gate is None:
            return base
        misreport_probability = self.misreport_gate.probability(self, observation)
        retained_mass = sum(
            probability
            for mode, probability in base.items()
            if mode != ReportMode.MISREPORTED
        )
        if retained_mass <= 0:
            raise ValueError("adaptive gate needs positive non-misreport prior mass")
        return {
            mode: (
                misreport_probability
                if mode == ReportMode.MISREPORTED
                else (1.0 - misreport_probability) * probability / retained_mass
            )
            for mode, probability in base.items()
        }

    @staticmethod
    def _is_direct_conflict(left: Observation, right: Observation) -> bool:
        return (
            left.key == right.key
            and left.value != UNKNOWN
            and right.value != UNKNOWN
            and left.value != right.value
        )

    def ranked_diseases(self) -> list[tuple[str, float]]:
        return sorted(self.belief.items(), key=lambda item: item[1], reverse=True)
