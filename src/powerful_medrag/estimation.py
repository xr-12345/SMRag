"""Estimate P(true clinical state | disease) from training cases."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .schema import UNKNOWN, ClinicalCase, FeatureKey, VariableSpec


@dataclass(frozen=True)
class StateEstimate:
    probabilities: Mapping[str, float]
    counts: Mapping[str, int]
    observed_count: int


class DiseaseStateModel:
    """Smoothed categorical distributions for latent clinical variables.

    For a binary symptom, this is the familiar Beta--Bernoulli estimate.  For a
    multi-valued variable it is a Dirichlet--Categorical estimate.  Only
    explicitly observed states contribute to the denominator.
    """

    def __init__(
        self,
        *,
        specs: Iterable[VariableSpec],
        diseases: Iterable[str],
        disease_priors: Mapping[str, float],
        estimates: Mapping[str, Mapping[FeatureKey, StateEstimate]],
        alpha: float,
        hierarchical_strength: float,
        disease_counts: Mapping[str, int] | None = None,
    ) -> None:
        self.specs = {spec.key: spec for spec in specs}
        self.diseases = tuple(sorted(diseases))
        self.disease_priors = dict(disease_priors)
        self.estimates = {
            disease: dict(variable_estimates)
            for disease, variable_estimates in estimates.items()
        }
        self.alpha = float(alpha)
        self.hierarchical_strength = float(hierarchical_strength)
        self.disease_counts = dict(disease_counts or {})
        self._validate()

    def _validate(self) -> None:
        if not self.diseases:
            raise ValueError("model needs at least one disease")
        if set(self.disease_priors) != set(self.diseases):
            raise ValueError("disease prior keys do not match diseases")
        if abs(sum(self.disease_priors.values()) - 1.0) > 1e-8:
            raise ValueError("disease priors must sum to one")
        for disease in self.diseases:
            for key, spec in self.specs.items():
                estimate = self.estimates[disease][key]
                if set(estimate.probabilities) != set(spec.values):
                    raise ValueError("state probability keys do not match variable spec")
                if abs(sum(estimate.probabilities.values()) - 1.0) > 1e-8:
                    raise ValueError("state probabilities must sum to one")

    @classmethod
    def fit(
        cls,
        cases: Iterable[ClinicalCase],
        specs: Iterable[VariableSpec],
        *,
        alpha: float = 1.0,
        disease_prior_alpha: float = 1.0,
        hierarchical_strength: float = 0.0,
    ) -> "DiseaseStateModel":
        """Fit disease priors and latent-state conditional distributions.

        ``alpha=1`` gives Laplace smoothing.  ``hierarchical_strength`` adds
        global feature frequencies as empirical-Bayes pseudo-counts, which is
        useful for rare diseases.
        """

        if alpha <= 0 or disease_prior_alpha <= 0:
            raise ValueError("smoothing parameters must be positive")
        if hierarchical_strength < 0:
            raise ValueError("hierarchical_strength cannot be negative")

        spec_list = list(specs)
        if not spec_list:
            raise ValueError("cannot fit without variable specs")
        spec_map = {spec.key: spec for spec in spec_list}

        local_counts: dict[str, dict[FeatureKey, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        disease_counts: Counter[str] = Counter()
        case_count = 0
        for case in cases:
            case_count += 1
            disease_counts[case.diagnosis] += 1
            for key, state in case.states.items():
                if key not in spec_map:
                    raise ValueError(
                        f"case {case.case_id or '<unknown>'} contains unspecified feature {key.token}"
                    )
                if state == UNKNOWN:
                    continue
                if state not in spec_map[key].values:
                    raise ValueError(
                        f"invalid state {state!r} for feature {key.display_name()}"
                    )
                local_counts[case.diagnosis][key][state] += 1

        if case_count == 0:
            raise ValueError("cannot fit an empty case collection")
        return cls.from_counts(
            specs=spec_list,
            disease_counts=disease_counts,
            state_counts=local_counts,
            alpha=alpha,
            disease_prior_alpha=disease_prior_alpha,
            hierarchical_strength=hierarchical_strength,
        )

    @classmethod
    def from_counts(
        cls,
        *,
        specs: Iterable[VariableSpec],
        disease_counts: Mapping[str, int],
        state_counts: Mapping[
            str, Mapping[FeatureKey, Mapping[str, int]]
        ],
        alpha: float = 1.0,
        disease_prior_alpha: float = 1.0,
        hierarchical_strength: float = 0.0,
    ) -> "DiseaseStateModel":
        """Build a model from sufficient statistics without materializing cases."""

        if alpha <= 0 or disease_prior_alpha <= 0:
            raise ValueError("smoothing parameters must be positive")
        if hierarchical_strength < 0:
            raise ValueError("hierarchical_strength cannot be negative")
        spec_list = list(specs)
        if not spec_list:
            raise ValueError("cannot fit without variable specs")
        if not disease_counts or min(disease_counts.values()) <= 0:
            raise ValueError("disease counts must be positive")

        diseases = sorted(disease_counts)
        case_count = sum(disease_counts.values())
        diseases = sorted(disease_counts)
        denominator = case_count + disease_prior_alpha * len(diseases)
        disease_priors = {
            disease: (disease_counts[disease] + disease_prior_alpha) / denominator
            for disease in diseases
        }

        global_counts: dict[FeatureKey, Counter[str]] = defaultdict(Counter)
        for disease in diseases:
            disease_state_counts = state_counts.get(disease, {})
            for spec in spec_list:
                raw_counts = disease_state_counts.get(spec.key, {})
                for value in spec.values:
                    count = int(raw_counts.get(value, 0))
                    if count < 0:
                        raise ValueError("state counts cannot be negative")
                    global_counts[spec.key][value] += count

        estimates: dict[str, dict[FeatureKey, StateEstimate]] = {}
        for disease in diseases:
            estimates[disease] = {}
            for spec in spec_list:
                values = spec.values
                global_total = sum(global_counts[spec.key][value] for value in values)
                global_denominator = global_total + alpha * len(values)
                global_probabilities = {
                    value: (global_counts[spec.key][value] + alpha)
                    / global_denominator
                    for value in values
                }
                counts = {
                    value: int(
                        state_counts.get(disease, {}).get(spec.key, {}).get(value, 0)
                    )
                    for value in values
                }
                observed_count = sum(counts.values())
                probability_denominator = (
                    observed_count + alpha * len(values) + hierarchical_strength
                )
                probabilities = {
                    value: (
                        counts[value]
                        + alpha
                        + hierarchical_strength * global_probabilities[value]
                    )
                    / probability_denominator
                    for value in values
                }
                estimates[disease][spec.key] = StateEstimate(
                    probabilities=probabilities,
                    counts=counts,
                    observed_count=observed_count,
                )

        return cls(
            specs=spec_list,
            diseases=diseases,
            disease_priors=disease_priors,
            estimates=estimates,
            alpha=alpha,
            hierarchical_strength=hierarchical_strength,
            disease_counts=disease_counts,
        )

    def state_distribution(self, disease: str, key: FeatureKey) -> dict[str, float]:
        try:
            return dict(self.estimates[disease][key].probabilities)
        except KeyError as exc:
            raise KeyError(f"unknown disease/feature pair: {disease!r}, {key.token!r}") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "alpha": self.alpha,
            "hierarchical_strength": self.hierarchical_strength,
            "disease_priors": self.disease_priors,
            "disease_counts": self.disease_counts,
            "specs": [spec.to_dict() for spec in self.specs.values()],
            "estimates": {
                disease: {
                    key.token: {
                        "probabilities": dict(estimate.probabilities),
                        "counts": dict(estimate.counts),
                        "observed_count": estimate.observed_count,
                    }
                    for key, estimate in variable_estimates.items()
                }
                for disease, variable_estimates in self.estimates.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DiseaseStateModel":
        specs = [VariableSpec.from_dict(item) for item in data["specs"]]  # type: ignore[arg-type]
        raw_estimates = data["estimates"]  # type: ignore[assignment]
        estimates: dict[str, dict[FeatureKey, StateEstimate]] = {}
        for disease, disease_estimates in raw_estimates.items():  # type: ignore[union-attr]
            estimates[str(disease)] = {}
            for token, raw in disease_estimates.items():
                estimates[str(disease)][FeatureKey.from_token(str(token))] = StateEstimate(
                    probabilities={
                        str(key): float(value)
                        for key, value in raw["probabilities"].items()
                    },
                    counts={
                        str(key): int(value) for key, value in raw["counts"].items()
                    },
                    observed_count=int(raw["observed_count"]),
                )
        disease_priors = {
            str(key): float(value)
            for key, value in data["disease_priors"].items()  # type: ignore[union-attr]
        }
        disease_counts = {
            str(key): int(value)
            for key, value in data.get("disease_counts", {}).items()  # type: ignore[union-attr]
        }
        return cls(
            specs=specs,
            diseases=disease_priors,
            disease_priors=disease_priors,
            estimates=estimates,
            alpha=float(data["alpha"]),
            hierarchical_strength=float(data["hierarchical_strength"]),
            disease_counts=disease_counts,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "DiseaseStateModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
