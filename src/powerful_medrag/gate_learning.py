"""Dependency-free fitting and calibration for the observable logistic gate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence

from .gating import LEARNED_GATE_FEATURES, LearnedMisreportGate, _sigmoid


@dataclass(frozen=True)
class GateTrainingExample:
    features: Mapping[str, float]
    label: int

    def __post_init__(self) -> None:
        if self.label not in {0, 1}:
            raise ValueError("gate label must be binary")
        missing = set(LEARNED_GATE_FEATURES) - set(self.features)
        if missing:
            raise ValueError(f"missing gate features: {sorted(missing)}")


def examples_from_gate_events(
    events: Iterable[object], *, target: str = "harmful_misreport"
) -> list[GateTrainingExample]:
    allowed_targets = {"latent_misreport", "wrong_report", "harmful_misreport"}
    if target not in allowed_targets:
        raise ValueError(f"unsupported gate target: {target}")
    examples: list[GateTrainingExample] = []
    for event in events:
        features = {
            "surprisal": float(getattr(event, "surprisal")),
            "direct_conflicts": float(getattr(event, "direct_conflicts")),
            "is_uncertain": float(getattr(event, "is_uncertain")),
            "is_unknown": float(getattr(event, "is_unknown")),
            "history_unreliable_fraction": float(
                getattr(event, "history_unreliable_fraction")
            ),
            "top_probability": float(getattr(event, "top_probability")),
            "normalized_belief_entropy": float(
                getattr(event, "normalized_belief_entropy")
            ),
            "diagnostic_impact": float(getattr(event, "diagnostic_impact")),
            "extraction_uncertainty": float(
                getattr(event, "extraction_uncertainty")
            ),
        }
        examples.append(GateTrainingExample(features, int(getattr(event, target))))
    return examples


def fit_logistic_gate(
    training_examples: Iterable[GateTrainingExample],
    *,
    calibration_examples: Iterable[GateTrainingExample] | None = None,
    learning_rate: float = 0.05,
    iterations: int = 2000,
    l2_strength: float = 0.01,
    minimum_probability: float = 0.005,
    single_answer_cap: float = 0.15,
    conflict_cap: float = 0.40,
) -> LearnedMisreportGate:
    """Fit logistic regression and optional held-out Platt calibration.

    The caller owns the data split.  Test examples must never be passed here.
    Features are standardized from the training examples only.
    """

    rows = list(training_examples)
    _validate_training_rows(rows)
    if learning_rate <= 0 or iterations <= 0 or l2_strength < 0:
        raise ValueError("invalid logistic fitting hyperparameters")

    feature_names = LEARNED_GATE_FEATURES
    means = {
        feature: fmean(float(row.features[feature]) for row in rows)
        for feature in feature_names
    }
    scales: dict[str, float] = {}
    for feature in feature_names:
        variance = fmean(
            (float(row.features[feature]) - means[feature]) ** 2 for row in rows
        )
        scales[feature] = max(math.sqrt(variance), 1e-6)
    matrix = [
        [
            (float(row.features[feature]) - means[feature]) / scales[feature]
            for feature in feature_names
        ]
        for row in rows
    ]
    labels = [row.label for row in rows]
    prevalence = min(max(fmean(labels), 1e-6), 1.0 - 1e-6)
    intercept = math.log(prevalence / (1.0 - prevalence))
    weights = [0.0] * len(feature_names)

    for iteration in range(iterations):
        intercept_gradient = 0.0
        gradients = [0.0] * len(weights)
        for vector, label in zip(matrix, labels):
            probability = _sigmoid(
                intercept + sum(weight * value for weight, value in zip(weights, vector))
            )
            residual = probability - label
            intercept_gradient += residual
            for index, value in enumerate(vector):
                gradients[index] += residual * value
        rate = learning_rate / math.sqrt(1.0 + iteration / 100.0)
        intercept -= rate * intercept_gradient / len(rows)
        for index in range(len(weights)):
            regularized = gradients[index] / len(rows) + l2_strength * weights[index]
            weights[index] -= rate * regularized

    calibration_intercept = 0.0
    calibration_slope = 1.0
    if calibration_examples is not None:
        calibration_rows = list(calibration_examples)
        _validate_training_rows(calibration_rows)
        logits = [
            intercept
            + sum(
                weights[index]
                * (float(row.features[feature]) - means[feature])
                / scales[feature]
                for index, feature in enumerate(feature_names)
            )
            for row in calibration_rows
        ]
        calibration_intercept, calibration_slope = _fit_platt_scaling(
            logits,
            [row.label for row in calibration_rows],
        )

    return LearnedMisreportGate(
        intercept=intercept,
        coefficients=dict(zip(feature_names, weights)),
        feature_means=means,
        feature_scales=scales,
        calibration_intercept=calibration_intercept,
        calibration_slope=calibration_slope,
        minimum_probability=minimum_probability,
        single_answer_cap=single_answer_cap,
        conflict_cap=conflict_cap,
    )


def _fit_platt_scaling(
    logits: Sequence[float], labels: Sequence[int]
) -> tuple[float, float]:
    if len(logits) != len(labels) or not logits:
        raise ValueError("calibration logits and labels must be equally non-empty")
    if not any(labels) or all(labels):
        raise ValueError("calibration needs both classes")
    intercept = 0.0
    slope = 1.0
    for iteration in range(1000):
        intercept_gradient = 0.0
        slope_gradient = 0.0
        for logit, label in zip(logits, labels):
            residual = _sigmoid(intercept + slope * logit) - label
            intercept_gradient += residual
            slope_gradient += residual * logit
        rate = 0.05 / math.sqrt(1.0 + iteration / 100.0)
        intercept -= rate * intercept_gradient / len(logits)
        slope = max(0.0, slope - rate * slope_gradient / len(logits))
    return intercept, slope


def _validate_training_rows(rows: Sequence[GateTrainingExample]) -> None:
    if not rows:
        raise ValueError("gate fitting needs examples")
    labels = {row.label for row in rows}
    if labels != {0, 1}:
        raise ValueError("gate fitting needs both classes")
    for row in rows:
        for feature in LEARNED_GATE_FEATURES:
            value = float(row.features[feature])
            if not math.isfinite(value):
                raise ValueError(f"non-finite gate feature: {feature}")


def save_learned_gate(gate: LearnedMisreportGate, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(gate.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_learned_gate(path: str | Path) -> LearnedMisreportGate:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("learned gate file must contain an object")
    return LearnedMisreportGate.from_dict(data)
