"""A small reproducible problem used by the demo and tests."""

from __future__ import annotations

import random

from .schema import ClinicalCase, FeatureKey, VariableSpec, make_binary_spec


TOY_PROBABILITIES: dict[str, dict[str, float]] = {
    "influenza": {
        "fever": 0.88,
        "dry_cough": 0.78,
        "myalgia": 0.82,
        "runny_nose": 0.32,
        "itchy_eyes": 0.05,
        "sore_throat": 0.48,
    },
    "common_cold": {
        "fever": 0.24,
        "dry_cough": 0.42,
        "myalgia": 0.18,
        "runny_nose": 0.86,
        "itchy_eyes": 0.16,
        "sore_throat": 0.70,
    },
    "allergic_rhinitis": {
        "fever": 0.03,
        "dry_cough": 0.16,
        "myalgia": 0.04,
        "runny_nose": 0.90,
        "itchy_eyes": 0.86,
        "sore_throat": 0.12,
    },
}


def toy_specs() -> list[VariableSpec]:
    questions = {
        "fever": "是否发热？",
        "dry_cough": "是否有干咳？",
        "myalgia": "是否有明显肌肉酸痛？",
        "runny_nose": "是否流鼻涕？",
        "itchy_eyes": "眼睛是否发痒？",
        "sore_throat": "是否咽痛？",
    }
    return [
        make_binary_spec(name, question=question)
        for name, question in questions.items()
    ]


def generate_toy_cases(
    *,
    cases_per_disease: int = 160,
    missing_rate: float = 0.08,
    seed: int = 7,
) -> tuple[list[ClinicalCase], list[VariableSpec]]:
    rng = random.Random(seed)
    specs = toy_specs()
    cases: list[ClinicalCase] = []
    for disease, probabilities in TOY_PROBABILITIES.items():
        for case_index in range(cases_per_disease):
            states: dict[FeatureKey, str] = {}
            for spec in specs:
                if rng.random() < missing_rate:
                    continue
                probability = probabilities[spec.key.name]
                states[spec.key] = "present" if rng.random() < probability else "absent"
            cases.append(
                ClinicalCase(
                    diagnosis=disease,
                    states=states,
                    case_id=f"{disease}-{case_index}",
                )
            )
    return cases, specs

