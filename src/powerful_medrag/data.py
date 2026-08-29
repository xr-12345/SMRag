"""Adapters for a transparent JSONL training format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from .schema import ClinicalCase, FeatureKey, VariableSpec


def load_specs(path: str | Path) -> list[VariableSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_specs = data["variables"] if isinstance(data, dict) else data
    return [VariableSpec.from_dict(item) for item in raw_specs]


def save_specs(specs: Iterable[VariableSpec], path: str | Path) -> None:
    payload = {"variables": [spec.to_dict() for spec in specs]}
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_jsonl_cases(path: str | Path) -> list[ClinicalCase]:
    """Load cases with ``diagnosis`` and explicit ``states`` mappings.

    A missing state is unknown/unrecorded, not negative.  Contextual features use
    the serialized ``FeatureKey.token`` representation.
    """

    cases: list[ClinicalCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                states = {
                    FeatureKey.from_token(str(token)): str(value)
                    for token, value in data["states"].items()
                }
                cases.append(
                    ClinicalCase(
                        diagnosis=str(data["diagnosis"]),
                        states=states,
                        case_id=str(data.get("case_id", line_number)),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid case at {path}:{line_number}: {exc}") from exc
    return cases


def case_to_dict(case: ClinicalCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "diagnosis": case.diagnosis,
        "states": {key.token: value for key, value in case.states.items()},
    }


def save_jsonl_cases(cases: Iterable[ClinicalCase], path: str | Path) -> None:
    lines = [json.dumps(case_to_dict(case), ensure_ascii=False) for case in cases]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def cases_from_records(records: Iterable[Mapping[str, object]]) -> list[ClinicalCase]:
    """Convenience adapter for already parsed EHR records."""

    cases: list[ClinicalCase] = []
    for index, record in enumerate(records):
        raw_states = record.get("states")
        if not isinstance(raw_states, Mapping):
            raise ValueError(f"record {index} has no states mapping")
        states = {
            key if isinstance(key, FeatureKey) else FeatureKey.from_token(str(key)): str(value)
            for key, value in raw_states.items()
        }
        cases.append(
            ClinicalCase(
                diagnosis=str(record["diagnosis"]),
                states=states,
                case_id=str(record.get("case_id", index)),
            )
        )
    return cases

