"""Streaming adapter for the official DDXPlus CSV/JSON release."""

from __future__ import annotations

import ast
import csv
import json
import random
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterable, Iterator, Mapping

from .schema import (
    CertaintyCue,
    ClinicalCase,
    FeatureKey,
    Observation,
    VariableSpec,
    make_binary_spec,
)
from .estimation import DiseaseStateModel


SEPARATOR = "_@_"


def _load_evidence_metadata(path: str | Path) -> dict[str, dict[str, object]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries: Iterable[tuple[str | None, Mapping[str, object]]]
    if isinstance(raw, dict):
        entries = (
            (str(key), value) for key, value in raw.items() if isinstance(value, Mapping)
        )
    elif isinstance(raw, list):
        entries = (
            (None, value) for value in raw if isinstance(value, Mapping)
        )
    else:
        raise ValueError("DDXPlus evidence metadata must be a JSON object or list")

    metadata: dict[str, dict[str, object]] = {}
    for fallback_name, entry in entries:
        name = str(entry.get("name", fallback_name or ""))
        if not name:
            raise ValueError("DDXPlus evidence entry has no name")
        metadata[name] = dict(entry)
    return metadata


def ddxplus_specs(evidence_json: str | Path) -> list[VariableSpec]:
    """Create binary/categorical/atomized-multichoice variable specs."""

    metadata = _load_evidence_metadata(evidence_json)
    specs: list[VariableSpec] = []
    for name, entry in metadata.items():
        data_type = str(entry.get("data_type", "B"))
        question = str(entry.get("question_en") or entry.get("question_fr") or name)
        code_question = str(entry.get("code_question", name))
        prerequisite = FeatureKey(code_question) if code_question != name else None
        if data_type == "B":
            base = make_binary_spec(name, question=question)
            specs.append(
                VariableSpec(
                    key=base.key,
                    values=base.values,
                    question=base.question,
                    cost=base.cost,
                    prerequisite=prerequisite,
                )
            )
            continue

        possible_values = tuple(str(value) for value in entry.get("possible-values", ()))
        if data_type == "C":
            if len(possible_values) < 2:
                raise ValueError(f"categorical evidence {name} needs possible-values")
            specs.append(
                VariableSpec(
                    key=FeatureKey(name),
                    values=possible_values,
                    question=question,
                    prerequisite=prerequisite,
                )
            )
            continue

        if data_type == "M":
            default_value = str(entry.get("default_value", ""))
            value_meaning = entry.get("value_meaning", {})
            for value in possible_values:
                if value == default_value:
                    continue
                meaning = value
                if isinstance(value_meaning, Mapping):
                    raw_meaning = value_meaning.get(value)
                    if isinstance(raw_meaning, Mapping):
                        meaning = str(raw_meaning.get("en") or raw_meaning.get("fr") or value)
                base = make_binary_spec(
                    name,
                    context={"option": value},
                    question=f"{question} [{meaning}]",
                )
                specs.append(
                    VariableSpec(
                        key=base.key,
                        values=base.values,
                        question=base.question,
                        cost=base.cost,
                        prerequisite=prerequisite,
                        askable=False,
                    )
                )
            continue
        raise ValueError(f"unsupported DDXPlus data_type {data_type!r} for {name}")
    return specs


def load_ddxplus_condition_graph(
    condition_json: str | Path,
    *,
    available_features: Iterable[FeatureKey] | None = None,
) -> dict[str, set[FeatureKey]]:
    """Load disease--evidence edges used by the structured MedRAG-RDC baseline."""

    raw = json.loads(Path(condition_json).read_text(encoding="utf-8"))
    entries = raw.values() if isinstance(raw, dict) else raw
    allowed = set(available_features) if available_features is not None else None
    graph: dict[str, set[FeatureKey]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        disease = str(entry["condition_name"])
        evidence_names = set()
        for field in ("symptoms", "antecedents"):
            values = entry.get(field, {})
            if isinstance(values, Mapping):
                evidence_names.update(str(name) for name in values)
        features = {FeatureKey(name) for name in evidence_names}
        if allowed is not None:
            features &= allowed
        graph[disease] = features
    if not graph:
        raise ValueError("DDXPlus condition graph is empty")
    return graph


@contextmanager
def _open_patient_csv(path: str | Path) -> Iterator[IO[str]]:
    patient_path = Path(path)
    if patient_path.suffix.lower() != ".zip":
        with patient_path.open("r", encoding="utf-8", newline="") as handle:
            yield handle
        return

    with zipfile.ZipFile(patient_path) as archive:
        file_names = [name for name in archive.namelist() if not name.endswith(("/", "\\"))]
        csv_names = [name for name in file_names if name.lower().endswith(".csv")]
        candidates = csv_names or file_names
        if len(candidates) != 1:
            raise ValueError(
                f"expected exactly one patient table in {patient_path}, found {len(candidates)}"
            )
        with archive.open(candidates[0]) as binary_handle:
            import io

            with io.TextIOWrapper(binary_handle, encoding="utf-8", newline="") as handle:
                yield handle


def iter_ddxplus_cases(
    patient_csv_or_zip: str | Path,
    evidence_json: str | Path,
    *,
    limit: int | None = None,
    assume_complete: bool = True,
) -> Iterator[ClinicalCase]:
    """Yield official DDXPlus patients without loading the cohort into memory.

    With ``assume_complete=True`` (appropriate for the synthetic official
    release), an unsynthesized binary value and a missing multi-choice option are
    explicit default/negative states.  Disable it for derivatives that do not
    preserve this completeness guarantee.
    """

    metadata = _load_evidence_metadata(evidence_json)
    with _open_patient_csv(patient_csv_or_zip) as handle:
        reader = csv.DictReader(handle)
        required = {"PATHOLOGY", "EVIDENCES"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"patient CSV must contain {sorted(required)}")
        for row_index, row in enumerate(reader):
            if limit is not None and row_index >= limit:
                break
            yield _row_to_case(row, row_index, metadata, assume_complete)


def sample_balanced_ddxplus_cases(
    patient_csv_or_zip: str | Path,
    evidence_json: str | Path,
    *,
    cases_per_disease: int,
    seed: int = 0,
    available_features: Iterable[FeatureKey] | None = None,
) -> list[ClinicalCase]:
    """Reservoir-sample up to N independent test patients per disease."""

    if cases_per_disease <= 0:
        raise ValueError("cases_per_disease must be positive")
    metadata = _load_evidence_metadata(evidence_json)
    rng = random.Random(seed)
    seen: Counter[str] = Counter()
    reservoirs: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    with _open_patient_csv(patient_csv_or_zip) as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            disease = row["PATHOLOGY"]
            seen[disease] += 1
            stored_row = {key: value for key, value in row.items() if value is not None}
            reservoir = reservoirs[disease]
            if len(reservoir) < cases_per_disease:
                reservoir.append((row_index, stored_row))
            else:
                replacement = rng.randrange(seen[disease])
                if replacement < cases_per_disease:
                    reservoir[replacement] = (row_index, stored_row)

    allowed = set(available_features) if available_features is not None else None
    cases = [
        _row_to_case(
            row,
            row_index,
            metadata,
            True,
            allowed_features=allowed,
        )
        for disease in sorted(reservoirs)
        for row_index, row in reservoirs[disease]
    ]
    return cases


def fit_ddxplus_model(
    patient_csv_or_zip: str | Path,
    evidence_json: str | Path,
    *,
    limit: int | None = None,
    alpha: float = 1.0,
    disease_prior_alpha: float = 1.0,
    hierarchical_strength: float = 0.0,
) -> DiseaseStateModel:
    """Fit official complete DDXPlus using sparse sufficient statistics."""

    metadata = _load_evidence_metadata(evidence_json)
    specs = ddxplus_specs(evidence_json)
    disease_counts: Counter[str] = Counter()
    active_counts: dict[str, dict[FeatureKey, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    with _open_patient_csv(patient_csv_or_zip) as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if limit is not None and row_index >= limit:
                break
            disease = row["PATHOLOGY"]
            disease_counts[disease] += 1
            active = _parse_active_evidences(row["EVIDENCES"], row_index)
            for name, active_values in active.items():
                if name not in metadata:
                    raise ValueError(f"unknown evidence {name} at row {row_index + 2}")
                entry = metadata[name]
                data_type = str(entry.get("data_type", "B"))
                if data_type == "B":
                    if None in active_values:
                        active_counts[disease][FeatureKey(name)]["present"] += 1
                elif data_type == "C":
                    selected = [value for value in active_values if value is not None]
                    if len(selected) > 1:
                        raise ValueError(
                            f"categorical evidence {name} has multiple values at row {row_index + 2}"
                        )
                    value = selected[0] if selected else str(entry["default_value"])
                    active_counts[disease][FeatureKey(name)][value] += 1
                elif data_type == "M":
                    default_value = str(entry.get("default_value", ""))
                    for value in active_values:
                        if value is not None and value != default_value:
                            key = FeatureKey.from_parts(name, {"option": value})
                            active_counts[disease][key]["present"] += 1

    spec_map = {spec.key: spec for spec in specs}
    for disease, disease_count in disease_counts.items():
        for key, spec in spec_map.items():
            if set(spec.values) == {"absent", "present"}:
                present = active_counts[disease][key]["present"]
                active_counts[disease][key]["absent"] = disease_count - present
            else:
                metadata_entry = metadata[key.name]
                default_value = str(metadata_entry["default_value"])
                observed = sum(active_counts[disease][key].values())
                active_counts[disease][key][default_value] += disease_count - observed

    return DiseaseStateModel.from_counts(
        specs=specs,
        disease_counts=disease_counts,
        state_counts=active_counts,
        alpha=alpha,
        disease_prior_alpha=disease_prior_alpha,
        hierarchical_strength=hierarchical_strength,
    )


def _parse_active_evidences(raw: str, row_index: int) -> dict[str, set[str | None]]:
    try:
        raw_evidences = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid EVIDENCES at patient row {row_index + 2}") from exc
    if not isinstance(raw_evidences, list):
        raise ValueError(f"EVIDENCES is not a list at patient row {row_index + 2}")
    active: dict[str, set[str | None]] = {}
    for token in raw_evidences:
        name, separator, value = str(token).partition(SEPARATOR)
        active.setdefault(name, set()).add(value if separator else None)
    return active


def _row_to_case(
    row: Mapping[str, str],
    row_index: int,
    metadata: Mapping[str, Mapping[str, object]],
    assume_complete: bool,
    allowed_features: set[FeatureKey] | None = None,
) -> ClinicalCase:
    active = _parse_active_evidences(row["EVIDENCES"], row_index)
    states: dict[FeatureKey, str] = {}
    for name, entry in metadata.items():
        data_type = str(entry.get("data_type", "B"))
        active_values = active.get(name, set())
        if data_type == "B":
            key = FeatureKey(name)
            if allowed_features is None or key in allowed_features:
                if None in active_values:
                    states[key] = "present"
                elif assume_complete:
                    states[key] = "absent"
        elif data_type == "C":
            key = FeatureKey(name)
            if allowed_features is not None and key not in allowed_features:
                continue
            non_null_values = [value for value in active_values if value is not None]
            if non_null_values:
                if len(non_null_values) > 1:
                    raise ValueError(
                        f"categorical evidence {name} has multiple values at row {row_index + 2}"
                    )
                states[key] = non_null_values[0]
            elif assume_complete and "default_value" in entry:
                states[key] = str(entry["default_value"])
        elif data_type == "M":
            if allowed_features is not None and not any(
                key.name == name for key in allowed_features
            ):
                continue
            default_value = str(entry.get("default_value", ""))
            for value in (str(item) for item in entry.get("possible-values", ())):
                if value == default_value:
                    continue
                key = FeatureKey.from_parts(name, {"option": value})
                if allowed_features is not None and key not in allowed_features:
                    continue
                if value in active_values:
                    states[key] = "present"
                elif assume_complete:
                    states[key] = "absent"
    initial = row.get("INITIAL_EVIDENCE", "")
    return ClinicalCase(
        diagnosis=row["PATHOLOGY"],
        states=states,
        case_id=str(row_index),
        initial_observations=(
            _initial_observation(initial, metadata),
        )
        if initial
        else (),
    )


def _initial_observation(
    token: str,
    metadata: Mapping[str, Mapping[str, object]],
) -> Observation:
    name, separator, value = token.partition(SEPARATOR)
    if name not in metadata:
        raise ValueError(f"unknown INITIAL_EVIDENCE: {token}")
    data_type = str(metadata[name].get("data_type", "B"))
    if data_type == "B":
        key = FeatureKey(name)
        observed_value = "present"
    elif data_type == "C":
        if not separator:
            raise ValueError(f"categorical INITIAL_EVIDENCE has no value: {token}")
        key = FeatureKey(name)
        observed_value = value
    elif data_type == "M":
        if not separator:
            raise ValueError(f"multi-choice INITIAL_EVIDENCE has no value: {token}")
        key = FeatureKey.from_parts(name, {"option": value})
        observed_value = "present"
    else:
        raise ValueError(f"unsupported INITIAL_EVIDENCE type: {data_type}")
    return Observation(key=key, value=observed_value, certainty=CertaintyCue.CERTAIN)
