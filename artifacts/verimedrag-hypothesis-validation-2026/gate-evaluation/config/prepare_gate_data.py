"""Prompt #7 gate-event merging + case-manifest generation/verification.

TRAIN split:
  - regenerate the deterministic 980-case manifest from release_train_patients.zip
    (sample-seed 101, cases-per-disease 20) — must match the collector's cases.
  - merge seed-101/102/103 event CSVs into train_gate_events.csv (one header,
    seed field preserved). The three seeds share the same 980 cases.

VALIDATION split:
  - read the frozen manifest (validation_case_manifest_with_duplicate_flag.csv).
  - merge seed-2026/2027/2028 event CSVs into validation_gate_events.csv.
  - verify every seed covers exactly the frozen 980 case_ids, and diagnosis
    matches the manifest. Never merge across splits: train and validation
    case_ids are both 0-based zip row indices and numerically overlap.

The two splits are written to separate files and never combined by case_id.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/Users/xr-12345/Desktop/SafeMedRAG")
sys.path.insert(0, str(REPO / "src"))

from powerful_medrag.ddxplus import sample_balanced_ddxplus_cases  # noqa: E402
from powerful_medrag.estimation import DiseaseStateModel  # noqa: E402

ROOT = REPO / "artifacts/verimedrag-hypothesis-validation-2026/gate-evaluation"
MODEL = REPO / "data/ddxplus/model-full.json"
EVIDENCES = REPO / "data/ddxplus/release_evidences.json"
TRAIN_PATIENTS = REPO / "data/ddxplus/release_train_patients.zip"
VALIDATE_PATIENTS = REPO / "data/ddxplus/release_validate_patients.zip"
FROZEN_MANIFEST = (
    REPO
    / "artifacts/verimedrag-hypothesis-validation-2026/formal/config"
    / "validation_case_manifest_with_duplicate_flag.csv"
)


def _read_event_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _merge_seed_events(seed_paths: list[Path], out_path: Path) -> int:
    fieldnames = None
    total = 0
    with out_path.open("w", encoding="utf-8", newline="") as out:
        writer = None
        for path in seed_paths:
            if not path.exists():
                raise FileNotFoundError(f"missing seed event file: {path}")
            rows = _read_event_rows(path)
            if fieldnames is None:
                fieldnames = list(rows[0].keys())
                writer = csv.DictWriter(out, fieldnames=fieldnames)
                writer.writeheader()
            else:
                if list(rows[0].keys()) != fieldnames:
                    raise ValueError(f"column mismatch in {path.name}")
            writer.writerows(rows)
            total += len(rows)
    return total


def _verify_seed_coverage(seed_paths: list[Path], expected_cases: set[str], label: str) -> None:
    for path in seed_paths:
        rows = _read_event_rows(path)
        case_ids = {r["case_id"] for r in rows}
        missing = expected_cases - case_ids
        extra = case_ids - expected_cases
        print(
            f"  {path.parent.name}: {len(rows):>7} rows, "
            f"{len(case_ids):>4} unique case_ids, "
            f"missing={len(missing)}, extra={len(extra)}"
        )
        if missing or extra:
            raise ValueError(f"{label} seed {path.parent.name} case coverage mismatch")


def prepare_train() -> None:
    model = DiseaseStateModel.load(MODEL)
    askable = {key for key, spec in model.specs.items() if spec.askable}
    cases = sample_balanced_ddxplus_cases(
        TRAIN_PATIENTS, EVIDENCES, cases_per_disease=20, seed=101,
        available_features=askable,
    )
    manifest_path = ROOT / "train-events" / "train_case_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["case_id", "diagnosis", "split", "sample_seed", "cases_per_disease"])
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "case_id": case.case_id,
                "diagnosis": case.diagnosis,
                "split": "train",
                "sample_seed": "101",
                "cases_per_disease": "20",
            })
    expected = {case.case_id for case in cases}
    print(f"train manifest: {len(cases)} cases across {len({c.diagnosis for c in cases})} diseases")
    seed_paths = [
        ROOT / "train-events" / f"seed-{seed}" / "ddxplus_gate_events.csv"
        for seed in (101, 102, 103)
    ]
    _verify_seed_coverage(seed_paths, expected, "train")
    total = _merge_seed_events(seed_paths, ROOT / "train-events" / "train_gate_events.csv")
    print(f"train_gate_events.csv: {total} rows (3 seeds merged)")


def prepare_validation() -> None:
    manifest_rows = _read_event_rows(FROZEN_MANIFEST)
    expected = {r["case_id"] for r in manifest_rows}
    diag_by_case = {r["case_id"]: r["diagnosis"] for r in manifest_rows}
    print(f"frozen validation manifest: {len(manifest_rows)} cases")
    seed_paths = [
        ROOT / "validation-events" / f"seed-{seed}" / "ddxplus_gate_events.csv"
        for seed in (2026, 2027, 2028)
    ]
    _verify_seed_coverage(seed_paths, expected, "validation")
    # Verify diagnosis consistency against the frozen manifest.
    for path in seed_paths:
        rows = _read_event_rows(path)
        mismatch = [
            (r["case_id"], r["diagnosis"], diag_by_case[r["case_id"]])
            for r in rows
            if r["diagnosis"] != diag_by_case[r["case_id"]]
        ]
        if mismatch:
            raise ValueError(f"diagnosis mismatch in {path.name}: {mismatch[:5]}...")
    total = _merge_seed_events(seed_paths, ROOT / "validation-events" / "validation_gate_events.csv")
    print(f"validation_gate_events.csv: {total} rows (3 seeds merged)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "train"
    if which == "train":
        prepare_train()
    elif which == "validation":
        prepare_validation()
    else:
        raise SystemExit("usage: prepare_gate_data.py [train|validation]")
