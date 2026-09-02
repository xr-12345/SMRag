"""Phase 4 -- generate the validation feature/label set.

Runs the frozen dynamic-RAG policy over the DDXPlus **validate** split (735
primary cases = N=20 manifest minus N=5 screen), at noise {0.2, 0.3} x seeds
{2027, 2028}, recording VerifyOld candidates (<=5 per state) with the same
deployable label-free features + realized labels as ``train_features.csv``.

Adapted from ``run_audit.py`` (Phase 3A) but: validate split, 2 seeds, and
VerifyOld-only recording.  Writes only into this directory.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import resource
import time
from pathlib import Path

from powerful_medrag.action_value import (
    brier_risk,
    realized_value_mc,
    replay_tracker,
    verify_value,
)
from powerful_medrag.belief import BeliefTracker
from powerful_medrag.channel import AnswerChannel
from powerful_medrag.ddxplus import sample_balanced_ddxplus_cases
from powerful_medrag.decision import (
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    RetrievalGateMode,
    RetrievalMode,
    run_reliability_aware_dialogue,
)
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.questioning import NumpyQuestionSelector
from powerful_medrag.reliability_experiment import _pilot_seed
from powerful_medrag.retrieval import MedicalRetriever
from powerful_medrag.schema import UNKNOWN
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator

OUT = Path(__file__).resolve().parent
MODEL_PATH = "data/ddxplus/model-full.json"
PATIENTS_PATH = "data/ddxplus/release_validate_patients.zip"
EVIDENCES_PATH = "data/ddxplus/release_evidences.json"
CORPUS_PATH = "data/medrag-textbooks"
SCREEN_MANIFEST_PATH = (
    "artifacts/verimedrag-rag-joint-gate-screen-n5/screen_case_manifest.csv"
)

CASE_SAMPLE_SEED = 2026
CASES_PER_DISEASE = 20
SEEDS = (2027, 2028)
NOISE_RATES = (0.2, 0.3)
N_STATES = 3
N_ROLLOUTS = 8
TOP_K_ASKNEW = 5
MAX_VERIFY = 5
C_NEW = 0.03
C_VERIFY = 0.03

# Identical schema to train_features.csv.
FEATURE_CSV_COLUMNS = [
    "case_id", "diagnosis", "noise_rate", "state_kind", "state_index",
    "turn_index", "n_reports",
    "retrospective_error_prob", "retrieval_impact", "heuristic_verify_utility",
    "error_prob_times_impact", "current_risk",
    "v_bayes", "v_real", "gross_brier_reduction",
    "top1_before_correct", "correct_to_wrong",
]


def build_config() -> ReliabilityAwarePolicyConfig:
    return ReliabilityAwarePolicyConfig(
        posterior_threshold=0.85,
        posterior_margin_threshold=0.70,
        minimum_action_utility=0.08,
        suspicious_report_threshold=0.05,
        verification_cost=0.03,
        minimum_unreliable_history_cues_for_verification=1,
        max_total_turns=15,
        retrieval_mode=RetrievalMode.DYNAMIC_RAG,
        retrieval_top_k=10,
        query_disease_top_k=5,
        retrieval_gate_mode=RetrievalGateMode.JOINT_GATE,
        retrieval_impact_weight=1.0,
    )


def load_model() -> DiseaseStateModel:
    return DiseaseStateModel.load(MODEL_PATH)


def load_primary_cases(model: DiseaseStateModel) -> list:
    askable = {k for k, s in model.specs.items() if s.askable}
    cases = sample_balanced_ddxplus_cases(
        PATIENTS_PATH, EVIDENCES_PATH,
        cases_per_disease=CASES_PER_DISEASE, seed=CASE_SAMPLE_SEED,
        available_features=askable,
    )
    screen_ids = {
        row["case_id"]
        for row in csv.DictReader(open(SCREEN_MANIFEST_PATH, newline="", encoding="utf-8"))
    }
    primary = [c for c in cases if c.case_id not in screen_ids]
    assert len(primary) == 735, f"expected 735 primary cases, got {len(primary)}"
    return primary


class StateRecordingPolicy(ReliabilityAwareActionPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_snapshots: list[dict] = []

    def choose_action(self, tracker, *, initial_observations, reports, asked,
                      verified_report_indices, verification_count,
                      oracle_states=None, report_risks=()):
        self.state_snapshots.append(
            {
                "belief": dict(tracker.belief),
                "reports": tuple(reports),
                "initial_observations": tuple(initial_observations),
                "asked": frozenset(asked),
                "verified": frozenset(verified_report_indices),
            }
        )
        return super().choose_action(
            tracker, initial_observations=initial_observations,
            reports=reports, asked=asked,
            verified_report_indices=verified_report_indices,
            verification_count=verification_count, oracle_states=oracle_states,
            report_risks=report_risks,
        )


def select_state_indices(snapshots: list[dict]) -> list[tuple[str, int]]:
    n = len(snapshots)
    if n == 0:
        return []
    if n == 1:
        return [("late", 0)]
    early = None
    for index, snapshot in enumerate(snapshots):
        if snapshot["reports"]:
            early = index
            break
    if early is None:
        early = 0
    late = n - 1
    middle = (early + late) // 2
    seen: set[int] = set()
    ordered: list[tuple[str, int]] = []
    for kind, index in (("early", early), ("middle", middle), ("late", late)):
        if index not in seen:
            seen.add(index)
            ordered.append((kind, index))
    ordered.sort(key=lambda item: item[1])
    return ordered


def _fmt(value: float | None) -> float | str:
    return round(float(value), 10) if value is not None else ""


def score_state(case, model, tracker, snapshot, *, noise, state_kind,
                state_index, seed, profile, channel, policy) -> list[dict]:
    reports = snapshot["reports"]
    initial_observations = snapshot["initial_observations"]
    verified = snapshot["verified"]
    current_risk = brier_risk(tracker.belief)
    rows: list[dict] = []
    if not reports:
        return rows
    verify_scores = policy._verification_scores(
        tracker, initial_observations=initial_observations, reports=reports,
        verified_report_indices=verified, report_risks=(),
    )
    for score in verify_scores[:MAX_VERIFY]:
        report_index = score.report_index
        existing_utility, _final = policy._verification_utility(score)
        product = score.error_probability * score.retrieval_impact
        v_bayes = verify_value(tracker, reports, report_index,
                               initial_observations, C_verify=C_VERIFY)
        rollout = realized_value_mc(
            case, model, tracker, reports, initial_observations,
            action_kind="verify", profile=profile, channel=channel,
            report_index=report_index, n_rollouts=N_ROLLOUTS, base_seed=seed,
            noise=noise, state_index=state_index, C_new=C_NEW, C_verify=C_VERIFY,
        )
        rows.append(
            {
                "case_id": case.case_id,
                "diagnosis": case.diagnosis,
                "noise_rate": noise,
                "state_kind": state_kind,
                "state_index": state_index,
                "turn_index": snapshot["turn_index"],
                "n_reports": len(reports),
                "retrospective_error_prob": _fmt(score.error_probability),
                "retrieval_impact": _fmt(score.retrieval_impact),
                "heuristic_verify_utility": _fmt(existing_utility),
                "error_prob_times_impact": _fmt(product),
                "current_risk": _fmt(current_risk),
                "v_bayes": _fmt(v_bayes),
                "v_real": _fmt(rollout.value),
                "gross_brier_reduction": _fmt(rollout.gross_brier_reduction),
                "top1_before_correct": rollout.top1_before_correct,
                "correct_to_wrong": _fmt(rollout.correct_to_wrong),
            }
        )
    return rows


def run_one(case, noise, seed, model, retriever, selector) -> tuple[list[dict], float, int]:
    config = build_config()
    profile = PatientProfile.from_noise_rate(noise)
    channel = AnswerChannel()
    patient = StructuredPatientSimulator(
        diagnosis=case.diagnosis, latent_states=case.states, model=model,
        profile=profile, seed=_pilot_seed(seed, case.case_id, noise),
    )
    policy = StateRecordingPolicy(config=config, retriever=retriever)
    t0 = time.perf_counter()
    run_reliability_aware_dialogue(
        patient, policy=policy, initial_observations=case.initial_observations
    )
    wall = time.perf_counter() - t0

    snapshots = policy.state_snapshots
    for turn_index, snapshot in enumerate(snapshots):
        snapshot["turn_index"] = turn_index
    reference = BeliefTracker(model, channel)
    rows: list[dict] = []
    indices = select_state_indices(snapshots)
    for state_index, (state_kind, turn_index) in enumerate(indices):
        snapshot = snapshots[turn_index]
        tracker = replay_tracker(reference, snapshot["initial_observations"],
                                 snapshot["reports"])
        rows.extend(
            score_state(case, model, tracker, snapshot, noise=noise,
                        state_kind=state_kind, state_index=state_index,
                        seed=seed, profile=profile, channel=channel, policy=policy)
        )
    return rows, wall, len(snapshots)


def write_csv(rows, path, fieldnames=FEATURE_CSV_COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


_WORKER = {}


def _init_worker(model, retriever, cases_by_id):
    _WORKER["model"] = model
    _WORKER["retriever"] = retriever
    _WORKER["cases_by_id"] = cases_by_id
    _WORKER["selector"] = NumpyQuestionSelector()


def _worker_task(task):
    case_id, noise, seed = task
    case = _WORKER["cases_by_id"][case_id]
    return run_one(case, noise, seed, _WORKER["model"], _WORKER["retriever"],
                   _WORKER["selector"])


def run(cases, workers, mode, sample_cases, model, retriever):
    cases = cases[:sample_cases] if mode == "smoke" else cases
    cases_by_id = {c.case_id: c for c in cases}
    tasks = [(c.case_id, noise, seed) for c in cases for noise in NOISE_RATES
             for seed in SEEDS]
    print(f"{mode}: {len(tasks)} case-noise-seed dialogues over {workers} workers")
    ctx = mp.get_context("fork")
    rows: list[dict] = []
    walls = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(model, retriever, cases_by_id)) as pool:
        for i, (r, wall, _) in enumerate(
            pool.imap_unordered(_worker_task, tasks, chunksize=4)
        ):
            rows.extend(r)
            walls.append(wall)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} done ({time.perf_counter()-t0:.0f}s)",
                      flush=True)
    elapsed = time.perf_counter() - t0
    write_csv(rows, OUT / "validation_features.csv")
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print(f"done in {elapsed:.0f}s; {len(rows)} verify rows; "
          f"mean wall/dialogue {sum(walls)/len(walls):.2f}s; peak {peak_mb:.0f}MB")


def write_manifest(cases):
    rows = [
        {"case_id": c.case_id, "diagnosis": c.diagnosis, "split": "validate",
         "sample_seed": CASE_SAMPLE_SEED, "cases_per_disease": CASES_PER_DISEASE,
         "primary_confirmation": 1}
        for c in cases
    ]
    write_csv(rows, OUT / "validation_worthiness_manifest.csv",
              fieldnames=["case_id", "diagnosis", "split", "sample_seed",
                          "cases_per_disease", "primary_confirmation"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--sample-cases", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = load_model()
    print(f"model loaded ({time.perf_counter()-t0:.2f}s)")
    t0 = time.perf_counter()
    retriever = MedicalRetriever(CORPUS_PATH)
    print(f"retriever built ({time.perf_counter()-t0:.2f}s): {len(retriever)} snippets")
    t0 = time.perf_counter()
    cases = load_primary_cases(model)
    print(f"loaded {len(cases)} primary cases ({time.perf_counter()-t0:.2f}s)")
    write_manifest(cases)
    print(f"wrote validation_worthiness_manifest.csv ({len(cases)} cases)")

    run(cases, args.workers, args.mode, args.sample_cases, model, retriever)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
