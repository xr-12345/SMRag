"""VeriMedRAG Phase 3A — unified action value rationality audit (runner).

Samples 3 states (early/middle/late) per trajectory from the frozen
dynamic-RAG policy over the DDXPlus **train** split (245 cases, 5/disease,
sample seed 3031) at noise rates {0.2, 0.3}.  At each state it enumerates the
top-5 EIG AskNew candidates and up to 5 VerifyOld candidates and records the 7
comparison scores:

    1. ordinary EIG                         (AskNew only)
    2. retrospective error probability      (VerifyOld only)
    3. retrieval impact = 1 - jaccard       (VerifyOld only)
    4. heuristic verification utility       (VerifyOld only)
    5. error_probability * retrieval_impact (VerifyOld only)
    6. proposed V_Bayes                     (both)
    7. evaluation-only V_real               (both, MC n=8)

Red lines honoured: train split only, no oracle in the deployable V_Bayes, no
LLM/dense/SFT/RL, no git commit, writes only into this directory.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import resource
import time
from pathlib import Path

from powerful_medrag.action_value import (
    asknew_value,
    brier_risk,
    realized_brier_loss,
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
from powerful_medrag.schema import UNKNOWN, FeatureKey
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator

OUT = Path(__file__).resolve().parent
MODEL_PATH = "data/ddxplus/model-full.json"
PATIENTS_PATH = "data/ddxplus/release_train_patients.zip"
EVIDENCES_PATH = "data/ddxplus/release_evidences.json"
CORPUS_PATH = "data/medrag-textbooks"

SEED = 3031
NOISE_RATES = (0.2, 0.3)
CASES_PER_DISEASE = 5
N_STATES = 3
N_ROLLOUTS = 8
TOP_K_ASKNEW = 5
MAX_VERIFY = 5
C_NEW = 0.03
C_VERIFY = 0.03

SAMPLE_COLUMNS = [
    "case_id", "diagnosis", "noise_rate", "state_kind", "state_index", "turn_index",
    "n_reports", "action_kind", "evidence_key", "evidence_name", "report_index",
    "current_risk", "current_brier", "current_true_prob", "eig",
    "retrospective_error_prob", "retrieval_impact", "heuristic_verify_utility",
    "error_prob_times_impact", "v_bayes", "v_real", "v_real_se",
    "gross_brier_reduction", "gross_nll_reduction", "top1_before_correct",
    "wrong_to_correct", "correct_to_wrong",
    "reported_value", "true_state", "is_report_wrong",
]


def build_config() -> ReliabilityAwarePolicyConfig:
    """Frozen state-generation config (the current dynamic-RAG policy)."""
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


def load_cases(model: DiseaseStateModel) -> list:
    askable = {key for key, spec in model.specs.items() if spec.askable}
    cases = sample_balanced_ddxplus_cases(
        PATIENTS_PATH,
        EVIDENCES_PATH,
        cases_per_disease=CASES_PER_DISEASE,
        seed=SEED,
        available_features=askable,
    )
    return cases


class StateRecordingPolicy(ReliabilityAwareActionPolicy):
    """Records the full decision state at every choose_action call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_snapshots: list[dict] = []

    def choose_action(
        self, tracker, *, initial_observations, reports, asked,
        verified_report_indices, verification_count, oracle_states=None,
        report_risks=(),
    ):
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
            tracker,
            initial_observations=initial_observations,
            reports=reports,
            asked=asked,
            verified_report_indices=verified_report_indices,
            verification_count=verification_count,
            oracle_states=oracle_states,
            report_risks=report_risks,
        )


def select_state_indices(snapshots: list[dict]) -> list[tuple[str, int]]:
    """Pick early / middle / late snapshot indices (distinct, increasing)."""
    n = len(snapshots)
    if n == 0:
        return []
    if n == 1:
        return [("late", 0)]
    # early: first snapshot with at least one report (a VerifyOld candidate)
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


def _blank() -> str:
    return ""


def _fmt(value: float | None) -> float | str:
    return round(float(value), 10) if value is not None else _blank()


def score_state(
    case,
    model,
    tracker,
    snapshot,
    *,
    noise,
    state_kind,
    state_index,
    profile,
    channel,
    policy,
    selector,
) -> list[dict]:
    reports = snapshot["reports"]
    initial_observations = snapshot["initial_observations"]
    asked = snapshot["asked"]
    verified = snapshot["verified"]
    current_risk = brier_risk(tracker.belief)
    current_brier = realized_brier_loss(tracker.belief, case.diagnosis)
    current_true_prob = tracker.belief.get(case.diagnosis, 0.0)
    rows: list[dict] = []

    # AskNew: top-5 by ordinary EIG.
    asknew_ranking = selector.rank(tracker, excluded=set(asked))
    for question in asknew_ranking[:TOP_K_ASKNEW]:
        key = question.key
        v_bayes = asknew_value(tracker, key, C_new=C_NEW, selector=selector)
        rollout = realized_value_mc(
            case, model, tracker, reports, initial_observations,
            action_kind="new", profile=profile, channel=channel, key=key,
            n_rollouts=N_ROLLOUTS, base_seed=SEED, noise=noise,
            state_index=state_index, C_new=C_NEW, C_verify=C_VERIFY,
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
                "action_kind": "new",
                "evidence_key": key.token,
                "evidence_name": key.name,
                "report_index": _blank(),
                "current_risk": _fmt(current_risk),
                "current_brier": _fmt(current_brier),
                "current_true_prob": _fmt(current_true_prob),
                "eig": _fmt(question.expected_information_gain),
                "retrospective_error_prob": _blank(),
                "retrieval_impact": _blank(),
                "heuristic_verify_utility": _blank(),
                "error_prob_times_impact": _blank(),
                "v_bayes": _fmt(v_bayes),
                "v_real": _fmt(rollout.value),
                "v_real_se": _fmt(rollout.standard_error),
                "gross_brier_reduction": _fmt(rollout.gross_brier_reduction),
                "gross_nll_reduction": _fmt(rollout.gross_nll_reduction),
                "top1_before_correct": rollout.top1_before_correct,
                "wrong_to_correct": _fmt(rollout.wrong_to_correct),
                "correct_to_wrong": _fmt(rollout.correct_to_wrong),
                "reported_value": _blank(),
                "true_state": _blank(),
                "is_report_wrong": _blank(),
            }
        )

    # VerifyOld: up to MAX_VERIFY candidates (top by retrospective score).
    if reports:
        verify_scores = policy._verification_scores(
            tracker,
            initial_observations=initial_observations,
            reports=reports,
            verified_report_indices=verified,
            report_risks=(),
        )
        for score in verify_scores[:MAX_VERIFY]:
            report_index = score.report_index
            report = reports[report_index]
            existing_utility, _final = policy._verification_utility(score)
            product = score.error_probability * score.retrieval_impact
            v_bayes = verify_value(
                tracker, reports, report_index, initial_observations,
                C_verify=C_VERIFY,
            )
            rollout = realized_value_mc(
                case, model, tracker, reports, initial_observations,
                action_kind="verify", profile=profile, channel=channel,
                report_index=report_index, n_rollouts=N_ROLLOUTS, base_seed=SEED,
                noise=noise, state_index=state_index, C_new=C_NEW, C_verify=C_VERIFY,
            )
            true_state = case.states.get(report.key, UNKNOWN)
            reported_value = report.value
            comparable = true_state != UNKNOWN and reported_value != UNKNOWN
            is_wrong = comparable and reported_value != true_state
            rows.append(
                {
                    "case_id": case.case_id,
                    "diagnosis": case.diagnosis,
                    "noise_rate": noise,
                    "state_kind": state_kind,
                    "state_index": state_index,
                    "turn_index": snapshot["turn_index"],
                    "n_reports": len(reports),
                    "action_kind": "verify",
                    "evidence_key": report.key.token,
                    "evidence_name": report.key.name,
                    "report_index": report_index,
                    "current_risk": _fmt(current_risk),
                    "current_brier": _fmt(current_brier),
                    "current_true_prob": _fmt(current_true_prob),
                    "eig": _blank(),
                    "retrospective_error_prob": _fmt(score.error_probability),
                    "retrieval_impact": _fmt(score.retrieval_impact),
                    "heuristic_verify_utility": _fmt(existing_utility),
                    "error_prob_times_impact": _fmt(product),
                    "v_bayes": _fmt(v_bayes),
                    "v_real": _fmt(rollout.value),
                    "v_real_se": _fmt(rollout.standard_error),
                    "gross_brier_reduction": _fmt(rollout.gross_brier_reduction),
                    "gross_nll_reduction": _fmt(rollout.gross_nll_reduction),
                    "top1_before_correct": rollout.top1_before_correct,
                    "wrong_to_correct": _fmt(rollout.wrong_to_correct),
                    "correct_to_wrong": _fmt(rollout.correct_to_wrong),
                    "reported_value": reported_value,
                    "true_state": true_state,
                    "is_report_wrong": int(is_wrong) if comparable else _blank(),
                }
            )
    return rows


def run_one(case, noise, model, retriever, selector) -> tuple[list[dict], float, int]:
    config = build_config()
    profile = PatientProfile.from_noise_rate(noise)
    channel = AnswerChannel()
    patient = StructuredPatientSimulator(
        diagnosis=case.diagnosis,
        latent_states=case.states,
        model=model,
        profile=profile,
        seed=_pilot_seed(SEED, case.case_id, noise),
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
        tracker = replay_tracker(
            reference, snapshot["initial_observations"], snapshot["reports"]
        )
        rows.extend(
            score_state(
                case, model, tracker, snapshot,
                noise=noise, state_kind=state_kind, state_index=state_index,
                profile=profile, channel=channel, policy=policy, selector=selector,
            )
        )
    return rows, wall, len(snapshots)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(cases):
    rows = [
        {
            "case_id": c.case_id,
            "diagnosis": c.diagnosis,
            "split": "train",
            "sample_seed": SEED,
            "patient_noise_seed": SEED,
            "cases_per_disease": CASES_PER_DISEASE,
        }
        for c in cases
    ]
    write_csv(rows, OUT / "train_action_value_audit_manifest.csv")
    return rows


_WORKER = {}


def _init_worker(model, retriever, cases_by_id):
    _WORKER["model"] = model
    _WORKER["retriever"] = retriever
    _WORKER["cases_by_id"] = cases_by_id
    _WORKER["selector"] = NumpyQuestionSelector()


def _worker_task(task):
    case_id, noise = task
    case = _WORKER["cases_by_id"][case_id]
    return run_one(case, noise, _WORKER["model"], _WORKER["retriever"], _WORKER["selector"])


def run(cases, workers, mode, sample_cases):
    cases = cases[:sample_cases] if mode == "smoke" else cases
    cases_by_id = {c.case_id: c for c in cases}
    tasks = [(c.case_id, noise) for c in cases for noise in NOISE_RATES]
    print(f"{mode}: {len(tasks)} case-noise dialogues over {workers} workers")
    ctx = mp.get_context("fork")
    sample_rows = []
    walls = []
    total_snapshots = 0
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(_WORKER["model"], _WORKER["retriever"], cases_by_id)) as pool:
        for i, (rows, wall, n_snapshots) in enumerate(
            pool.imap_unordered(_worker_task, tasks, chunksize=4)
        ):
            sample_rows.extend(rows)
            walls.append(wall)
            total_snapshots += n_snapshots
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(tasks)} done ({time.perf_counter() - t0:.0f}s)",
                      flush=True)
    elapsed = time.perf_counter() - t0
    write_csv(sample_rows, OUT / "action_value_samples.csv")
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    runtime_rows = [
        {"metric": "cases", "value": len(cases)},
        {"metric": "dialogues", "value": len(tasks)},
        {"metric": "sample_rows", "value": len(sample_rows)},
        {"metric": "mean_wall_seconds_per_dialogue",
         "value": round(sum(walls) / len(walls), 3) if walls else 0.0},
        {"metric": "total_wall_seconds", "value": round(elapsed, 1)},
        {"metric": "mean_snapshots_per_dialogue",
         "value": round(total_snapshots / len(tasks), 2) if tasks else 0.0},
        {"metric": "peak_rss_mb", "value": round(peak_mb, 1)},
    ]
    write_csv(runtime_rows, OUT / "runtime_profile.csv")
    print(
        f"done in {elapsed:.0f}s; {len(sample_rows)} sample rows, "
        f"mean wall/dialogue {sum(walls)/len(walls):.2f}s"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--sample-cases", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = load_model()
    print(f"model loaded ({time.perf_counter() - t0:.2f}s): {len(model.diseases)} diseases")
    t0 = time.perf_counter()
    retriever = MedicalRetriever(CORPUS_PATH)
    print(f"retriever built ({time.perf_counter() - t0:.2f}s): {len(retriever)} snippets")
    t0 = time.perf_counter()
    cases = load_cases(model)
    print(f"loaded {len(cases)} cases ({time.perf_counter() - t0:.2f}s)")
    write_manifest(cases)
    print(f"wrote manifest for {len(cases)} cases")

    _WORKER["model"] = model
    _WORKER["retriever"] = retriever
    run(cases, args.workers, args.mode, args.sample_cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
