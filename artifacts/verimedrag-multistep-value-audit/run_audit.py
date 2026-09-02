"""VeriMedRAG Phase 7 (Prompt #16) — multi-step verification-value audit runner.

Generates and audits *multi-step* rollout labels.  For a frozen belief state
``H_t`` and a candidate first action ``a``, it executes ``a`` and then continues
with the frozen heuristic policy ``pi_heuristic`` until the interview ends, and
records the terminal cost

    J(a | H_t) = E[ Brier(b_T, D*) + c_q * N_future | H_t, a, pi_heuristic ]

with ``Q_multi(a) = -J(a)`` and, for verifying report ``i``,

    V_multi(i) = J(AskNew_best | H_t) - J(VerifyOld(i) | H_t).

Scope: DDXPlus **train** split, 49 cases (1/disease, balanced), sample seed
4041, noise {0.2, 0.3}, patient seed 4041, max turns 15, dynamic RAG, c_q = 0.03.
Up to 2 states per case (early / late).  8 terminal rollouts per state-action
with common random numbers.

This runner only *labels* — it never retrains, never wires anything into the
online policy, and the true disease / latent state enters ONLY the evaluation-side
terminal metrics (never the continuation policy).

Red lines honoured: train split only, no retraining, no online integration, no
oracle correction, no true state in the prediction path, AskNew / Stop unchanged,
no LLM / dense / SFT / RL, no git commit, writes only into this directory.
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
    top1_error,
    verify_value,
)
from powerful_medrag.channel import AnswerChannel
from powerful_medrag.ddxplus import sample_balanced_ddxplus_cases
from powerful_medrag.decision import (
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    RetrievalGateMode,
    RetrievalMode,
)
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.multistep_value import (
    C_QUESTION,
    multistep_rollout,
    reconstruct_tracker,
    run_heuristic_probe,
    select_states,
)
from powerful_medrag.questioning import NumpyQuestionSelector
from powerful_medrag.reliability_experiment import _pilot_seed
from powerful_medrag.retrieval import MedicalRetriever
from powerful_medrag.schema import UNKNOWN
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator

OUT = Path(__file__).resolve().parent
MODEL_PATH = "data/ddxplus/model-full.json"
PATIENTS_PATH = "data/ddxplus/release_train_patients.zip"
EVIDENCES_PATH = "data/ddxplus/release_evidences.json"
CORPUS_PATH = "data/medrag-textbooks"

SEED = 4041
NOISE_RATES = (0.2, 0.3)
CASES_PER_DISEASE = 1
N_ROLLOUTS = 8
MAX_TOTAL_TURNS = 15
MAX_VERIFY_CANDIDATES = 6  # heuristic verify candidates evaluated per state
C_VERIFY = C_QUESTION

ACTION_COLUMNS = [
    "case_id", "diagnosis", "noise_rate", "state_kind", "state_index", "turn_index",
    "n_reports", "n_verify_candidates",
    "action_kind", "evidence_key", "evidence_name", "report_index", "verify_rank",
    "current_risk", "current_brier", "current_true_prob", "top1_before_correct",
    "retrospective_error_prob", "retrieval_impact", "heuristic_verify_utility",
    "v_bayes",
    "j_value", "j_se", "q_multi",
    "terminal_brier", "terminal_brier_se", "terminal_nll", "terminal_top1_error",
    "future_questions", "future_questions_se",
    "premature_stop", "wrong_to_correct", "correct_to_wrong",
    "reported_value", "true_state", "is_report_wrong",
]

ROLLOUT_COLUMNS = [
    "case_id", "diagnosis", "noise_rate", "state_kind", "state_index",
    "action_kind", "report_index", "rollout_index",
    "terminal_brier", "terminal_nll", "future_questions",
    "premature_stop", "wrong_to_correct", "correct_to_wrong",
]


def build_config() -> ReliabilityAwarePolicyConfig:
    """The frozen heuristic policy (same as Phases 5/6)."""
    return ReliabilityAwarePolicyConfig(
        posterior_threshold=0.85,
        posterior_margin_threshold=0.70,
        minimum_action_utility=0.08,
        suspicious_report_threshold=0.05,
        verification_cost=0.03,
        minimum_unreliable_history_cues_for_verification=1,
        max_total_turns=MAX_TOTAL_TURNS,
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
    return sample_balanced_ddxplus_cases(
        PATIENTS_PATH, EVIDENCES_PATH,
        cases_per_disease=CASES_PER_DISEASE, seed=SEED,
        available_features=askable,
    )


def _blank() -> str:
    return ""


def _fmt(value) -> float | str:
    return round(float(value), 10) if value is not None else _blank()


def _report_wrong(report, states) -> bool:
    true_state = states.get(report.key, UNKNOWN)
    return (
        true_state != UNKNOWN
        and report.value != UNKNOWN
        and report.value != true_state
    )


def score_state(
    case, model, tracker, snapshot, *, noise, state_kind, state_index,
    profile, channel, policy, selector,
) -> tuple[list[dict], list[dict]]:
    """Enumerate AskNew_best + VerifyOld candidates and label each multi-step."""
    frozen = snapshot.frozen
    reports = frozen.reports
    initial_observations = frozen.initial_observations
    current_risk = brier_risk(tracker.belief)
    current_brier = realized_brier_loss(tracker.belief, case.diagnosis)
    current_true_prob = tracker.belief.get(case.diagnosis, 0.0)
    top1_before_correct = 1 - top1_error(tracker.belief, case.diagnosis)

    verify_scores = policy._verification_scores(
        tracker,
        initial_observations=initial_observations,
        reports=reports,
        verified_report_indices=frozen.verified,
        report_risks=(),
    )
    score_by_index = {s.report_index: s for s in verify_scores}
    rank_by_index = {s.report_index: r for r, s in enumerate(verify_scores)}
    wrong_indices = [i for i in range(len(reports)) if _report_wrong(reports[i], case.states)]

    # Heuristic candidates (top-k) plus any truly-wrong report so the oracle
    # upper bound is always captured even when a wrong report ranks past the cap.
    candidate_indices: list[int] = [s.report_index for s in verify_scores[:MAX_VERIFY_CANDIDATES]]
    for i in wrong_indices:
        if i not in candidate_indices:
            candidate_indices.append(i)

    action_rows: list[dict] = []
    rollout_rows: list[dict] = []

    def _rollout_row(case_id, diagnosis, action_kind, report_index, rollout):
        for k in range(len(rollout.terminal_brier_list)):
            rollout_rows.append(
                {
                    "case_id": case_id,
                    "diagnosis": diagnosis,
                    "noise_rate": noise,
                    "state_kind": state_kind,
                    "state_index": state_index,
                    "action_kind": action_kind,
                    "report_index": report_index if report_index is not None else _blank(),
                    "rollout_index": k,
                    "terminal_brier": _fmt(rollout.terminal_brier_list[k]),
                    "terminal_nll": _fmt(rollout.terminal_nll_list[k]),
                    "future_questions": rollout.future_questions_list[k],
                    "premature_stop": rollout.premature_stop_list[k],
                    "wrong_to_correct": rollout.wrong_to_correct_list[k],
                    "correct_to_wrong": rollout.correct_to_wrong_list[k],
                }
            )

    def _base(extra):
        row = {
            "case_id": case.case_id,
            "diagnosis": case.diagnosis,
            "noise_rate": noise,
            "state_kind": state_kind,
            "state_index": state_index,
            "turn_index": snapshot.turn_index,
            "n_reports": len(reports),
            "n_verify_candidates": len(verify_scores),
            "current_risk": _fmt(current_risk),
            "current_brier": _fmt(current_brier),
            "current_true_prob": _fmt(current_true_prob),
            "top1_before_correct": top1_before_correct,
        }
        row.update(extra)
        return row

    # AskNew_best.
    best_new = snapshot.new_actions[0]
    new_key = best_new.key
    v_bayes_new = asknew_value(tracker, new_key, C_new=C_QUESTION, selector=selector)
    rollout_new = multistep_rollout(
        case, model, policy=policy, profile=profile, channel=channel,
        frozen=frozen, first_kind="new", first_key=new_key, first_report_index=None,
        base_seed=SEED, noise=noise, state_index=state_index, n_rollouts=N_ROLLOUTS,
        c_q=C_QUESTION, max_total_turns=MAX_TOTAL_TURNS,
    )
    action_rows.append(
        _base(
            {
                "action_kind": "new",
                "evidence_key": new_key.token,
                "evidence_name": new_key.name,
                "report_index": _blank(),
                "verify_rank": _blank(),
                "retrospective_error_prob": _blank(),
                "retrieval_impact": _blank(),
                "heuristic_verify_utility": _blank(),
                "v_bayes": _fmt(v_bayes_new),
                "j_value": _fmt(rollout_new.j_value),
                "j_se": _fmt(rollout_new.j_se),
                "q_multi": _fmt(rollout_new.q_multi),
                "terminal_brier": _fmt(rollout_new.terminal_brier),
                "terminal_brier_se": _fmt(rollout_new.terminal_brier_se),
                "terminal_nll": _fmt(rollout_new.terminal_nll),
                "terminal_top1_error": _fmt(rollout_new.terminal_top1_error),
                "future_questions": _fmt(rollout_new.future_questions),
                "future_questions_se": _fmt(rollout_new.future_questions_se),
                "premature_stop": _fmt(rollout_new.premature_stop),
                "wrong_to_correct": _fmt(rollout_new.wrong_to_correct),
                "correct_to_wrong": _fmt(rollout_new.correct_to_wrong),
                "reported_value": _blank(),
                "true_state": _blank(),
                "is_report_wrong": _blank(),
            }
        )
    )
    _rollout_row(case.case_id, case.diagnosis, "new", None, rollout_new)

    # VerifyOld candidates.
    for report_index in candidate_indices:
        report = reports[report_index]
        score = score_by_index[report_index]
        existing_utility, _final = policy._verification_utility(score)
        v_bayes_verify = verify_value(
            tracker, reports, report_index, initial_observations, C_verify=C_VERIFY,
        )
        rollout_verify = multistep_rollout(
            case, model, policy=policy, profile=profile, channel=channel,
            frozen=frozen, first_kind="verify", first_key=None,
            first_report_index=report_index,
            base_seed=SEED, noise=noise, state_index=state_index,
            n_rollouts=N_ROLLOUTS, c_q=C_QUESTION, max_total_turns=MAX_TOTAL_TURNS,
        )
        true_state = case.states.get(report.key, UNKNOWN)
        comparable = true_state != UNKNOWN and report.value != UNKNOWN
        action_rows.append(
            _base(
                {
                    "action_kind": "verify",
                    "evidence_key": report.key.token,
                    "evidence_name": report.key.name,
                    "report_index": report_index,
                    "verify_rank": rank_by_index.get(report_index, _blank()),
                    "retrospective_error_prob": _fmt(score.error_probability),
                    "retrieval_impact": _fmt(score.retrieval_impact),
                    "heuristic_verify_utility": _fmt(existing_utility),
                    "v_bayes": _fmt(v_bayes_verify),
                    "j_value": _fmt(rollout_verify.j_value),
                    "j_se": _fmt(rollout_verify.j_se),
                    "q_multi": _fmt(rollout_verify.q_multi),
                    "terminal_brier": _fmt(rollout_verify.terminal_brier),
                    "terminal_brier_se": _fmt(rollout_verify.terminal_brier_se),
                    "terminal_nll": _fmt(rollout_verify.terminal_nll),
                    "terminal_top1_error": _fmt(rollout_verify.terminal_top1_error),
                    "future_questions": _fmt(rollout_verify.future_questions),
                    "future_questions_se": _fmt(rollout_verify.future_questions_se),
                    "premature_stop": _fmt(rollout_verify.premature_stop),
                    "wrong_to_correct": _fmt(rollout_verify.wrong_to_correct),
                    "correct_to_wrong": _fmt(rollout_verify.correct_to_wrong),
                    "reported_value": report.value,
                    "true_state": true_state if comparable else _blank(),
                    "is_report_wrong": int(_report_wrong(report, case.states)) if comparable else _blank(),
                }
            )
        )
        _rollout_row(case.case_id, case.diagnosis, "verify", report_index, rollout_verify)

    return action_rows, rollout_rows


def run_one(case, noise, model, retriever, selector):
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
    policy = ReliabilityAwareActionPolicy(config=config, retriever=retriever)

    t0 = time.perf_counter()
    snapshots, stop_reason = run_heuristic_probe(
        patient, policy, initial_observations=case.initial_observations,
        channel=channel, max_total_turns=MAX_TOTAL_TURNS,
    )
    probe_wall = time.perf_counter() - t0

    action_rows: list[dict] = []
    rollout_rows: list[dict] = []
    n_rollouts = 0
    rollout_time = 0.0

    selected = select_states(snapshots)
    for state_index, (state_kind, snapshot) in enumerate(selected):
        tracker = reconstruct_tracker(
            model, channel, snapshot.frozen.initial_observations,
            snapshot.frozen.reports,
        )
        t1 = time.perf_counter()
        act_rows, rol_rows = score_state(
            case, model, tracker, snapshot, noise=noise, state_kind=state_kind,
            state_index=state_index, profile=profile, channel=channel,
            policy=policy, selector=selector,
        )
        rollout_time += time.perf_counter() - t1
        action_rows.extend(act_rows)
        rollout_rows.extend(rol_rows)
        n_rollouts += N_ROLLOUTS * (1 + len(
            {r["report_index"] for r in act_rows if r["action_kind"] == "verify"}
        ))

    wall = (time.perf_counter() - t0) + probe_wall
    return {
        "action_rows": action_rows,
        "rollout_rows": rollout_rows,
        "wall": wall,
        "probe_wall": probe_wall,
        "rollout_time": rollout_time,
        "n_snapshots": len(snapshots),
        "n_states": len(selected),
        "n_rollouts": n_rollouts,
        "stop_reason": stop_reason,
    }


def write_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    write_csv(rows, OUT / "case_manifest.csv",
              fieldnames=["case_id", "diagnosis", "split", "sample_seed",
                          "patient_noise_seed", "cases_per_disease"])


_WORKER = {}


def _init_worker(model, retriever, cases_by_id):
    _WORKER["model"] = model
    _WORKER["retriever"] = retriever
    _WORKER["cases_by_id"] = cases_by_id
    _WORKER["selector"] = NumpyQuestionSelector()


def _worker_task(task):
    case_id, noise = task
    case = _WORKER["cases_by_id"][case_id]
    return run_one(case, noise, _WORKER["model"], _WORKER["retriever"],
                   _WORKER["selector"])


def run(cases, workers, mode, sample_cases):
    cases = cases[:sample_cases] if mode == "smoke" else cases
    cases_by_id = {c.case_id: c for c in cases}
    tasks = [(c.case_id, noise) for c in cases for noise in NOISE_RATES]
    print(f"{mode}: {len(tasks)} case-noise dialogues over {workers} workers")
    ctx = mp.get_context("fork")
    action_rows: list[dict] = []
    rollout_rows: list[dict] = []
    walls = []
    n_rollouts_total = 0
    n_states_total = 0
    rollout_time_total = 0.0
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(_WORKER["model"], _WORKER["retriever"], cases_by_id)) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker_task, tasks, chunksize=2)):
            action_rows.extend(res["action_rows"])
            rollout_rows.extend(res["rollout_rows"])
            walls.append(res["wall"])
            n_rollouts_total += res["n_rollouts"]
            n_states_total += res["n_states"]
            rollout_time_total += res["rollout_time"]
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(tasks)} done ({time.perf_counter() - t0:.0f}s)",
                      flush=True)
    elapsed = time.perf_counter() - t0
    write_csv(action_rows, OUT / "action_labels.csv", ACTION_COLUMNS)
    write_csv(rollout_rows, OUT / "rollout_samples.csv", ROLLOUT_COLUMNS)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    single_rollout = rollout_time_total / n_rollouts_total if n_rollouts_total else 0.0
    runtime_rows = [
        {"metric": "mode", "value": mode},
        {"metric": "cases", "value": len(cases)},
        {"metric": "dialogues", "value": len(tasks)},
        {"metric": "action_rows", "value": len(action_rows)},
        {"metric": "rollout_rows", "value": len(rollout_rows)},
        {"metric": "n_states", "value": n_states_total},
        {"metric": "n_rollouts", "value": n_rollouts_total},
        {"metric": "single_rollout_seconds",
         "value": round(single_rollout, 4)},
        {"metric": "total_wall_seconds", "value": round(elapsed, 1)},
        {"metric": "mean_wall_per_dialogue",
         "value": round(sum(walls) / len(walls), 3) if walls else 0.0},
        {"metric": "peak_rss_mb", "value": round(peak_mb, 1)},
    ]
    write_csv(runtime_rows, OUT / "runtime_profile.csv",
              fieldnames=["metric", "value"])
    print(
        f"done in {elapsed:.0f}s; {len(action_rows)} action rows, "
        f"{n_rollouts_total} rollouts, mean wall/dialogue "
        f"{sum(walls)/len(walls):.2f}s, single rollout {single_rollout:.3f}s, "
        f"peak {peak_mb:.0f}MB"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--sample-cases", type=int, default=10)
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

    _WORKER["model"] = model
    _WORKER["retriever"] = retriever
    run(cases, args.workers, args.mode, args.sample_cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
