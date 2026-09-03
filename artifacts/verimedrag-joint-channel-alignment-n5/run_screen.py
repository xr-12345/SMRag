"""Phase 8C -- correlated re-ask environment/inference N=5 validation screen.

Runs three strategies over the frozen DDXPlus **validation** manifest (first 5
cases per disease of the frozen N=20 list, 245 cases) at noise {0.2, 0.3} x
seeds {2026, 2027, 2028}, with the correlated re-ask patient simulator.

Strategies (spec section 9):
  * heuristic_baseline             -- existing heuristic policy (DYNAMIC_RAG +
                                      JOINT_GATE), byte-identical regression.
  * unified_brier_audit_corrected  -- Phase 8A corrected policy (NO_RAG).
  * joint_channel_brier_audit      -- Phase 8B joint tracker (no UNKNOWN
                                      compression, single shared channel).

Environment / inference scenarios (spec section 6, ``rho_env / rho_model``):
  matched:    0.0/0.0, 0.5/0.5, 0.9/0.9
  mismatched: 0.0/0.5, 0.9/0.5

The two BeliefTracker-based strategies have no rho_model, so they run only at
the three rho_env environments (0.0, 0.5, 0.9); the joint strategy runs all five
scenarios.  Every strategy shares the SAME ``JointStructuredPatientSimulator``
for a given rho_env -- no strategy gets a different patient answer mechanism.

Red lines honoured: validate split only (never test), no rho fitted from data,
no oracle in any policy, no LLM / dense / SFT / RL, no retraining, no N=20,
no threshold sweep, no git commit, writes only here.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from powerful_medrag.action_value import nll_loss
from powerful_medrag.channel import AnswerChannel, ReportMode
from powerful_medrag.decision import (
    ActionKind,
    PolicyTurn,
    ReliabilityAwareDialogueResult,
    ReliabilityAwarePolicyConfig,
    RetrievalGateMode,
    RetrievalMode,
    run_reliability_aware_dialogue,
)
from powerful_medrag.ddxplus import sample_balanced_ddxplus_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.joint_channel_policy import (
    JointChannelBrierAuditPolicy,
)
from powerful_medrag.joint_patient_simulator import JointStructuredPatientSimulator
from powerful_medrag.joint_reliability_belief import JointReliabilityBeliefTracker
from powerful_medrag.joint_report_channel import JointReportChannel, VerificationType
from powerful_medrag.reliability_experiment import _pilot_seed
from powerful_medrag.retrieval import MedicalRetriever
from powerful_medrag.schema import UNKNOWN
from powerful_medrag.simulator import PatientProfile
from powerful_medrag.worthiness_policy import (
    CorrectedUnifiedBrierAuditPolicy,
    WorthinessStrategy,
    build_policy,
)

OUT = Path(__file__).resolve().parent
MODEL_PATH = "data/ddxplus/model-full.json"
PATIENTS_PATH = "data/ddxplus/release_validate_patients.zip"
EVIDENCES_PATH = "data/ddxplus/release_evidences.json"
CORPUS_PATH = "data/medrag-textbooks"
# Frozen N=20 manifest (first 5 per disease -> 245 cases), spec section 9.
MANIFEST_SOURCE = (
    "artifacts/verimedrag-rag-joint-gate-confirmation-n20/validation_case_manifest.csv"
)

NOISE_RATES = (0.2, 0.3)
SEEDS = (2026, 2027, 2028)

HEURISTIC_BASELINE = "heuristic_baseline"
CORRECTED = "unified_brier_audit_corrected"
JOINT = "joint_channel_brier_audit"

# (rho_env, rho_model, label)
RHO_SCENARIOS = (
    (0.0, 0.0, "matched_0_0"),
    (0.5, 0.5, "matched_0.5_0.5"),
    (0.9, 0.9, "matched_0.9_0.9"),
    (0.0, 0.5, "mismatched_0_0.5"),
    (0.9, 0.5, "mismatched_0.9_0.5"),
)


class RecordingCorrectedPolicy(CorrectedUnifiedBrierAuditPolicy):
    """Corrected policy + a per-turn copy of ``last_decision_log``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_logs: list[dict] = []

    def choose_action(self, *args, **kwargs):
        action = super().choose_action(*args, **kwargs)
        self.decision_logs.append(dict(self.last_decision_log))
        return action


# --------------------------------------------------------------------------- #
# Config / policy construction
# --------------------------------------------------------------------------- #


def build_config(strategy: str) -> ReliabilityAwarePolicyConfig:
    if strategy == HEURISTIC_BASELINE:
        # Frozen heuristic (Phase 5 online screen) -- byte-identical regression.
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
    # corrected / joint: no retrieval; Phase 8A defaults (minimum_action_utility
    # = 0.03, tau_V = 0.03, tau_A = 0.03).
    return ReliabilityAwarePolicyConfig(max_total_turns=15)


def _make_policy(strategy, *, config, retriever):
    if strategy == HEURISTIC_BASELINE:
        return build_policy(
            WorthinessStrategy.HEURISTIC_VERIFY, config=config, retriever=retriever
        )
    if strategy == CORRECTED:
        return RecordingCorrectedPolicy(config=config, retriever=retriever)
    raise ValueError(f"unknown strategy {strategy!r}")


# --------------------------------------------------------------------------- #
# Model / cases
# --------------------------------------------------------------------------- #


def load_model() -> DiseaseStateModel:
    return DiseaseStateModel.load(MODEL_PATH)


def load_selected_cases(model: DiseaseStateModel, per_disease: int) -> list:
    askable = {k for k, s in model.specs.items() if s.askable}
    cases = sample_balanced_ddxplus_cases(
        PATIENTS_PATH, EVIDENCES_PATH,
        cases_per_disease=20, seed=2026, available_features=askable,
    )
    by_id = {c.case_id: c for c in cases}
    manifest: dict[str, list[str]] = {}
    with open(MANIFEST_SOURCE, newline="") as f:
        for row in csv.DictReader(f):
            manifest.setdefault(row["diagnosis"], []).append(row["case_id"])
    selected = [cid for d in sorted(manifest) for cid in manifest[d][:per_disease]]
    missing = [cid for cid in selected if cid not in by_id]
    if missing:
        raise ValueError(f"manifest case_ids not reproduced: {missing[:5]}")
    return [by_id[cid] for cid in selected]


# --------------------------------------------------------------------------- #
# Outcome computation (shared by all three strategies)
# --------------------------------------------------------------------------- #


def compute_outcomes(case, model, result, strategy, noise, seed, rho_env, rho_model,
                     mode_agreement_rate=None, answer_agreement_rate=None):
    ranking = sorted(result.belief, key=result.belief.__getitem__, reverse=True)
    predicted = ranking[0]
    brier = sum(
        (result.belief[d] - (1.0 if d == case.diagnosis else 0.0)) ** 2
        for d in model.diseases
    )
    nll = nll_loss(result.belief, case.diagnosis)
    confident_stop = ("posterior" in result.stop_reason) or (
        "confidence" in result.stop_reason
    )
    unnecessary = sum(t.verification_was_unnecessary for t in result.turns)
    resolved = sum(t.verification_resolved_wrong_report for t in result.turns)
    total_turns = result.new_questions + result.verification_questions

    # Realized prediction-flip outcome of each verification (pre/post top-1).
    turn_flip: dict[int, dict] = {}
    prev_belief = None
    for i, t in enumerate(result.turns):
        if t.action.kind is ActionKind.VERIFY and prev_belief is not None:
            pre_correct = max(prev_belief, key=prev_belief.__getitem__) == case.diagnosis
            post_correct = max(t.belief, key=t.belief.__getitem__) == case.diagnosis
            turn_flip[i] = {
                "correct_to_wrong": int(pre_correct and not post_correct),
                "wrong_to_correct": int(not pre_correct and post_correct),
            }
        prev_belief = t.belief
    correct_to_wrong_flips = sum(f["correct_to_wrong"] for f in turn_flip.values())
    wrong_to_correct_flips = sum(f["wrong_to_correct"] for f in turn_flip.values())

    action_seq = "".join(
        "N" if t.action.kind is ActionKind.NEW else
        "V" if t.action.kind is ActionKind.VERIFY else "S"
        for t in result.turns
    )
    verify_report_seq = ",".join(
        t.action.key.name if t.action.key is not None else ""
        for t in result.turns if t.action.kind is ActionKind.VERIFY
    )

    return {
        "case_id": case.case_id,
        "diagnosis": case.diagnosis,
        "strategy": strategy,
        "noise_rate": noise,
        "seed": seed,
        "rho_env": rho_env,
        "rho_model": "" if rho_model is None else rho_model,
        "predicted_diagnosis": predicted,
        "correct_top1": int(predicted == case.diagnosis),
        "correct_top3": int(case.diagnosis in ranking[:3]),
        "confidence": round(result.belief[predicted], 8),
        "brier_score": round(brier, 8),
        "nll_score": round(nll, 8),
        "posterior_entropy": round(
            -sum(p * np.log(p) for p in result.belief.values() if p > 0), 8
        ),
        "new_questions": result.new_questions,
        "verification_questions": result.verification_questions,
        "total_atomic_questions": total_turns,
        "unnecessary_verifications": unnecessary,
        "resolved_wrong_reports": resolved,
        "correct_to_wrong_flips": correct_to_wrong_flips,
        "wrong_to_correct_flips": wrong_to_correct_flips,
        "premature_stop": int(predicted != case.diagnosis and confident_stop),
        "uncertain_output": int(result.uncertain_output),
        "stop_reason": result.stop_reason,
        "mode_agreement_rate": (
            round(mode_agreement_rate, 8) if mode_agreement_rate is not None else ""
        ),
        "answer_agreement_rate": (
            round(answer_agreement_rate, 8) if answer_agreement_rate is not None else ""
        ),
        "action_seq": action_seq,
        "verify_report_seq": verify_report_seq,
    }


# --------------------------------------------------------------------------- #
# Per-trajectory runs (true state enters ONLY here, never the policy)
# --------------------------------------------------------------------------- #


def run_belief_tracker_one(case, noise, seed, strategy, rho_env, model, retriever):
    config = build_config(strategy)
    patient = JointStructuredPatientSimulator(
        diagnosis=case.diagnosis,
        latent_states=case.states,
        model=model,
        profile=PatientProfile.from_noise_rate(noise),
        seed=_pilot_seed(seed, case.case_id, noise),
        rho_env=rho_env,
        case_id=case.case_id,
    )
    policy = _make_policy(strategy, config=config, retriever=retriever)
    t0 = time.perf_counter()
    result = run_reliability_aware_dialogue(
        patient, policy=policy, initial_observations=case.initial_observations
    )
    wall = time.perf_counter() - t0

    row = compute_outcomes(
        case, model, result, strategy, noise, seed, rho_env, None
    )
    row["wall_clock"] = round(wall, 6)

    decision_logs = getattr(policy, "decision_logs", [])
    turn_rows = _turn_rows(case, noise, seed, strategy, rho_env, None,
                           decision_logs[: len(result.turns)])
    return row, turn_rows, []


def run_joint_one(case, noise, seed, rho_env, rho_model, model):
    patient = JointStructuredPatientSimulator(
        diagnosis=case.diagnosis,
        latent_states=case.states,
        model=model,
        profile=PatientProfile.from_noise_rate(noise),
        seed=_pilot_seed(seed, case.case_id, noise),
        rho_env=rho_env,
        case_id=case.case_id,
    )
    channel = JointReportChannel(AnswerChannel(), repeat_mode_persistence=rho_model)
    tracker = JointReliabilityBeliefTracker(model, channel)
    policy = JointChannelBrierAuditPolicy(model, channel=channel, tracker=tracker)
    for obs in case.initial_observations:
        tracker.observe_single(obs)

    turns: list[PolicyTurn] = []
    decision_logs: list[dict] = []
    reliability_rows: list[dict] = []
    verification_count = 0
    stop_reason = "total_turn_budget"
    uncertain_output = False

    t0 = time.perf_counter()
    while len(turns) < policy.config.max_total_turns:
        action = policy.choose_action()
        decision_logs.append(dict(policy.last_decision_log))
        if action.kind is ActionKind.STOP:
            stop_reason = action.explanation
            uncertain_output = "uncertainty" in action.explanation
            break
        if action.kind is ActionKind.NEW:
            key = action.key
            observation, true_mode = patient.answer(key)
            tracker.observe_single(observation)
            true_state = patient.latent_states.get(key, UNKNOWN)
            true_wrong = (
                true_state != UNKNOWN
                and observation.value != UNKNOWN
                and observation.value != true_state
            )
            reliability_rows.append(
                {
                    "case_id": case.case_id,
                    "diagnosis": case.diagnosis,
                    "strategy": JOINT,
                    "noise_rate": noise,
                    "seed": seed,
                    "rho_env": rho_env,
                    "rho_model": rho_model,
                    "key": key.name,
                    "value": observation.value,
                    "certainty": observation.certainty.value,
                    "true_mode": true_mode.value,
                    "true_state": true_state,
                    "true_wrong": int(true_wrong),
                    "true_misreported": int(true_mode is ReportMode.MISREPORTED),
                    "p_mode": round(tracker.p_mode_misreported(key), 8),
                    "p_wrong": round(tracker.p_wrong(key), 8),
                }
            )
            turns.append(
                PolicyTurn(
                    index=len(turns) + 1,
                    action=action,
                    observation=observation,
                    true_report_mode=true_mode,
                    belief=dict(tracker.belief),
                )
            )
            continue

        key = action.key
        if key is None or key not in tracker.memory:
            raise RuntimeError("verification action has an invalid feature key")
        original = tracker.memory[key].original
        clarification, true_mode = patient.answer(key)
        true_state = patient.latent_states.get(key, UNKNOWN)
        original_wrong = (
            true_state != UNKNOWN
            and original.value != UNKNOWN
            and original.value != true_state
        )
        resolved_wrong = (
            true_state != UNKNOWN
            and clarification.value != UNKNOWN
            and clarification.value != true_state
        )
        tracker.observe_verification(key, clarification, VerificationType.REPEAT)
        verification_count += 1
        turns.append(
            PolicyTurn(
                index=len(turns) + 1,
                action=action,
                observation=clarification,
                true_report_mode=true_mode,
                belief=dict(tracker.belief),
                verification_changed_report=(
                    clarification.value != original.value
                    or clarification.certainty != original.certainty
                ),
                verification_was_unnecessary=not original_wrong,
                verification_resolved_wrong_report=original_wrong and not resolved_wrong,
            )
        )
    wall = time.perf_counter() - t0

    predicted = max(tracker.belief, key=tracker.belief.__getitem__)
    result = ReliabilityAwareDialogueResult(
        true_diagnosis=patient.diagnosis,
        predicted_diagnosis=predicted,
        belief=dict(tracker.belief),
        turns=tuple(turns),
        stop_reason=stop_reason,
        new_questions=sum(t.action.kind is ActionKind.NEW for t in turns),
        verification_questions=sum(t.action.kind is ActionKind.VERIFY for t in turns),
        uncertain_output=uncertain_output,
    )

    # First/re-ask mode & answer agreement over verified features (eval-only).
    verified_keys = [
        t.action.key for t in turns if t.action.kind is ActionKind.VERIFY
    ]
    mode_agree = answer_agree = 0
    n_verified = 0
    for key in verified_keys:
        chain = patient.mode_chain_for(key)
        answers = patient.answer_chain_for(key)
        if len(chain) >= 2:
            n_verified += 1
            mode_agree += int(chain[0] == chain[1])
        if len(answers) >= 2:
            answer_agree += int(answers[0].value == answers[1].value)
    mode_agreement_rate = mode_agree / n_verified if n_verified else None
    answer_agreement_rate = answer_agree / n_verified if n_verified else None

    row = compute_outcomes(
        case, model, result, JOINT, noise, seed, rho_env, rho_model,
        mode_agreement_rate=mode_agreement_rate,
        answer_agreement_rate=answer_agreement_rate,
    )
    row["wall_clock"] = round(wall, 6)

    turn_rows = _turn_rows(case, noise, seed, JOINT, rho_env, rho_model,
                           decision_logs[: len(turns)])
    return row, turn_rows, reliability_rows


def _turn_rows(case, noise, seed, strategy, rho_env, rho_model, decision_logs):
    rows = []
    for i, l in enumerate(decision_logs):
        rows.append(
            {
                "case_id": case.case_id,
                "diagnosis": case.diagnosis,
                "strategy": strategy,
                "noise_rate": noise,
                "seed": seed,
                "rho_env": rho_env,
                "rho_model": "" if rho_model is None else rho_model,
                "turn_index": i,
                "best_new_net_value": l.get("best_new_net_value"),
                "best_verify_net_value": l.get("best_verify_net_value"),
                "best_verify_gross_gain": l.get("best_verify_gross_gain"),
                "best_new_gross_gain": l.get("best_new_gross_gain"),
                "max_mode_misreport": l.get("max_mode_misreport"),
                "suspicious_report_threshold": l.get("suspicious_report_threshold"),
                "reliability_ready": l.get("reliability_ready"),
                "base_stop_ready": l.get("base_stop_ready"),
                "audit_blocks_stop": l.get("audit_blocks_stop"),
                "stop_blocked_by_verify_audit": l.get("stop_blocked_by_verify_audit"),
                "chosen_action": l.get("chosen_action"),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Worker pool
# --------------------------------------------------------------------------- #

_WORKER = {}


def _init_worker(model, retriever, cases_by_id):
    _WORKER["model"] = model
    _WORKER["retriever"] = retriever
    _WORKER["cases_by_id"] = cases_by_id


def _worker_task(task):
    case_id, noise, seed, strategy, rho_env, rho_model = task
    case = _WORKER["cases_by_id"][case_id]
    if strategy == JOINT:
        return run_joint_one(case, noise, seed, rho_env, rho_model, _WORKER["model"])
    return run_belief_tracker_one(
        case, noise, seed, strategy, rho_env, _WORKER["model"], _WORKER["retriever"]
    )


def write_csv(rows, path, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fields = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


OUTCOME_FIELDS = [
    "case_id", "diagnosis", "strategy", "noise_rate", "seed", "rho_env",
    "rho_model", "predicted_diagnosis", "correct_top1", "correct_top3",
    "confidence", "brier_score", "nll_score", "posterior_entropy",
    "new_questions", "verification_questions", "total_atomic_questions",
    "unnecessary_verifications", "resolved_wrong_reports",
    "correct_to_wrong_flips", "wrong_to_correct_flips", "premature_stop",
    "uncertain_output", "stop_reason", "mode_agreement_rate",
    "answer_agreement_rate", "action_seq", "verify_report_seq", "wall_clock",
]

TURN_FIELDS = [
    "case_id", "diagnosis", "strategy", "noise_rate", "seed", "rho_env",
    "rho_model", "turn_index", "best_new_net_value", "best_verify_net_value",
    "best_verify_gross_gain", "best_new_gross_gain", "max_mode_misreport",
    "suspicious_report_threshold", "reliability_ready", "base_stop_ready",
    "audit_blocks_stop", "stop_blocked_by_verify_audit", "chosen_action",
]

RELIABILITY_FIELDS = [
    "case_id", "diagnosis", "strategy", "noise_rate", "seed", "rho_env",
    "rho_model", "key", "value", "certainty", "true_mode", "true_state",
    "true_wrong", "true_misreported", "p_mode", "p_wrong",
]


def _build_tasks(cases, noises, seeds):
    tasks = []
    for c in cases:
        for noise in noises:
            for seed in seeds:
                for rho_env in (0.0, 0.5, 0.9):
                    tasks.append((c.case_id, noise, seed, HEURISTIC_BASELINE, rho_env, None))
                    tasks.append((c.case_id, noise, seed, CORRECTED, rho_env, None))
                for rho_env, rho_model, _label in RHO_SCENARIOS:
                    tasks.append((c.case_id, noise, seed, JOINT, rho_env, rho_model))
    return tasks


def _run_grid(cases, noises, seeds, workers):
    cases_by_id = {c.case_id: c for c in cases}
    tasks = _build_tasks(cases, noises, seeds)
    print(f"run: {len(tasks)} trajectories over {workers} workers", flush=True)
    ctx = mp.get_context("fork")
    outcome_rows, turn_rows, reliability_rows = [], [], []
    walls = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(_WORKER["model"], _WORKER["retriever"], cases_by_id)) as pool:
        for i, (row, turns, reliability) in enumerate(
            pool.imap_unordered(_worker_task, tasks, chunksize=4)
        ):
            outcome_rows.append(row)
            turn_rows.extend(turns)
            reliability_rows.extend(reliability)
            walls.append(row["wall_clock"])
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(tasks)} done "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    write_csv(outcome_rows, OUT / "case_outcomes.csv", OUTCOME_FIELDS)
    write_csv(turn_rows, OUT / "turn_action_values.csv", TURN_FIELDS)
    write_csv(reliability_rows, OUT / "reliability_predictions.csv",
              RELIABILITY_FIELDS)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    mean_wall = sum(walls) / len(walls) if walls else 0.0
    print(f"done in {elapsed:.0f}s; {len(outcome_rows)} case rows, "
          f"{len(turn_rows)} turn rows, {len(reliability_rows)} reliability rows; "
          f"mean wall {mean_wall:.2f}s; peak {peak_mb:.0f}MB", flush=True)
    write_csv(
        [
            {"metric": "trajectories", "value": len(tasks)},
            {"metric": "workers", "value": workers},
            {"metric": "elapsed_seconds", "value": round(elapsed, 1)},
            {"metric": "mean_wall_seconds_per_dialogue",
             "value": round(mean_wall, 3)},
            {"metric": "peak_rss_mb", "value": round(peak_mb, 1)},
        ],
        OUT / "runtime_profile.csv",
    )


def write_manifest(cases, per_disease):
    rows = [
        {"case_id": c.case_id, "diagnosis": c.diagnosis, "split": "validate",
         "sample_seed": 2026, "cases_per_disease": per_disease}
        for c in cases
    ]
    write_csv(rows, OUT / "case_manifest.csv",
              fieldnames=["case_id", "diagnosis", "split", "sample_seed",
                          "cases_per_disease"])


def write_config():
    config = {
        "phase": "joint-channel-alignment-n5",
        "scope": "correlated re-ask simulator (rho_env) vs joint inference "
                 "(rho_model) alignment; N=5 validation screen",
        "strategies": [HEURISTIC_BASELINE, CORRECTED, JOINT],
        "rho_scenarios": [
            {"rho_env": r[0], "rho_model": r[1], "label": r[2]}
            for r in RHO_SCENARIOS
        ],
        "cases": 245,
        "cases_per_disease": 5,
        "noise_rates": list(NOISE_RATES),
        "seeds": list(SEEDS),
        "max_total_turns": 15,
        "manifest": "frozen N=20 manifest (first 5 per disease)",
        "manifest_source": MANIFEST_SOURCE,
        "frozen_params": {
            "verification_audit_threshold": 0.03,
            "verification_advantage_margin": 0.03,
            "verification_cost": 0.03,
            "new_question_cost_weight": 0.01,
            "maximum_verifications": 1,
            "max_total_turns": 15,
        },
        "policy_by_strategy": {
            HEURISTIC_BASELINE: "ReliabilityAwareActionPolicy (DYNAMIC_RAG, "
                                "JOINT_GATE, minimum_action_utility=0.08)",
            CORRECTED: "CorrectedUnifiedBrierAuditPolicy (NO_RAG, "
                       "minimum_action_utility=0.03)",
            JOINT: "JointChannelBrierAuditPolicy (standalone, no RAG, "
                   "minimum_action_utility=0.03; rho_model per scenario)",
        },
        "red_lines": [
            "no test split", "no rho fitted from data", "no retraining",
            "no feature/model change", "no threshold sweep",
            "no true state in prediction", "no oracle", "no N=20",
            "RAG never enters the joint Brier value", "no git commit",
        ],
    }
    (OUT / "CONFIG.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = load_model()
    print(f"model loaded ({time.perf_counter()-t0:.2f}s): {len(model.diseases)} diseases",
          flush=True)
    t0 = time.perf_counter()
    retriever = MedicalRetriever(CORPUS_PATH)
    print(f"retriever built ({time.perf_counter()-t0:.2f}s): {len(retriever)} snippets",
          flush=True)

    _WORKER["model"] = model
    _WORKER["retriever"] = retriever

    if args.mode == "smoke":
        cases = load_selected_cases(model, per_disease=1)
        print(f"smoke: {len(cases)} cases (1/disease)", flush=True)
        write_manifest(cases, per_disease=1)
        write_config()
        _run_grid(cases, (0.3,), (2026,), args.workers)
        return 0

    cases = load_selected_cases(model, per_disease=5)
    print(f"full: {len(cases)} cases (5/disease)", flush=True)
    write_manifest(cases, per_disease=5)
    write_config()
    _run_grid(cases, NOISE_RATES, SEEDS, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
