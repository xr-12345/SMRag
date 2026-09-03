"""Prompt #18 -- corrected unified-Brier audit policy, N=5 validation screen.

Runs five strategies over the frozen DDXPlus **validation** manifest (first 5
cases per disease of the frozen N=20 list, 245 cases) at noise {0.2, 0.3} x
seeds {2026, 2027, 2028}, fully paired at the case level:
245 x 2 x 3 x 5 = 7,350 trajectories.

Strategies (labels are the comparison names from spec section 11):
  * heuristic_baseline                 -- existing heuristic policy, byte-identical
                                          regression vs Phase 5 online screen.
  * model_based_v_bayes                -- AskNew by EIG, VerifyOld by deployable
                                          V_Bayes (the expensive reference).
  * unified_brier_audit_forced_verify  -- Prompt #17 behaviour (gross gain forces
                                          VerifyOld), kept as ablation.
  * unified_brier_audit_corrected      -- Prompt #18 correction (gross gain only
                                          blocks Stop; global tau_A margin).
  * unified_brier_audit_corrected_dynamic_rag -- corrected + dynamic RAG (RAG never
                                          enters the Brier value; probe only).

Red lines honoured: validate split only (never test), no oracle in any policy,
no LLM / dense / SFT / RL, no retraining, no threshold sweep, no git commit,
writes only here.
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

from powerful_medrag.action_value import nll_loss
from powerful_medrag.decision import (
    ActionKind,
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    RetrievalGateMode,
    RetrievalMode,
    run_reliability_aware_dialogue,
)
from powerful_medrag.ddxplus import sample_balanced_ddxplus_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.reliability_experiment import _pilot_seed
from powerful_medrag.retrieval import MedicalRetriever
from powerful_medrag.schema import FeatureKey, UNKNOWN
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator
from powerful_medrag.worthiness_policy import (
    CorrectedUnifiedBrierAuditPolicy,
    UnifiedBrierReliabilityAuditPolicy,
    WorthinessStrategy,
    build_policy,
)

OUT = Path(__file__).resolve().parent
MODEL_PATH = "data/ddxplus/model-full.json"
PATIENTS_PATH = "data/ddxplus/release_validate_patients.zip"
EVIDENCES_PATH = "data/ddxplus/release_evidences.json"
CORPUS_PATH = "data/medrag-textbooks"
# Frozen N=20 manifest (first 5 per disease -> 245 cases), spec section 11.
MANIFEST_SOURCE = (
    "artifacts/verimedrag-rag-joint-gate-confirmation-n20/validation_case_manifest.csv"
)

NOISE_RATES = (0.2, 0.3)
SEEDS = (2026, 2027, 2028)

HEURISTIC_BASELINE = "heuristic_baseline"
MODEL_BASED_VBAYES = "model_based_v_bayes"
FORCED_VERIFY = "unified_brier_audit_forced_verify"
CORRECTED = "unified_brier_audit_corrected"
CORRECTED_DYNAMIC_RAG = "unified_brier_audit_corrected_dynamic_rag"

STRATEGIES = (
    HEURISTIC_BASELINE,
    MODEL_BASED_VBAYES,
    FORCED_VERIFY,
    CORRECTED,
    CORRECTED_DYNAMIC_RAG,
)


# --------------------------------------------------------------------------- #
# Recording policies (capture the per-turn decision trace)
# --------------------------------------------------------------------------- #


class RecordingCorrectedPolicy(CorrectedUnifiedBrierAuditPolicy):
    """Corrected policy + a per-turn copy of ``last_decision_log``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_logs: list[dict] = []

    def choose_action(self, *args, **kwargs):
        action = super().choose_action(*args, **kwargs)
        self.decision_logs.append(dict(self.last_decision_log))
        return action


class RecordingForcedPolicy(UnifiedBrierReliabilityAuditPolicy):
    """Prompt #17 forced-verify policy + a minimal per-turn trace."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forced_trace: list[dict] = []

    def choose_action(self, *args, **kwargs):
        action = super().choose_action(*args, **kwargs)
        self.forced_trace.append(
            {
                "g_verify": self._last_audit_g_verify,
                "argmax": self._last_audit_argmax,
                "action_kind": action.kind.value,
                "report_index": action.report_index,
            }
        )
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
    if strategy == CORRECTED_DYNAMIC_RAG:
        # Same corrected policy but with dynamic RAG (never enters Brier value).
        return ReliabilityAwarePolicyConfig(
            max_total_turns=15,
            retrieval_mode=RetrievalMode.DYNAMIC_RAG,
            retrieval_top_k=10,
            query_disease_top_k=5,
            retrieval_impact_weight=1.0,
        )
    # model_based_v_bayes / forced_verify / corrected: no retrieval; the Phase 8A
    # defaults (minimum_action_utility=0.03, tau_V=0.03, tau_A=0.03) apply.
    return ReliabilityAwarePolicyConfig(max_total_turns=15)


def _make_policy(strategy: str, *, config, retriever):
    if strategy == HEURISTIC_BASELINE:
        return build_policy(
            WorthinessStrategy.HEURISTIC_VERIFY, config=config, retriever=retriever
        )
    if strategy == MODEL_BASED_VBAYES:
        return build_policy(
            WorthinessStrategy.MODEL_BASED_VBAYES_VERIFY, config=config
        )
    if strategy == FORCED_VERIFY:
        return RecordingForcedPolicy(config=config, retriever=retriever)
    if strategy in (CORRECTED, CORRECTED_DYNAMIC_RAG):
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
# Per-trajectory run (true state enters ONLY here, never the policy)
# --------------------------------------------------------------------------- #


def run_one(case, noise, seed, strategy, model, retriever):
    config = build_config(strategy)
    patient = StructuredPatientSimulator(
        diagnosis=case.diagnosis,
        latent_states=case.states,
        model=model,
        profile=PatientProfile.from_noise_rate(noise),
        seed=_pilot_seed(seed, case.case_id, noise),
    )
    policy = _make_policy(strategy, config=config, retriever=retriever)
    t0 = time.perf_counter()
    result = run_reliability_aware_dialogue(
        patient, policy=policy, initial_observations=case.initial_observations
    )
    wall = time.perf_counter() - t0

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
        str(t.action.report_index)
        for t in result.turns if t.action.kind is ActionKind.VERIFY
    )

    # Per-turn decision trace (corrected strategies only).
    decision_logs = getattr(policy, "decision_logs", [])
    decision_logs = decision_logs[: len(result.turns)]
    stop_blocked_count = sum(
        1 for l in decision_logs if l.get("stop_blocked_by_verify_audit")
    )
    # Forced-verify trace (forced strategy only).
    forced_trace = getattr(policy, "forced_trace", [])
    forced_verify_count = sum(
        1 for t in forced_trace if t.get("action_kind") == ActionKind.VERIFY.value
    )

    row = {
        "case_id": case.case_id,
        "diagnosis": case.diagnosis,
        "strategy": strategy,
        "noise_rate": noise,
        "seed": seed,
        "predicted_diagnosis": predicted,
        "correct_top1": int(predicted == case.diagnosis),
        "correct_top3": int(case.diagnosis in ranking[:3]),
        "confidence": round(result.belief[predicted], 8),
        "brier_score": round(brier, 8),
        "nll_score": round(nll, 8),
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
        "stop_blocked_count": stop_blocked_count,
        "forced_verify_count": forced_verify_count,
        "action_seq": action_seq,
        "verify_report_seq": verify_report_seq,
        "wall_clock": round(wall, 6),
    }

    turn_rows: list[dict] = []
    stop_audit_rows: list[dict] = []
    for i, l in enumerate(decision_logs):
        turn_rows.append(
            {
                "case_id": case.case_id,
                "diagnosis": case.diagnosis,
                "strategy": strategy,
                "noise_rate": noise,
                "seed": seed,
                "turn_index": i,
                "best_eig_question": l.get("best_eig_question"),
                "best_brier_question": l.get("best_brier_question"),
                "eig_brier_agreement": l.get("eig_brier_agreement"),
                "best_new_gross_gain": l.get("best_new_gross_gain"),
                "best_new_net_value": l.get("best_new_net_value"),
                "best_verify_report_index": l.get("best_verify_report_index"),
                "best_verify_gross_gain": l.get("best_verify_gross_gain"),
                "best_verify_net_value": l.get("best_verify_net_value"),
                "verification_audit_threshold": l.get("verification_audit_threshold"),
                "verification_advantage_margin": l.get("verification_advantage_margin"),
                "base_stop_ready": l.get("base_stop_ready"),
                "audit_blocks_stop": l.get("audit_blocks_stop"),
                "stop_blocked_by_verify_audit": l.get("stop_blocked_by_verify_audit"),
                "chosen_action": l.get("chosen_action"),
                "retrieval_triggered": l.get("retrieval_triggered"),
            }
        )
        if l.get("base_stop_ready") or l.get("audit_blocks_stop"):
            stop_audit_rows.append(
                {
                    "case_id": case.case_id,
                    "diagnosis": case.diagnosis,
                    "strategy": strategy,
                    "noise_rate": noise,
                    "seed": seed,
                    "turn_index": i,
                    "base_stop_ready": l.get("base_stop_ready"),
                    "audit_blocks_stop": l.get("audit_blocks_stop"),
                    "stop_blocked_by_verify_audit": l.get("stop_blocked_by_verify_audit"),
                    "best_verify_gross_gain": l.get("best_verify_gross_gain"),
                    "verification_audit_threshold": l.get("verification_audit_threshold"),
                    "chosen_action": l.get("chosen_action"),
                }
            )
    return row, turn_rows, stop_audit_rows


# --------------------------------------------------------------------------- #
# Worker pool
# --------------------------------------------------------------------------- #

_WORKER = {}


def _init_worker(model, retriever, cases_by_id):
    _WORKER["model"] = model
    _WORKER["retriever"] = retriever
    _WORKER["cases_by_id"] = cases_by_id


def _worker_task(task):
    case_id, noise, seed, strategy = task
    case = _WORKER["cases_by_id"][case_id]
    return run_one(case, noise, seed, strategy, _WORKER["model"], _WORKER["retriever"])


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
    "case_id", "diagnosis", "strategy", "noise_rate", "seed",
    "predicted_diagnosis", "correct_top1", "correct_top3", "confidence",
    "brier_score", "nll_score", "new_questions", "verification_questions",
    "total_atomic_questions", "unnecessary_verifications",
    "resolved_wrong_reports", "correct_to_wrong_flips",
    "wrong_to_correct_flips", "premature_stop", "uncertain_output",
    "stop_reason", "stop_blocked_count", "forced_verify_count",
    "action_seq", "verify_report_seq", "wall_clock",
]

TURN_FIELDS = [
    "case_id", "diagnosis", "strategy", "noise_rate", "seed", "turn_index",
    "best_eig_question", "best_brier_question", "eig_brier_agreement",
    "best_new_gross_gain", "best_new_net_value", "best_verify_report_index",
    "best_verify_gross_gain", "best_verify_net_value",
    "verification_audit_threshold", "verification_advantage_margin",
    "base_stop_ready", "audit_blocks_stop", "stop_blocked_by_verify_audit",
    "chosen_action", "retrieval_triggered",
]

STOP_AUDIT_FIELDS = [
    "case_id", "diagnosis", "strategy", "noise_rate", "seed", "turn_index",
    "base_stop_ready", "audit_blocks_stop", "stop_blocked_by_verify_audit",
    "best_verify_gross_gain", "verification_audit_threshold", "chosen_action",
]


def _run_grid(cases, noises, seeds, strategies, workers):
    cases_by_id = {c.case_id: c for c in cases}
    tasks = [
        (c.case_id, noise, seed, strategy)
        for c in cases
        for noise in noises
        for seed in seeds
        for strategy in strategies
    ]
    print(f"run: {len(tasks)} trajectories over {workers} workers", flush=True)
    ctx = mp.get_context("fork")
    outcome_rows, turn_rows, stop_audit_rows = [], [], []
    walls = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(_WORKER["model"], _WORKER["retriever"], cases_by_id)) as pool:
        for i, (row, turns, stop_audit) in enumerate(
            pool.imap_unordered(_worker_task, tasks, chunksize=4)
        ):
            outcome_rows.append(row)
            turn_rows.extend(turns)
            stop_audit_rows.extend(stop_audit)
            walls.append(row["wall_clock"])
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(tasks)} done "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    write_csv(outcome_rows, OUT / "case_outcomes.csv", OUTCOME_FIELDS)
    write_csv(turn_rows, OUT / "turn_action_values.csv", TURN_FIELDS)
    write_csv(stop_audit_rows, OUT / "stop_audit_events.csv", STOP_AUDIT_FIELDS)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    mean_wall = sum(walls) / len(walls) if walls else 0.0
    print(f"done in {elapsed:.0f}s; {len(outcome_rows)} case rows, "
          f"{len(turn_rows)} turn rows, {len(stop_audit_rows)} stop-audit rows; "
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
        "phase": "unified-brier-audit-corrected-n5",
        "scope": "corrected unified-Brier audit (gross gain blocks Stop, not force "
                 "VerifyOld) + global tau_A margin; N=5 validation screen",
        "strategies": list(STRATEGIES),
        "cases": 245,
        "cases_per_disease": 5,
        "noise_rates": list(NOISE_RATES),
        "seeds": list(SEEDS),
        "max_total_turns": 15,
        "trajectories": 245 * len(NOISE_RATES) * len(SEEDS) * len(STRATEGIES),
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
            MODEL_BASED_VBAYES: "ModelBasedVBayesPolicy (NO_RAG, minimum_action_utility=0.03)",
            FORCED_VERIFY: "UnifiedBrierReliabilityAuditPolicy forced-verify ablation "
                           "(NO_RAG, minimum_action_utility=0.03)",
            CORRECTED: "CorrectedUnifiedBrierAuditPolicy (NO_RAG, minimum_action_utility=0.03)",
            CORRECTED_DYNAMIC_RAG: "CorrectedUnifiedBrierAuditPolicy (DYNAMIC_RAG, "
                                   "minimum_action_utility=0.03)",
        },
        "red_lines": [
            "no test split", "no retraining", "no feature/model change",
            "no threshold sweep", "no true state in prediction", "no oracle",
            "RAG never enters Brier value", "no git commit",
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
        _run_grid(cases, (0.3,), (2026,), list(STRATEGIES), args.workers)
        return 0

    cases = load_selected_cases(model, per_disease=5)
    print(f"full: {len(cases)} cases (5/disease)", flush=True)
    write_manifest(cases, per_disease=5)
    write_config()
    _run_grid(cases, NOISE_RATES, SEEDS, list(STRATEGIES), args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
