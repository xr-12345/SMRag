"""Phase 6 -- VerifyOld drop-in isolation screen (Prompt #15).

Runs four strictly-isolated strategies over the frozen DDXPlus validation
manifest at noise {0.2, 0.3} x seeds {2026, 2027, 2028}, fully paired at the
case level.

Strategies (see ``powerful_medrag.worthiness_dropin``):
  * heuristic_baseline  -- old heuristic policy, byte-identical (regression).
  * learned_full_rerank -- controller decides VerifyOld; learned re-ranks the
                           same candidates (harm gate + max net_value), always
                           still does exactly one VerifyOld.
  * learned_full_filter -- controller decides VerifyOld; learned verifies only a
                           candidate that passes net_value>0 AND harm gate, else
                           replaces the verify with the controller's best AskNew
                           (never Stop).
  * learned_norag_filter-- same as filter but the frozen no-RAG model.

Red lines honoured: validate split only (never test), no oracle in any policy,
no LLM / dense / SFT / RL, no retraining, no tau_harm change, AskNew EIG
untouched, Stop thresholds untouched, learned never opens a VerifyOld, no
unified-Brier AskNew pricing, no git commit, writes only here.
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
from powerful_medrag.schema import UNKNOWN, FeatureKey
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator
from powerful_medrag.worthiness_dropin import (
    DropinStrategy,
    build_dropin_policy,
)
from powerful_medrag.worthiness_policy import load_frozen_worthiness_model

OUT = Path(__file__).resolve().parent
MODEL_PATH = "data/ddxplus/model-full.json"
PATIENTS_PATH = "data/ddxplus/release_validate_patients.zip"
EVIDENCES_PATH = "data/ddxplus/release_evidences.json"
CORPUS_PATH = "data/medrag-textbooks"
# Frozen N=5 manifest source (first 5 per disease of the frozen N=20 list).
MANIFEST_SOURCE = (
    "artifacts/verimedrag-rag-joint-gate-screen-n5/screen_case_manifest.csv"
)

NOISE_RATES = (0.2, 0.3)
SEEDS = (2026, 2027, 2028)

STRATEGIES = (
    DropinStrategy.HEURISTIC_BASELINE.value,
    DropinStrategy.LEARNED_FULL_RERANK.value,
    DropinStrategy.LEARNED_FULL_FILTER.value,
    DropinStrategy.LEARNED_NORAG_FILTER.value,
)

# retrieval mode per strategy (full-RAG vs no-RAG is an end-to-end ablation).
RETRIEVAL_MODE = {
    DropinStrategy.HEURISTIC_BASELINE.value: RetrievalMode.DYNAMIC_RAG,
    DropinStrategy.LEARNED_FULL_RERANK.value: RetrievalMode.DYNAMIC_RAG,
    DropinStrategy.LEARNED_FULL_FILTER.value: RetrievalMode.DYNAMIC_RAG,
    DropinStrategy.LEARNED_NORAG_FILTER.value: RetrievalMode.NO_RAG,
}

# Only the full-RAG learned strategies carry RAG features in their frozen model.
_LEARNED_MODEL_KIND = {
    DropinStrategy.LEARNED_FULL_RERANK.value: "real",
    DropinStrategy.LEARNED_FULL_FILTER.value: "real",
    DropinStrategy.LEARNED_NORAG_FILTER.value: "none",
}


def build_config(strategy: str) -> ReliabilityAwarePolicyConfig:
    return ReliabilityAwarePolicyConfig(
        posterior_threshold=0.85,
        posterior_margin_threshold=0.70,
        minimum_action_utility=0.08,
        suspicious_report_threshold=0.05,
        verification_cost=0.03,
        minimum_unreliable_history_cues_for_verification=1,
        max_total_turns=15,
        retrieval_mode=RETRIEVAL_MODE[strategy],
        retrieval_top_k=10,
        query_disease_top_k=5,
        retrieval_gate_mode=RetrievalGateMode.JOINT_GATE,
        retrieval_impact_weight=1.0,
    )


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
# Evaluation-side helpers (true state enters ONLY here, never the policy)
# --------------------------------------------------------------------------- #


def _report_wrong(patient, evidence_code, report_value) -> int:
    """1 if the report's value contradicts the true latent state."""
    if evidence_code is None or report_value is None:
        return 0
    true_state = patient.latent_states.get(FeatureKey(evidence_code), UNKNOWN)
    return int(
        true_state != UNKNOWN
        and report_value != UNKNOWN
        and report_value != true_state
    )


def _make_policy(strategy, *, config, retriever, worthiness_real, worthiness_none):
    s = DropinStrategy(strategy)
    model = {
        DropinStrategy.LEARNED_FULL_RERANK: worthiness_real,
        DropinStrategy.LEARNED_FULL_FILTER: worthiness_real,
        DropinStrategy.LEARNED_NORAG_FILTER: worthiness_none,
    }.get(s)
    return build_dropin_policy(
        s, config=config, retriever=retriever, worthiness_model=model
    )


# --------------------------------------------------------------------------- #
# Per-trajectory run
# --------------------------------------------------------------------------- #


def run_one(case, noise, seed, strategy, model, retriever, worthiness_real,
            worthiness_none):
    config = build_config(strategy)
    patient = StructuredPatientSimulator(
        diagnosis=case.diagnosis,
        latent_states=case.states,
        model=model,
        profile=PatientProfile.from_noise_rate(noise),
        seed=_pilot_seed(seed, case.case_id, noise),
    )
    policy = _make_policy(
        strategy, config=config, retriever=retriever,
        worthiness_real=worthiness_real, worthiness_none=worthiness_none,
    )
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

    # Realized prediction-flip outcome of each verification (pre/post top-1 vs
    # true disease), keyed by 0-indexed turn position.
    turn_flip: dict[int, dict] = {}
    prev_belief = None
    for i, t in enumerate(result.turns):
        if t.action.kind is ActionKind.VERIFY and prev_belief is not None:
            pre_correct = max(prev_belief, key=prev_belief.__getitem__) == case.diagnosis
            post_correct = max(t.belief, key=t.belief.__getitem__) == case.diagnosis
            turn_flip[i] = {
                "pre_correct": int(pre_correct),
                "post_correct": int(post_correct),
                "correct_to_wrong": int(pre_correct and not post_correct),
                "wrong_to_correct": int(not pre_correct and post_correct),
            }
        prev_belief = t.belief
    correct_to_wrong_flips = sum(f["correct_to_wrong"] for f in turn_flip.values())
    wrong_to_correct_flips = sum(f["wrong_to_correct"] for f in turn_flip.values())

    # Action-type + verify-report sequences (from the executed turns).
    action_seq = "".join(
        "N" if t.action.kind is ActionKind.NEW else
        "V" if t.action.kind is ActionKind.VERIFY else "S"
        for t in result.turns
    )
    verify_report_seq = ",".join(
        str(t.action.report_index)
        for t in result.turns if t.action.kind is ActionKind.VERIFY
    )

    # Drop-in controller/final log (learned strategies only).  One entry per
    # ``choose_action`` call; entry ``i`` corresponds to ``result.turns[i]``.
    logs = getattr(policy, "last_dropin_log", [])
    logs = logs[: len(result.turns)]  # drop any trailing STOP decision

    verify_opportunities = sum(
        1 for l in logs if l["controller_action_type"] == "verify"
    ) if logs else result.verification_questions
    rerank_changed = sum(
        1 for l in logs
        if l["controller_action_type"] == "verify"
        and l["final_action_type"] == "verify"
        and l["learned_selected_report"] != l["heuristic_selected_report"]
    ) if logs else 0
    filter_rejected = sum(
        1 for l in logs if l["replaced_verify_with_asknew"]
    ) if logs else 0
    extra_stop = sum(
        1 for l in logs
        if l["final_action_type"] == "stop"
        and l["controller_action_type"] != "stop"
    ) if logs else 0
    learned_score_seconds = sum(
        l.get("learned_score_seconds", 0.0) for l in logs
    ) if logs else 0.0

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
        "verify_opportunities": verify_opportunities,
        "rerank_changed": rerank_changed,
        "filter_rejected": filter_rejected,
        "extra_stop": extra_stop,
        "action_seq": action_seq,
        "verify_report_seq": verify_report_seq,
        "learned_score_seconds": round(learned_score_seconds, 6),
        "wall_clock": round(wall, 6),
    }

    turn_rows: list[dict] = []
    candidate_rows: list[dict] = []

    if logs:
        # learned strategies: reconstruct per-verify-turn detail from the log.
        for i, log in enumerate(logs):
            if log["controller_action_type"] != "verify":
                continue
            t = result.turns[i]
            by_idx = {c["report_index"]: c for c in log["candidates"]}
            h_report = log["heuristic_selected_report"]
            l_report = log["learned_selected_report"]
            h_c = by_idx.get(h_report) if h_report is not None else None
            l_c = by_idx.get(l_report) if l_report is not None else None
            h_wrong = _report_wrong(
                patient,
                h_c["evidence_code"] if h_c else None,
                h_c["report_value"] if h_c else None,
            )
            # learned selected report wrong only when a verify was actually done
            l_wrong = (
                _report_wrong(
                    patient,
                    l_c["evidence_code"] if l_c else None,
                    l_c["report_value"] if l_c else None,
                )
                if log["final_action_type"] == "verify" and l_c is not None
                else 0
            )
            turn_rows.append(
                {
                    "case_id": case.case_id,
                    "diagnosis": case.diagnosis,
                    "strategy": strategy,
                    "noise_rate": noise,
                    "seed": seed,
                    "turn_index": i,
                    "controller_action_type": log["controller_action_type"],
                    "final_action_type": log["final_action_type"],
                    "heuristic_selected_report": h_report,
                    "learned_selected_report": l_report,
                    "heuristic_report_wrong": h_wrong,
                    "learned_report_wrong": l_wrong,
                    "fallback_to_heuristic": int(log["fallback_to_heuristic"]),
                    "replaced_verify_with_asknew": int(
                        log["replaced_verify_with_asknew"]
                    ),
                    "best_asknew_evidence": log.get("best_asknew_evidence") or "",
                    "num_candidates": len(log["candidates"]),
                    "learned_score_seconds": log.get("learned_score_seconds", 0.0),
                }
            )
            for c in log["candidates"]:
                candidate_rows.append(
                    {
                        "case_id": case.case_id,
                        "diagnosis": case.diagnosis,
                        "strategy": strategy,
                        "noise_rate": noise,
                        "seed": seed,
                        "turn_index": i,
                        "report_index": c["report_index"],
                        "evidence_code": c["evidence_code"],
                        "report_value": c["report_value"],
                        "gain_hat": c["gain_hat"],
                        "harm_hat": c["harm_hat"],
                        "net_value": c["net_value"],
                        "passed_gain_gate": int(c["passed_gain_gate"]),
                        "passed_harm_gate": int(c["passed_harm_gate"]),
                        "error_probability": c["error_probability"],
                        "diagnostic_influence": c["diagnostic_influence"],
                        "retrieval_impact": c["retrieval_impact"],
                        "selected": int(
                            l_report is not None
                            and c["report_index"] == l_report
                            and log["final_action_type"] == "verify"
                        ),
                        "report_wrong": _report_wrong(
                            patient, c["evidence_code"], c["report_value"]
                        ),
                    }
                )
    else:
        # baseline: no learned log; emit a per-verify-turn row from the turns.
        for i, t in enumerate(result.turns):
            if t.action.kind is not ActionKind.VERIFY:
                continue
            report_index = t.action.report_index
            turn_rows.append(
                {
                    "case_id": case.case_id,
                    "diagnosis": case.diagnosis,
                    "strategy": strategy,
                    "noise_rate": noise,
                    "seed": seed,
                    "turn_index": i,
                    "controller_action_type": "verify",
                    "final_action_type": "verify",
                    "heuristic_selected_report": report_index,
                    "learned_selected_report": report_index,
                    "heuristic_report_wrong": int(
                        not t.verification_was_unnecessary
                    ),
                    "learned_report_wrong": int(
                        not t.verification_was_unnecessary
                    ),
                    "fallback_to_heuristic": 0,
                    "replaced_verify_with_asknew": 0,
                    "best_asknew_evidence": "",
                    "num_candidates": "",
                    "learned_score_seconds": 0.0,
                }
            )
    return row, turn_rows, candidate_rows


# --------------------------------------------------------------------------- #
# Worker pool
# --------------------------------------------------------------------------- #

_WORKER = {}


def _init_worker(model, retriever, cases_by_id, worthiness_real, worthiness_none):
    _WORKER["model"] = model
    _WORKER["retriever"] = retriever
    _WORKER["cases_by_id"] = cases_by_id
    _WORKER["worthiness_real"] = worthiness_real
    _WORKER["worthiness_none"] = worthiness_none


def _worker_task(task):
    case_id, noise, seed, strategy = task
    case = _WORKER["cases_by_id"][case_id]
    return run_one(case, noise, seed, strategy, _WORKER["model"],
                   _WORKER["retriever"], _WORKER["worthiness_real"],
                   _WORKER["worthiness_none"])


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
    "stop_reason", "verify_opportunities", "rerank_changed", "filter_rejected",
    "extra_stop", "action_seq", "verify_report_seq", "learned_score_seconds",
    "wall_clock",
]

TURN_FIELDS = [
    "case_id", "diagnosis", "strategy", "noise_rate", "seed", "turn_index",
    "controller_action_type", "final_action_type",
    "heuristic_selected_report", "learned_selected_report",
    "heuristic_report_wrong", "learned_report_wrong",
    "fallback_to_heuristic", "replaced_verify_with_asknew",
    "best_asknew_evidence", "num_candidates", "learned_score_seconds",
]

CANDIDATE_FIELDS = [
    "case_id", "diagnosis", "strategy", "noise_rate", "seed", "turn_index",
    "report_index", "evidence_code", "report_value", "gain_hat", "harm_hat",
    "net_value", "passed_gain_gate", "passed_harm_gate", "error_probability",
    "diagnostic_influence", "retrieval_impact", "selected", "report_wrong",
]


def write_manifest(cases):
    rows = [
        {"case_id": c.case_id, "diagnosis": c.diagnosis, "split": "validate",
         "sample_seed": 2026, "cases_per_disease": len(cases) // 49}
        for c in cases
    ]
    write_csv(rows, OUT / "case_manifest.csv",
              fieldnames=["case_id", "diagnosis", "split", "sample_seed",
                          "cases_per_disease"])


def _run_grid(cases, noises, seeds, strategies, workers):
    cases_by_id = {c.case_id: c for c in cases}
    tasks = [
        (c.case_id, noise, seed, strategy)
        for c in cases
        for noise in noises
        for seed in seeds
        for strategy in strategies
    ]
    print(f"run: {len(tasks)} trajectories over {workers} workers")
    ctx = mp.get_context("fork")
    outcome_rows, turn_rows, candidate_rows = [], [], []
    walls = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(_WORKER["model"], _WORKER["retriever"], cases_by_id,
                            _WORKER["worthiness_real"],
                            _WORKER["worthiness_none"])) as pool:
        for i, (row, turns, candidates) in enumerate(
            pool.imap_unordered(_worker_task, tasks, chunksize=4)
        ):
            outcome_rows.append(row)
            turn_rows.extend(turns)
            candidate_rows.extend(candidates)
            walls.append(row["wall_clock"])
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} done "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    write_csv(outcome_rows, OUT / "case_outcomes.csv", OUTCOME_FIELDS)
    write_csv(turn_rows, OUT / "verification_turns.csv", TURN_FIELDS)
    write_csv(candidate_rows, OUT / "verification_candidates.csv",
              CANDIDATE_FIELDS)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    mean_wall = sum(walls) / len(walls) if walls else 0.0
    print(f"done in {elapsed:.0f}s; {len(outcome_rows)} case rows, "
          f"{len(turn_rows)} turn rows, {len(candidate_rows)} candidate rows; "
          f"mean wall {mean_wall:.2f}s; peak {peak_mb:.0f}MB")
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


# --------------------------------------------------------------------------- #
# 49-case validation smoke + checks
# --------------------------------------------------------------------------- #

_PHASE5_OUTCOMES = (
    "artifacts/verimedrag-worthiness-online-screen-n5/case_outcomes.csv"
)


def _load_smoke_outcomes():
    rows = list(csv.DictReader(open(OUT / "case_outcomes.csv", newline="")))
    return rows


def smoke_checks(cases, workers):
    """Run the 49-case validation smoke (noise=0.3, seed=2026) and gate checks."""
    _run_grid(cases, noises=(0.3,), seeds=(2026,), strategies=list(STRATEGIES),
              workers=workers)
    rows = _load_smoke_outcomes()
    base = [r for r in rows if r["strategy"] == DropinStrategy.HEURISTIC_BASELINE.value]

    def mean(strategy, key):
        vals = [float(r[key]) for r in rows if r["strategy"] == strategy]
        return sum(vals) / len(vals) if vals else float("nan")

    def frac(strategy, key):
        vals = [float(r[key]) for r in rows if r["strategy"] == strategy]
        return sum(vals) / len(vals) if vals else float("nan")

    checks = []

    # 1. heuristic regression vs Phase 5 heuristic (noise=0.3, seed=2026).
    p5 = {
        r["case_id"]: r
        for r in csv.DictReader(open(_PHASE5_OUTCOMES, newline=""))
        if r["strategy"] == "heuristic_verify"
        and abs(float(r["noise_rate"]) - 0.3) < 1e-9
        and r["seed"] == "2026"
    }
    mismatches = []
    missing = []
    for r in base:
        o = p5.get(r["case_id"])
        if o is None:
            missing.append(r["case_id"])
            continue
        for key in ("predicted_diagnosis", "correct_top1", "brier_score",
                    "new_questions", "verification_questions"):
            if str(r[key]) != str(o[key]):
                mismatches.append((r["case_id"], key, r[key], o[key]))
    regression_ok = len(mismatches) == 0 and not missing
    checks.append(("heuristic_regression", regression_ok,
                   f"{len(mismatches)} mismatches, {len(missing)} missing "
                   f"of {len(base)} cases"))

    # 2. learned question count must not collapse (spec: < heuristic by >1.0).
    base_q = mean(DropinStrategy.HEURISTIC_BASELINE.value, "total_atomic_questions")
    q_deltas = {}
    for s in (DropinStrategy.LEARNED_FULL_RERANK.value,
              DropinStrategy.LEARNED_FULL_FILTER.value,
              DropinStrategy.LEARNED_NORAG_FILTER.value):
        q_deltas[s] = mean(s, "total_atomic_questions") - base_q
    q_ok = all(d > -1.0 for d in q_deltas.values())
    checks.append(("question_budget_preserved", q_ok,
                   f"base_q={base_q:.2f}; deltas={ {k: round(v,3) for k, v in q_deltas.items()} }"))

    # 3. learned never opens a Stop (extra_stop == 0, from the drop-in log).
    extra_stop = sum(int(r["extra_stop"]) for r in rows)
    no_stop_ok = extra_stop == 0
    checks.append(("no_extra_stop", no_stop_ok, f"extra_stop total={extra_stop}"))

    # 4. rerank changes some report_index.
    rerank_changed = frac(DropinStrategy.LEARNED_FULL_RERANK.value, "rerank_changed")
    rerank_ok = rerank_changed > 0
    checks.append(("rerank_changes_report", rerank_ok,
                   f"mean rerank_changed={rerank_changed:.3f}"))

    # 5. filter rejects some verifications (replace with AskNew).
    filter_rejected = frac(DropinStrategy.LEARNED_FULL_FILTER.value, "filter_rejected")
    filter_ok = filter_rejected > 0
    checks.append(("filter_replaces_verify", filter_ok,
                   f"mean filter_rejected={filter_rejected:.3f}"))

    # 6. informational: realized premature-stop rate (downstream diagnostic, not
    # an isolation gate -- the isolation gate is extra_stop==0 above).
    base_ps = frac(DropinStrategy.HEURISTIC_BASELINE.value, "premature_stop")
    ps_detail = {s: round(frac(s, "premature_stop"), 3)
                 for s in (DropinStrategy.LEARNED_FULL_RERANK.value,
                           DropinStrategy.LEARNED_FULL_FILTER.value,
                           DropinStrategy.LEARNED_NORAG_FILTER.value)}
    checks.append(("premature_stop_rate_info", True,
                   f"base={base_ps:.3f}; learned={ps_detail}"))

    lines = ["Phase 6 -- 49-case smoke checks", ""]
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= bool(ok)
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    # hard gate on question budget
    lines.append("")
    lines.append("GATE question_budget_preserved "
                 f"{'PASS' if q_ok else 'FAIL'}")
    (OUT / "smoke_checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    print(f"\nsmoke {'PASS' if all_ok else 'FAIL'}")
    return 0 if (all_ok and q_ok) else 1


def write_config():
    config = {
        "phase": "worthiness-dropin-screen-n5",
        "scope": "strictly-isolated VerifyOld drop-in (rerank / filter); N=5 screen",
        "strategies": STRATEGIES,
        "retrieval_mode_by_strategy": {
            k: v.value for k, v in RETRIEVAL_MODE.items()
        },
        "cases": 245,
        "cases_per_disease": 5,
        "smoke_cases": 49,
        "noise_rates": list(NOISE_RATES),
        "seeds": list(SEEDS),
        "max_total_turns": 15,
        "trajectories": 245 * len(NOISE_RATES) * len(SEEDS) * len(STRATEGIES),
        "policy": {
            "posterior_threshold": 0.85,
            "posterior_margin_threshold": 0.70,
            "minimum_action_utility": 0.08,
            "suspicious_report_threshold": 0.05,
            "verification_cost": 0.03,
            "max_total_turns": 15,
            "retrieval_top_k": 10,
            "query_disease_top_k": 5,
            "retrieval_gate_mode": "joint_gate",
            "retrieval_impact_weight": 1.0,
        },
        "frozen": {
            "gain_model": "hgb",
            "harm_model": "hgb",
            "tau_harm_full_rag": 0.41292830528839586,
            "tau_harm_no_rag": 0.7935174855595156,
            "verification_cost": 0.03,
            "model_dir": "artifacts/verimedrag-verification-worthiness-offline/models",
        },
        "isolation": [
            "controller = old heuristic decides action TYPE (AskNew/VerifyOld/Stop)",
            "learned acts only when controller == VerifyOld",
            "rerank: same candidates, harm gate + max net_value, always 1 VerifyOld",
            "filter: verify if net_value>0 AND harm gate, else replace with AskNew",
            "learned never opens a VerifyOld, never changes AskNew/Stop",
            "no unified-Brier AskNew pricing (AskNew EIG + thresholds unchanged)",
        ],
        "red_lines": [
            "no test split", "no retraining", "no feature/model change",
            "no tau_harm change", "no AskNew EIG reorder", "no Stop change",
            "no true state in prediction", "no oracle correction",
            "no learned default", "no git commit",
        ],
    }
    (OUT / "CONFIG.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = load_model()
    print(f"model loaded ({time.perf_counter()-t0:.2f}s): {len(model.diseases)} diseases")
    t0 = time.perf_counter()
    retriever = MedicalRetriever(CORPUS_PATH)
    print(f"retriever built ({time.perf_counter()-t0:.2f}s): {len(retriever)} snippets")

    # preload frozen models once, then fork
    worthiness_real = load_frozen_worthiness_model("real")
    worthiness_none = load_frozen_worthiness_model("none")
    print("frozen worthiness models loaded")

    _WORKER["model"] = model
    _WORKER["retriever"] = retriever
    _WORKER["worthiness_real"] = worthiness_real
    _WORKER["worthiness_none"] = worthiness_none

    if args.mode == "smoke":
        cases = load_selected_cases(model, per_disease=1)
        print(f"smoke: {len(cases)} cases (1/disease)")
        write_manifest(cases)
        write_config()
        return smoke_checks(cases, args.workers)

    cases = load_selected_cases(model, per_disease=5)
    print(f"full: {len(cases)} cases (5/disease)")
    write_manifest(cases)
    write_config()
    _run_grid(cases, NOISE_RATES, SEEDS, list(STRATEGIES), args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
