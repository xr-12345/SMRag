"""VeriMedRAG Phase 2C — frozen-config N=20 confirmation runner.

Runs the reliability-aware dialogue over the full frozen DDXPlus validation
manifest (980 cases, 20/disease) under 3 frozen configs, 3 seeds, 4 noise rates
— 35,280 paired trajectories.  No tuning this phase.

Extra vs Phase 2B: 3 seeds, NLL / posterior margin / confidence per case, and
candidate-level logging so the retrieval-trigger precision vs candidate
misreport base rate can be computed (eval-side truth only).

Red lines honoured: validate split only, no oracle in the policy, no
LLM/dense/SFT/RL, no git commit, writes only into this directory.
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import resource
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from powerful_medrag.ddxplus import sample_balanced_ddxplus_cases
from powerful_medrag.decision import (
    ActionKind,
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    RetrievalGateMode,
    RetrievalMode,
    run_reliability_aware_dialogue,
)
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.retrieval import MedicalRetriever
from powerful_medrag.reliability_experiment import _pilot_seed
from powerful_medrag.schema import UNKNOWN
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator

OUT = Path(__file__).resolve().parent
MODEL_PATH = "data/ddxplus/model-full.json"
PATIENTS_PATH = "data/ddxplus/release_validate_patients.zip"
EVIDENCES_PATH = "data/ddxplus/release_evidences.json"
CORPUS_PATH = "data/medrag-textbooks"
MANIFEST_PATH = (
    "artifacts/verimedrag-hypothesis-validation-2026/config/validation_case_manifest.csv"
)
SCREEN_MANIFEST_PATH = (
    "artifacts/verimedrag-rag-joint-gate-screen-n5/screen_case_manifest.csv"
)
SEEDS = (2026, 2027, 2028)
NOISE_RATES = (0.0, 0.1, 0.2, 0.3)

CONFIGS = {
    "rank_only_w0.0": (RetrievalGateMode.RANK_ONLY, 0.0),
    "rank_only_w1.0": (RetrievalGateMode.RANK_ONLY, 1.0),
    "joint_gate_w1.0": (RetrievalGateMode.JOINT_GATE, 1.0),
}


def build_config(label: str) -> ReliabilityAwarePolicyConfig:
    gate_mode, weight = CONFIGS[label]
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
        retrieval_gate_mode=gate_mode,
        retrieval_impact_weight=weight,
    )


def load_model() -> DiseaseStateModel:
    return DiseaseStateModel.load(MODEL_PATH)


def load_all_cases(model: DiseaseStateModel) -> list:
    askable = {k for k, s in model.specs.items() if s.askable}
    cases = sample_balanced_ddxplus_cases(
        PATIENTS_PATH,
        EVIDENCES_PATH,
        cases_per_disease=20,
        seed=2026,
        available_features=askable,
    )
    by_id = {c.case_id: c for c in cases}
    manifest: dict[str, list[str]] = {}
    with open(MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            manifest.setdefault(row["diagnosis"], []).append(row["case_id"])
    ordered = [cid for d in sorted(manifest) for cid in manifest[d]]
    missing = [cid for cid in ordered if cid not in by_id]
    if missing:
        raise ValueError(f"manifest case_ids not reproduced: {missing[:5]}")
    return [by_id[cid] for cid in ordered]


def setup_manifests(cases: list) -> tuple[list, list]:
    """Write the confirmation/screen manifests; return (primary, full) case lists."""
    screen_ids = {
        row["case_id"] for row in csv.DictReader(open(SCREEN_MANIFEST_PATH, newline=""))
    }
    # copy the frozen validation manifest into this directory
    shutil.copyfile(MANIFEST_PATH, OUT / "validation_case_manifest.csv")
    primary = [c for c in cases if c.case_id not in screen_ids]
    full = list(cases)

    # confirmation_case_manifest.csv (735)
    write_csv(
        [
            {
                "case_id": c.case_id,
                "diagnosis": c.diagnosis,
                "split": "validate",
                "sample_seed": 2026,
                "cases_per_disease": 20,
                "in_primary_confirmation": int(c.case_id not in screen_ids),
            }
            for c in full
        ],
        OUT / "confirmation_case_manifest.csv",
    )
    # screening_overlap_check.csv
    overlap = sorted(screen_ids & {c.case_id for c in primary})
    write_csv(
        [
            {
                "screen_cases": len(screen_ids),
                "primary_cases": len(primary),
                "full_cases": len(full),
                "overlap_count": len(overlap),
                "overlap_ok": int(len(overlap) == 0),
            }
        ],
        OUT / "screening_overlap_check.csv",
    )
    print(
        f"manifests: full={len(full)} primary={len(primary)} screen={len(screen_ids)} "
        f"overlap={len(overlap)}"
    )
    if overlap:
        raise ValueError(f"primary subset overlaps screen subset: {overlap[:5]}")
    return primary, full


class InstrumentedPolicy(ReliabilityAwareActionPolicy):
    """Tracks per-turn gate-opening, counterfactual volume, and candidate lists."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = {
            "counterfactual_retrieves": 0,
            "delete_counterfactuals": 0,
            "counterfactual_sweeps": 0,
            "turn_logs": [],
        }

    def _retrieval_impacts(self, tracker, reports, verified_report_indices):
        active = sum(
            1
            for i, r in enumerate(reports)
            if i not in verified_report_indices and r.value != UNKNOWN
        )
        self.stats["counterfactual_sweeps"] += 1
        self.stats["delete_counterfactuals"] += active
        self.stats["counterfactual_retrieves"] += 1 + active
        return super()._retrieval_impacts(tracker, reports, verified_report_indices)

    def choose_action(self, tracker, *, initial_observations, reports, asked,
                      verified_report_indices, verification_count,
                      oracle_states=None, report_risks=()):
        action = super().choose_action(
            tracker,
            initial_observations=initial_observations,
            reports=reports,
            asked=asked,
            verified_report_indices=verified_report_indices,
            verification_count=verification_count,
            oracle_states=oracle_states,
            report_risks=report_risks,
        )
        log = list(self.last_verification_log)
        opened = any(bool(e.get("opened_by_retrieval")) for e in log)
        order = tuple(
            e["report_index"]
            for e in sorted(log, key=lambda e: -e["final_verification_utility"])
        )
        candidates = [
            (e["report_index"], e["error_probability"], e["normalized_retrieval_impact"])
            for e in log
        ]
        reports_snapshot = [(r.key.token, r.value) for r in reports]
        self.stats["turn_logs"].append(
            {
                "chosen_action": action.kind.value,
                "opened_by_retrieval": opened,
                "retrieval_triggered": bool(getattr(action, "retrieval_triggered", False)),
                "report_index": action.report_index,
                "top_verify_report_index": order[0] if order else None,
                "verify_order": "|".join(str(x) for x in order),
                "candidates": candidates,
                "reports_snapshot": reports_snapshot,
            }
        )
        return action


@dataclass
class CaseResult:
    row: dict
    turns: list[dict]
    candidate_events: list[dict]


def run_one(case, seed: int, noise: float, label: str, model, retriever) -> CaseResult:
    cfg = build_config(label)
    patient = StructuredPatientSimulator(
        diagnosis=case.diagnosis,
        latent_states=case.states,
        model=model,
        profile=PatientProfile.from_noise_rate(noise),
        seed=_pilot_seed(seed, case.case_id, noise),
    )
    policy = InstrumentedPolicy(config=cfg, retriever=retriever)
    t0 = time.perf_counter()
    result = run_reliability_aware_dialogue(
        patient, policy=policy, initial_observations=case.initial_observations
    )
    wall = time.perf_counter() - t0

    ranking = sorted(result.belief, key=result.belief.__getitem__, reverse=True)
    predicted = ranking[0]
    top1_prob = result.belief[predicted]
    top2_prob = result.belief[ranking[1]] if len(ranking) > 1 else 0.0
    true_prob = result.belief[case.diagnosis]
    brier = sum(
        (result.belief[d] - (1.0 if d == case.diagnosis else 0.0)) ** 2
        for d in model.diseases
    )
    nll = -math.log(max(true_prob, 1e-12))
    confident_stop = ("posterior" in result.stop_reason) or (
        "confidence" in result.stop_reason
    )
    verifies = [t for t in result.turns if t.action.kind is ActionKind.VERIFY]
    retrig = [t for t in verifies if t.verification_triggered_by_retrieval]
    retrig_hit = [t for t in retrig if t.verification_resolved_wrong_report]
    logs = policy.stats["turn_logs"]
    gate_open_asknew = sum(
        1 for tl in logs if tl["opened_by_retrieval"] and tl["chosen_action"] == "new"
    )
    gate_open_verify = sum(
        1 for tl in logs if tl["opened_by_retrieval"] and tl["chosen_action"] == "verify"
    )
    gate_open_stop = sum(
        1 for tl in logs if tl["opened_by_retrieval"] and tl["chosen_action"] == "stop"
    )

    row = {
        "case_id": case.case_id,
        "diagnosis": case.diagnosis,
        "seed": seed,
        "noise_rate": noise,
        "config": label,
        "predicted_diagnosis": predicted,
        "correct_top1": int(predicted == case.diagnosis),
        "correct_top3": int(case.diagnosis in ranking[:3]),
        "confidence": round(top1_prob, 8),
        "posterior_margin": round(top1_prob - top2_prob, 8),
        "brier_score": round(brier, 8),
        "nll": round(nll, 8),
        "new_questions": result.new_questions,
        "verification_questions": result.verification_questions,
        "total_atomic_questions": result.new_questions + result.verification_questions,
        "interaction_turns": len(result.turns),
        "unnecessary_verifications": sum(
            t.verification_was_unnecessary for t in result.turns
        ),
        "resolved_wrong_reports": sum(
            t.verification_resolved_wrong_report for t in result.turns
        ),
        "premature_stop": int(predicted != case.diagnosis and confident_stop),
        "uncertain_output": int(result.uncertain_output),
        "stop_reason": result.stop_reason,
        "retrieval_triggered_verifications": len(retrig),
        "retrieval_triggered_hits": len(retrig_hit),
        "gate_open_but_asknew": gate_open_asknew,
        "gate_open_and_verify": gate_open_verify,
        "gate_open_but_stop": gate_open_stop,
        "ordinary_retrieves": len(result.retrieval_log),
        "counterfactual_retrieves": policy.stats["counterfactual_retrieves"],
        "delete_counterfactuals": policy.stats["delete_counterfactuals"],
        "wall_clock": round(wall, 6),
    }

    turns = [
        {
            "case_id": case.case_id,
            "diagnosis": case.diagnosis,
            "seed": seed,
            "noise_rate": noise,
            "config": label,
            "turn_index": i,
            "chosen_action": tl["chosen_action"],
            "opened_by_retrieval": tl["opened_by_retrieval"],
            "retrieval_triggered": tl["retrieval_triggered"],
            "report_index": tl["report_index"],
            "top_verify_report_index": tl["top_verify_report_index"],
            "verify_order": tl["verify_order"],
        }
        for i, tl in enumerate(logs)
    ]

    true_by_token = {k.token: v for k, v in case.states.items()}
    candidate_events = []
    for i, tl in enumerate(logs):
        snap = tl["reports_snapshot"]
        for (report_index, err_prob, norm_impact) in tl["candidates"]:
            if report_index is None or report_index >= len(snap):
                continue
            key_token, value = snap[report_index]
            true_state = true_by_token.get(key_token)
            comparable = (
                value != UNKNOWN
                and true_state is not None
                and true_state != UNKNOWN
            )
            is_wrong = comparable and value != true_state
            was_verified = (
                tl["chosen_action"] == "verify" and tl["report_index"] == report_index
            )
            candidate_events.append(
                {
                    "case_id": case.case_id,
                    "diagnosis": case.diagnosis,
                    "seed": seed,
                    "noise_rate": noise,
                    "config": label,
                    "turn_index": i,
                    "report_index": report_index,
                    "error_probability": round(err_prob, 8),
                    "normalized_retrieval_impact": round(norm_impact, 8),
                    "opened_by_retrieval": tl["opened_by_retrieval"],
                    "was_verified": int(was_verified),
                    "was_triggered": int(tl["retrieval_triggered"] and was_verified),
                    "is_comparable": int(comparable),
                    "is_actually_wrong": int(is_wrong),
                }
            )
    return CaseResult(row=row, turns=turns, candidate_events=candidate_events)


_WORKER = {}


def _init_worker(model, retriever, cases_by_id):
    _WORKER["model"] = model
    _WORKER["retriever"] = retriever
    _WORKER["cases_by_id"] = cases_by_id


def _worker_task(task):
    case_id, seed, noise, label = task
    case = _WORKER["cases_by_id"][case_id]
    return run_one(case, seed, noise, label, _WORKER["model"], _WORKER["retriever"])


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class CountingRetriever:
    def __init__(self, retriever):
        self._r = retriever
        self.calls = 0
        self.hits = 0

    def __len__(self):
        return len(self._r)

    def retrieve(self, query, k=10, *, use_cache=True):
        self.calls += 1
        if use_cache and query in self._r.cache:
            self.hits += 1
        return self._r.retrieve(query, k)


def runtime_audit(model, retriever, cases, sample_cases: int) -> None:
    print("=== runtime audit ===")
    counting = CountingRetriever(retriever)
    label = "joint_gate_w1.0"  # most expensive: joint gate + weight 1.0
    seed = 2026
    noise = 0.2
    probe_q = "influenza do you have a fever are you experiencing muscle pain"
    t0 = time.perf_counter()
    counting.retrieve(probe_q)
    cold_latency = time.perf_counter() - t0
    t0 = time.perf_counter()
    counting.retrieve(probe_q)
    warm_latency = time.perf_counter() - t0
    per_case = []
    for case in cases[:sample_cases]:
        res = run_one(case, seed, noise, label, model, counting)
        per_case.append(res.row)
    total = sum(r["wall_clock"] for r in per_case)
    hit_rate = counting.hits / counting.calls if counting.calls else 0.0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    rows = [
        {"metric": "index_snippets", "value": len(retriever)},
        {"metric": "cold_query_latency_ms", "value": round(cold_latency * 1000, 1)},
        {"metric": "warm_query_latency_ms", "value": round(warm_latency * 1000, 3)},
        {"metric": "cache_hit_rate", "value": round(hit_rate, 4)},
        {"metric": "mean_wall_seconds_per_dialogue", "value": round(total / len(per_case), 3)},
        {"metric": "peak_rss_mb", "value": round(peak_mb, 1)},
        {"metric": "mean_ordinary_retrieves", "value": round(sum(r["ordinary_retrieves"] for r in per_case) / len(per_case), 2)},
        {"metric": "mean_counterfactual_retrieves", "value": round(sum(r["counterfactual_retrieves"] for r in per_case) / len(per_case), 2)},
        {"metric": "counterfactual_delete_count", "value": round(sum(r["delete_counterfactuals"] for r in per_case) / len(per_case), 2)},
        {"metric": "counterfactual_weaken_count", "value": 0.0},
        {"metric": "counterfactual_flip_count", "value": 0.0},
        {"metric": "n20_total_trajectories", "value": 980 * 3 * 4 * 3},
        {"metric": "projected_n20_hours_single_thread", "value": round((total / len(per_case)) * 35280 / 3600, 1)},
    ]
    write_csv(rows, OUT / "runtime_profile.csv")
    print(f"cache hit rate: {hit_rate:.3f}; mean wall/dialogue: {total/len(per_case):.2f}s")
    print(f"projected N=20 single-thread: {rows[-1]['value']} h")


def full_run(model, retriever, cases, workers: int) -> None:
    cases_by_id = {c.case_id: c for c in cases}
    tasks = [
        (case.case_id, seed, noise, label)
        for case in cases
        for seed in SEEDS
        for noise in NOISE_RATES
        for label in CONFIGS
    ]
    print(f"full run: {len(tasks)} trajectories over {workers} workers")
    ctx = mp.get_context("fork")
    outcome_rows = []
    turn_rows = []
    candidate_rows = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(model, retriever, cases_by_id)) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker_task, tasks, chunksize=8)):
            outcome_rows.append(res.row)
            turn_rows.extend(res.turns)
            candidate_rows.extend(res.candidate_events)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(tasks)} done ({time.perf_counter()-t0:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    write_csv(outcome_rows, OUT / "case_outcomes.csv")
    write_csv(turn_rows, OUT / "turn_observations.csv")
    write_csv(candidate_rows, OUT / "candidate_events.csv")
    print(
        f"done in {elapsed:.0f}s; {len(outcome_rows)} case rows, "
        f"{len(turn_rows)} turn rows, {len(candidate_rows)} candidate rows"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("runtime-audit", "full"), default="full")
    ap.add_argument("--sample-cases", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = load_model()
    print(f"model loaded ({time.perf_counter()-t0:.2f}s): {len(model.diseases)} diseases")
    t0 = time.perf_counter()
    retriever = MedicalRetriever(CORPUS_PATH)
    print(f"retriever built ({time.perf_counter()-t0:.2f}s): {len(retriever)} snippets")
    t0 = time.perf_counter()
    cases = load_all_cases(model)
    print(f"loaded {len(cases)} cases ({time.perf_counter()-t0:.2f}s)")
    primary, full = setup_manifests(cases)

    if args.mode == "runtime-audit":
        runtime_audit(model, retriever, full, args.sample_cases)
    else:
        full_run(model, retriever, full, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
