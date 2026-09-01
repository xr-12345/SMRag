"""VeriMedRAG Phase 2B — N=5 retrieval joint-gate screening runner.

Runs the retrieval-aware reliability dialogue over the frozen DDXPlus validation
manifest (first 5 cases per disease, 245 cases) under 7 dynamic_rag configs and
two noise rates (0.2 / 0.3), paired at the case level.

Two modes:
  * ``cost-audit``  — a small single-process sample that measures the retrieval
    cost profile (index build, cold/warm latency, counterfactual volume, cache
    hit rate, wall-clock, peak memory) and projects N=20 runtime.
  * ``full``        — the complete 245 x 2 x 7 run (fork-parallel across cases).

Red lines honoured: validate split only (never test), no oracle in the policy,
no LLM / dense / SFT / RL, no git commit, writes only into this directory.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import resource
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
NOISE_RATES = (0.2, 0.3)
PATIENT_SEED = 2026

# ---- frozen screening configs (label -> gate mode, retrieval_impact_weight) ----
CONFIGS = {
    "rank_only_w0.0": (RetrievalGateMode.RANK_ONLY, 0.0),
    "rank_only_w0.1": (RetrievalGateMode.RANK_ONLY, 0.1),
    "rank_only_w0.5": (RetrievalGateMode.RANK_ONLY, 0.5),
    "rank_only_w1.0": (RetrievalGateMode.RANK_ONLY, 1.0),
    "joint_gate_w0.1": (RetrievalGateMode.JOINT_GATE, 0.1),
    "joint_gate_w0.5": (RetrievalGateMode.JOINT_GATE, 0.5),
    "joint_gate_w1.0": (RetrievalGateMode.JOINT_GATE, 1.0),
}
BASELINE_LABEL = "rank_only_w0.0"


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


def load_selected_cases(model: DiseaseStateModel) -> tuple[list, dict]:
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
    selected = [cid for d in sorted(manifest) for cid in manifest[d][:5]]
    missing = [cid for cid in selected if cid not in by_id]
    if missing:
        raise ValueError(f"manifest case_ids not reproduced: {missing[:5]}")
    return [by_id[cid] for cid in selected], by_id


class InstrumentedPolicy(ReliabilityAwareActionPolicy):
    """Tracks per-turn gate-opening and the counterfactual sweep volume."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = {
            "counterfactual_retrieves": 0,   # baseline + per-active-report deletes
            "delete_counterfactuals": 0,     # per-report "delete" interventions only
            "counterfactual_sweeps": 0,      # number of _retrieval_impacts() calls
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
        self.stats["counterfactual_retrieves"] += 1 + active  # baseline + deletes
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
        self.stats["turn_logs"].append(
            {
                "chosen_action": action.kind.value,
                "opened_by_retrieval": opened,
                "retrieval_triggered": bool(getattr(action, "retrieval_triggered", False)),
                "report_index": action.report_index,
                "top_verify_report_index": order[0] if order else None,
                "verify_order": "|".join(str(x) for x in order),
            }
        )
        return action


@dataclass
class CaseResult:
    row: dict
    turns: list[dict]


def run_one(case, noise: float, label: str, model, retriever) -> CaseResult:
    cfg = build_config(label)
    patient = StructuredPatientSimulator(
        diagnosis=case.diagnosis,
        latent_states=case.states,
        model=model,
        profile=PatientProfile.from_noise_rate(noise),
        seed=_pilot_seed(PATIENT_SEED, case.case_id, noise),
    )
    policy = InstrumentedPolicy(config=cfg, retriever=retriever)
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
        "config": label,
        "noise_rate": noise,
        "seed": PATIENT_SEED,
        "predicted_diagnosis": predicted,
        "correct_top1": int(predicted == case.diagnosis),
        "correct_top3": int(case.diagnosis in ranking[:3]),
        "brier_score": round(brier, 8),
        "new_questions": result.new_questions,
        "verification_questions": result.verification_questions,
        "total_atomic_questions": result.new_questions + result.verification_questions,
        "unnecessary_verifications": sum(
            t.verification_was_unnecessary for t in result.turns
        ),
        "resolved_wrong_reports": sum(
            t.verification_resolved_wrong_report for t in result.turns
        ),
        "premature_stop": int(predicted != case.diagnosis and confident_stop),
        "uncertain_output": int(result.uncertain_output),
        "retrieval_triggered_verifications": len(retrig),
        "retrieval_triggered_hits": len(retrig_hit),
        "gate_open_but_asknew": gate_open_asknew,
        "gate_open_and_verify": gate_open_verify,
        "gate_open_but_stop": gate_open_stop,
        "ordinary_retrieves": len(result.retrieval_log),  # one dynamic retrieve / loop
        "counterfactual_retrieves": policy.stats["counterfactual_retrieves"],
        "delete_counterfactuals": policy.stats["delete_counterfactuals"],
        "wall_clock": round(wall, 6),
    }
    turns = [
        {
            "case_id": case.case_id,
            "diagnosis": case.diagnosis,
            "config": label,
            "noise_rate": noise,
            "turn_index": i,
            **tl,
        }
        for i, tl in enumerate(logs)
    ]
    return CaseResult(row=row, turns=turns)


# ---- module-level globals set by the worker initializer (fork shares them) ----
_WORKER = {}


def _init_worker(model, retriever, cases_by_id):
    _WORKER["model"] = model
    _WORKER["retriever"] = retriever
    _WORKER["cases_by_id"] = cases_by_id


def _worker_task(task):
    case_id, noise, label = task
    case = _WORKER["cases_by_id"][case_id]
    res = run_one(case, noise, label, _WORKER["model"], _WORKER["retriever"])
    return res


# ---- cost audit ------------------------------------------------------------


class CountingRetriever:
    def __init__(self, retriever):
        self._r = retriever
        self.calls = 0
        self.hits = 0
        self.miss_seconds = 0.0

    def __len__(self):
        return len(self._r)

    def retrieve(self, query, k=10, *, use_cache=True):
        self.calls += 1
        if use_cache and query in self._r.cache:
            self.hits += 1
            return self._r.retrieve(query, k)
        t0 = time.perf_counter()
        result = self._r.retrieve(query, k)
        self.miss_seconds += time.perf_counter() - t0
        return result


def cost_audit(model, retriever, cases, sample_cases: int) -> None:
    print("=== cost audit ===")
    counting = CountingRetriever(retriever)
    # index build time (already built; re-time a fresh build on a copy is skipped)
    # cold single-query latency
    probe_q = "influenza do you have a fever are you experiencing muscle pain"
    t0 = time.perf_counter()
    counting.retrieve(probe_q)
    cold_latency = time.perf_counter() - t0
    t0 = time.perf_counter()
    counting.retrieve(probe_q)
    warm_latency = time.perf_counter() - t0
    print(f"index snippets: {len(retriever)}")
    print(f"cold single-query latency: {cold_latency*1000:.1f} ms")
    print(f"warm (cache hit) latency: {warm_latency*1000:.3f} ms")

    label = "joint_gate_w0.5"  # most expensive: joint gate + weight
    t_start = time.perf_counter()
    per_case = []
    for case in cases[:sample_cases]:
        for noise in NOISE_RATES:
            res = run_one(case, noise, label, model, counting)
            per_case.append((case.case_id, noise, res.row))
    total_wall = time.perf_counter() - t_start
    n_dialogues = len(per_case)
    mean_wall = total_wall / n_dialogues
    hit_rate = counting.hits / counting.calls if counting.calls else 0.0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    print(f"sample dialogues: {n_dialogues}")
    print(f"total wall: {total_wall:.1f}s  mean/dialogue: {mean_wall:.2f}s")
    print(f"retrieve calls={counting.calls} hits={counting.hits} hit_rate={hit_rate:.3f}")
    print(f"peak RSS: {peak_mb:.1f} MB")

    # counterfactual delete/weaken/flip split (current impl: delete-only)
    rows = [r for _, _, r in per_case]
    mean_cf = sum(r["counterfactual_retrieves"] for r in rows) / len(rows)
    mean_del = sum(r["delete_counterfactuals"] for r in rows) / len(rows)
    mean_ord = sum(r["ordinary_retrieves"] for r in rows) / len(rows)
    print(f"mean ordinary retrieves/dialogue: {mean_ord:.1f}")
    print(f"mean counterfactual retrieves/dialogue: {mean_cf:.1f}")
    print(f"delete/weaken/flip: {mean_del:.1f} / 0 / 0 (current impl uses delete only)")

    # projections
    n20_dialogues = 980 * 3 * 4 * len(CONFIGS)  # 980 cases x 3 seeds x 4 noise x 7 configs
    n5_dialogues = 245 * 2 * len(CONFIGS)
    proj_single = mean_wall * n20_dialogues / 3600.0
    print(f"N=5 screening dialogues: {n5_dialogues}")
    print(f"N=20 formal dialogues: {n20_dialogues}")
    print(f"projected N=20 single-threaded: {proj_single:.1f} h")
    for workers in (8, 12, 16):
        print(f"projected N=20 with {workers} workers: {proj_single/workers:.1f} h")

    # write retrieval_cost_profile.csv
    profile_rows = [
        {
            "metric": "index_snippets", "value": len(retriever),
        },
        {"metric": "cold_query_latency_ms", "value": round(cold_latency * 1000, 1)},
        {"metric": "warm_query_latency_ms", "value": round(warm_latency * 1000, 3)},
        {"metric": "cache_hit_rate", "value": round(hit_rate, 4)},
        {"metric": "mean_wall_seconds_per_dialogue", "value": round(mean_wall, 3)},
        {"metric": "mean_ordinary_retrieves", "value": round(mean_ord, 2)},
        {"metric": "mean_counterfactual_retrieves", "value": round(mean_cf, 2)},
        {"metric": "counterfactual_delete_count", "value": round(mean_del, 2)},
        {"metric": "counterfactual_weaken_count", "value": 0.0},
        {"metric": "counterfactual_flip_count", "value": 0.0},
        {"metric": "peak_rss_mb", "value": round(peak_mb, 1)},
        {"metric": "n5_screening_dialogues", "value": n5_dialogues},
        {"metric": "n20_formal_dialogues", "value": n20_dialogues},
        {"metric": "projected_n20_hours_single_thread", "value": round(proj_single, 1)},
    ]
    write_csv(profile_rows, OUT / "retrieval_cost_profile.csv")


# ---- full run --------------------------------------------------------------


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


def write_manifest(cases) -> None:
    rows = [
        {
            "case_id": case.case_id,
            "diagnosis": case.diagnosis,
            "split": "validate",
            "sample_seed": 2026,
            "cases_per_disease": 5,
        }
        for case in cases
    ]
    write_csv(rows, OUT / "screen_case_manifest.csv")
    print(f"wrote screen_case_manifest.csv: {len(rows)} cases")


def full_run(model, retriever, cases, workers: int) -> None:
    cases_by_id = {c.case_id: c for c in cases}
    tasks = [
        (case.case_id, noise, label)
        for case in cases
        for noise in NOISE_RATES
        for label in CONFIGS
    ]
    print(f"full run: {len(tasks)} dialogues over {workers} workers")
    ctx = mp.get_context("fork")
    outcome_rows = []
    turn_rows = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(model, retriever, cases_by_id)) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker_task, tasks, chunksize=8)):
            outcome_rows.append(res.row)
            turn_rows.extend(res.turns)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} done ({time.perf_counter()-t0:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    write_csv(outcome_rows, OUT / "case_outcomes.csv")
    write_csv(turn_rows, OUT / "turn_observations.csv")
    print(f"done in {elapsed:.0f}s; {len(outcome_rows)} case rows, {len(turn_rows)} turn rows")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("cost-audit", "full"), default="full")
    ap.add_argument("--sample-cases", type=int, default=15)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = load_model()
    print(f"model loaded ({time.perf_counter()-t0:.2f}s): {len(model.diseases)} diseases")
    t0 = time.perf_counter()
    retriever = MedicalRetriever(CORPUS_PATH)
    print(f"retriever built ({time.perf_counter()-t0:.2f}s): {len(retriever)} snippets")
    t0 = time.perf_counter()
    cases, _ = load_selected_cases(model)
    print(f"selected {len(cases)} cases ({time.perf_counter()-t0:.2f}s)")
    write_manifest(cases)

    if args.mode == "cost-audit":
        cost_audit(model, retriever, cases, args.sample_cases)
    else:
        full_run(model, retriever, cases, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
