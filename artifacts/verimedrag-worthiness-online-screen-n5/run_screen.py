"""Phase 5 -- N=5 online learned-worthiness integration screen.

Runs the four strategies over the frozen DDXPlus validation manifest (first 5
cases per disease, 245 cases) at noise {0.2, 0.3} x seeds {2026, 2027, 2028},
fully paired at the case level: 245 x 2 x 3 x 4 = 5,880 trajectories.

Strategies (see ``powerful_medrag.worthiness_policy``):
  * heuristic_verify            -- existing policy, unchanged (baseline).
  * model_based_vbayes_verify   -- AskNew by EIG, VerifyOld by deployable V_Bayes.
  * learned_worthiness_full_rag -- frozen full-RAG model + real retrieval features.
  * learned_worthiness_no_rag   -- frozen no-RAG model, no retrieval features.

Red lines honoured: validate split only (never test), no oracle in any policy,
no LLM / dense / SFT / RL, no retraining, no git commit, writes only here.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

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
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator
from powerful_medrag.worthiness_policy import (
    WorthinessStrategy,
    LearnedWorthinessPolicy,
    ModelBasedVBayesPolicy,
    build_policy,
    load_frozen_worthiness_model,
)

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
    WorthinessStrategy.HEURISTIC_VERIFY.value,
    WorthinessStrategy.MODEL_BASED_VBAYES_VERIFY.value,
    WorthinessStrategy.LEARNED_WORTHINESS_FULL_RAG.value,
    WorthinessStrategy.LEARNED_WORTHINESS_NO_RAG.value,
)

# retrieval mode per strategy (dynamic RAG only where retrieval is used).
RETRIEVAL_MODE = {
    WorthinessStrategy.HEURISTIC_VERIFY.value: RetrievalMode.DYNAMIC_RAG,
    WorthinessStrategy.MODEL_BASED_VBAYES_VERIFY.value: RetrievalMode.NO_RAG,
    WorthinessStrategy.LEARNED_WORTHINESS_FULL_RAG.value: RetrievalMode.DYNAMIC_RAG,
    WorthinessStrategy.LEARNED_WORTHINESS_NO_RAG.value: RetrievalMode.NO_RAG,
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


def load_selected_cases(model: DiseaseStateModel) -> list:
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
    selected = [cid for d in sorted(manifest) for cid in manifest[d][:5]]
    missing = [cid for cid in selected if cid not in by_id]
    if missing:
        raise ValueError(f"manifest case_ids not reproduced: {missing[:5]}")
    return [by_id[cid] for cid in selected]


# --------------------------------------------------------------------------- #
# Recording policies (capture the per-turn verification log)
# --------------------------------------------------------------------------- #


class _RecordingMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._turn_logs = []

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
        self._turn_logs.append(
            {
                "chosen_action": action.kind.value,
                "chosen_report_index": action.report_index,
                "verification_log": [dict(e) for e in self.last_verification_log],
            }
        )
        return action


class RecordingHeuristic(_RecordingMixin, ReliabilityAwareActionPolicy):
    pass


class RecordingModelBased(_RecordingMixin, ModelBasedVBayesPolicy):
    pass


class RecordingLearned(_RecordingMixin, LearnedWorthinessPolicy):
    pass


def _make_recording_policy(strategy, *, config, retriever, worthiness_real,
                           worthiness_none):
    s = WorthinessStrategy(strategy)
    if s is WorthinessStrategy.HEURISTIC_VERIFY:
        return RecordingHeuristic(config=config, retriever=retriever)
    if s is WorthinessStrategy.MODEL_BASED_VBAYES_VERIFY:
        return RecordingModelBased(config=config, retriever=retriever)
    if s is WorthinessStrategy.LEARNED_WORTHINESS_FULL_RAG:
        return RecordingLearned(
            worthiness_real, config=config, retriever=retriever
        )
    if s is WorthinessStrategy.LEARNED_WORTHINESS_NO_RAG:
        return RecordingLearned(
            worthiness_none, config=config, retriever=retriever
        )
    raise ValueError(strategy)


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
    policy = _make_recording_policy(
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
    confident_stop = ("posterior" in result.stop_reason) or (
        "confidence" in result.stop_reason
    )
    unnecessary = sum(t.verification_was_unnecessary for t in result.turns)
    resolved = sum(t.verification_resolved_wrong_report for t in result.turns)
    total_turns = result.new_questions + result.verification_questions

    # Realized prediction-flip outcome of each verification (pre/post top-1 vs
    # the true disease).  ``turns[i-1].belief`` is the belief right before this
    # verification; ``turns[i].belief`` is the belief after it.
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
        "new_questions": result.new_questions,
        "verification_questions": result.verification_questions,
        "total_atomic_questions": total_turns,
        "interaction_turns": total_turns,
        "unnecessary_verifications": unnecessary,
        "resolved_wrong_reports": resolved,
        "correct_to_wrong_flips": correct_to_wrong_flips,
        "wrong_to_correct_flips": wrong_to_correct_flips,
        "premature_stop": int(predicted != case.diagnosis and confident_stop),
        "uncertain_output": int(result.uncertain_output),
        "stop_reason": result.stop_reason,
        "wall_clock": round(wall, 6),
    }

    candidates = []
    for turn_index, tl in enumerate(policy._turn_logs):
        chosen_report = tl["chosen_report_index"]
        flip = turn_flip.get(turn_index)
        for c in tl["verification_log"]:
            is_selected = (
                chosen_report is not None
                and c.get("report_index") == chosen_report
            )
            candidates.append(
                {
                    "case_id": case.case_id,
                    "diagnosis": case.diagnosis,
                    "strategy": strategy,
                    "noise_rate": noise,
                    "seed": seed,
                    "turn_index": turn_index,
                    "chosen_action": tl["chosen_action"],
                    "report_index": c.get("report_index"),
                    "evidence_code": c.get("evidence_code", ""),
                    "gain_hat": c.get("gain_hat", ""),
                    "harm_hat": c.get("harm_hat", ""),
                    "net_value": c.get("net_value", ""),
                    "verification_cost": c.get("verification_cost", ""),
                    "passed_gain_gate": c.get("passed_gain_gate", ""),
                    "passed_harm_gate": c.get("passed_harm_gate", ""),
                    "v_bayes_verify": c.get("v_bayes_verify", ""),
                    "heuristic_verify_utility": c.get(
                        "heuristic_verify_utility", ""
                    ),
                    "retrieval_impact": c.get("retrieval_impact", ""),
                    "error_prob_times_impact": c.get(
                        "error_prob_times_impact", ""
                    ),
                    "error_probability": c.get("error_probability", ""),
                    "diagnostic_influence": c.get("diagnostic_influence", ""),
                    "best_asknew_eig": c.get("best_asknew_eig", ""),
                    "best_asknew_v_bayes": c.get("best_asknew_v_bayes", ""),
                    "selected": int(is_selected),
                    "realized_correct_to_wrong": (
                        flip["correct_to_wrong"] if (is_selected and flip) else ""
                    ),
                    "realized_wrong_to_correct": (
                        flip["wrong_to_correct"] if (is_selected and flip) else ""
                    ),
                }
            )
    return row, candidates


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
    "brier_score", "new_questions", "verification_questions",
    "total_atomic_questions", "interaction_turns", "unnecessary_verifications",
    "resolved_wrong_reports", "correct_to_wrong_flips",
    "wrong_to_correct_flips", "premature_stop", "uncertain_output",
    "stop_reason", "wall_clock",
]

CANDIDATE_FIELDS = [
    "case_id", "diagnosis", "strategy", "noise_rate", "seed", "turn_index",
    "chosen_action", "report_index", "evidence_code", "gain_hat", "harm_hat",
    "net_value", "verification_cost", "passed_gain_gate", "passed_harm_gate",
    "v_bayes_verify", "heuristic_verify_utility", "retrieval_impact",
    "error_prob_times_impact", "error_probability", "diagnostic_influence",
    "best_asknew_eig", "best_asknew_v_bayes", "selected",
    "realized_correct_to_wrong", "realized_wrong_to_correct",
]


def write_manifest(cases):
    rows = [
        {"case_id": c.case_id, "diagnosis": c.diagnosis, "split": "validate",
         "sample_seed": 2026, "cases_per_disease": 5}
        for c in cases
    ]
    write_csv(rows, OUT / "case_manifest.csv",
              fieldnames=["case_id", "diagnosis", "split", "sample_seed",
                          "cases_per_disease"])


def full_run(model, retriever, cases, workers, worthiness_real, worthiness_none):
    cases_by_id = {c.case_id: c for c in cases}
    tasks = [
        (c.case_id, noise, seed, strategy)
        for c in cases
        for noise in NOISE_RATES
        for seed in SEEDS
        for strategy in STRATEGIES
    ]
    print(f"full run: {len(tasks)} trajectories over {workers} workers")
    ctx = mp.get_context("fork")
    outcome_rows = []
    candidate_rows = []
    walls = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(model, retriever, cases_by_id, worthiness_real,
                            worthiness_none)) as pool:
        for i, (row, candidates) in enumerate(
            pool.imap_unordered(_worker_task, tasks, chunksize=4)
        ):
            outcome_rows.append(row)
            candidate_rows.extend(candidates)
            walls.append(row["wall_clock"])
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} done "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    write_csv(outcome_rows, OUT / "case_outcomes.csv", OUTCOME_FIELDS)
    write_csv(candidate_rows, OUT / "verification_candidates.csv",
              CANDIDATE_FIELDS)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    mean_wall = sum(walls) / len(walls) if walls else 0.0
    print(f"done in {elapsed:.0f}s; {len(outcome_rows)} case rows, "
          f"{len(candidate_rows)} candidate rows; mean wall {mean_wall:.2f}s; "
          f"peak {peak_mb:.0f}MB")
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


def smoke(model, retriever, cases, worthiness_real, worthiness_none, n=3):
    print(f"=== smoke ({n} cases x 2 noise x 3 seeds x 4 strategies) ===")
    t0 = time.perf_counter()
    count = 0
    for case in cases[:n]:
        for noise in NOISE_RATES:
            for seed in SEEDS:
                for strategy in STRATEGIES:
                    row, candidates = run_one(
                        case, noise, seed, strategy, model, retriever,
                        worthiness_real, worthiness_none,
                    )
                    count += 1
                    print(f"  {count}: {row['case_id']} {strategy} "
                          f"noise={noise} seed={seed} "
                          f"-> {row['predicted_diagnosis']} "
                          f"(top1={row['correct_top1']}, "
                          f"new={row['new_questions']}, "
                          f"verify={row['verification_questions']}, "
                          f"wall={row['wall_clock']:.2f}s)")
    print(f"smoke done in {time.perf_counter()-t0:.0f}s for {count} trajectories")


def write_config():
    import json
    config = {
        "phase": "worthiness-online-screen-n5",
        "scope": "wire frozen learned worthiness into AskNew/VerifyOld/Stop; N=5 screen",
        "strategies": STRATEGIES,
        "retrieval_mode_by_strategy": {
            k: v.value for k, v in RETRIEVAL_MODE.items()
        },
        "cases": 245,
        "cases_per_disease": 5,
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
            "maximum_verifications": 1,
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
            "c_new": 0.03,
            "model_dir": "artifacts/verimedrag-verification-worthiness-offline/models",
        },
        "red_lines": [
            "no test split", "no retraining", "no feature/model change",
            "no tau_harm change", "no AskNew EIG reorder",
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
    t0 = time.perf_counter()
    cases = load_selected_cases(model)
    print(f"selected {len(cases)} cases ({time.perf_counter()-t0:.2f}s)")
    write_manifest(cases)
    write_config()

    if args.mode == "smoke":
        worthiness_real = load_frozen_worthiness_model("real")
        worthiness_none = load_frozen_worthiness_model("none")
        print("frozen worthiness models loaded")
        smoke(model, retriever, cases, worthiness_real, worthiness_none)
        return 0

    # full run: load frozen models once and fork
    worthiness_real = load_frozen_worthiness_model("real")
    worthiness_none = load_frozen_worthiness_model("none")
    print("frozen worthiness models loaded")
    full_run(model, retriever, cases, args.workers, worthiness_real,
             worthiness_none)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
