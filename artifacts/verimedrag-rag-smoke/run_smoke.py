"""VeriMedRAG phase-1 N=1 smoke: does minimal BM25 retrieval enter the decision?

Runs one DDXPlus validate case through three retrieval modes (no / static /
dynamic), then measures whether retrieval changes the VerifyOld ranking and the
counterfactual retrieval-impact metrics.

Red lines honoured here:
  * uses release_validate_patients.zip only (never the test split);
  * the policy never reads latent state (no oracle_selection / oracle_correction);
  * writes only into this new artifacts directory.

Deterministic: one case, one seed, one noise rate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from powerful_medrag.belief import BeliefTracker
from powerful_medrag.channel import answer_channel_without_misreport
from powerful_medrag.decision import (
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    RetrievalMode,
    run_reliability_aware_dialogue,
)
from powerful_medrag.ddxplus import (
    iter_ddxplus_cases,
    load_ddxplus_condition_graph,
)
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.gating import HistoryReliabilityMisreportGate
from powerful_medrag.questioning import NumpyQuestionSelector
from powerful_medrag.retrieval import (
    MedicalRetriever,
    apply_counterfactual,
    build_retrieval_query,
    counterfactual_retrieval,
    jaccard,
    ndcg,
    rbo,
)
from powerful_medrag.schema import UNKNOWN
from powerful_medrag.simulator import PatientProfile, StructuredPatientSimulator

REPO = Path("/Users/xr-12345/Desktop/SafeMedRAG")
DDX = REPO / "data" / "ddxplus"
CORPUS = REPO / "data" / "medrag-textbooks"
OUT = REPO / "artifacts" / "verimedrag-rag-smoke"

NOISE = 0.2
SEED = 2026
MAX_TURNS = 8
RETRIEVAL_IMPACT_WEIGHT = 0.5


def _snapshot(patient, model, initial_observations, n_questions):
    """Manually ask n questions to build a mid-dialogue report snapshot."""
    tracker = BeliefTracker(
        model,
        answer_channel_without_misreport(),
        misreport_gate=HistoryReliabilityMisreportGate(),
    )
    for obs in initial_observations:
        tracker.update(obs)
    asked = {obs.key for obs in initial_observations}
    selector = NumpyQuestionSelector()
    reports = []
    for _ in range(n_questions):
        question = selector.rank(tracker, excluded=asked)[0]
        obs, _mode = patient.answer(question.key)
        reports.append(obs)
        tracker.update(obs)
        asked.add(question.key)
    return tracker, reports


def _replay_belief(model, initial_observations, reports):
    tracker = BeliefTracker(
        model,
        answer_channel_without_misreport(),
        misreport_gate=HistoryReliabilityMisreportGate(),
    )
    for obs in initial_observations:
        tracker.update(obs)
    for obs in reports:
        tracker.update(obs)
    return tracker.belief


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    model = DiseaseStateModel.load(DDX / "model-full.json")
    case = next(
        iter_ddxplus_cases(
            DDX / "release_validate_patients.zip",
            DDX / "release_evidences.json",
            limit=1,
        )
    )
    graph = load_ddxplus_condition_graph(
        DDX / "release_conditions.json", available_features=set(model.specs)
    )
    feature_names = {
        key: spec.question or key.name for key, spec in model.specs.items()
    }

    # ---- Part A: data scale + runtime ------------------------------------- #
    t0 = time.perf_counter()
    retriever = MedicalRetriever(CORPUS)
    index_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    retriever.retrieve("fever cough", k=10)
    retrieve_ms = (time.perf_counter() - t0) * 1000

    def make_patient():
        return StructuredPatientSimulator(
            diagnosis=case.diagnosis,
            latent_states=case.states,
            model=model,
            profile=PatientProfile.from_noise_rate(NOISE),
            seed=SEED,
        )

    # ---- Part B: three-mode dialogue -------------------------------------- #
    modes = {
        "no_rag": ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(max_total_turns=MAX_TURNS),
        ),
        "static_rag": ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(
                max_total_turns=MAX_TURNS,
                retrieval_mode=RetrievalMode.STATIC_RAG,
                retrieval_impact_weight=RETRIEVAL_IMPACT_WEIGHT,
            ),
            retriever=retriever,
        ),
        "dynamic_rag": ReliabilityAwareActionPolicy(
            config=ReliabilityAwarePolicyConfig(
                max_total_turns=MAX_TURNS,
                retrieval_mode=RetrievalMode.DYNAMIC_RAG,
                retrieval_impact_weight=RETRIEVAL_IMPACT_WEIGHT,
            ),
            retriever=retriever,
        ),
    }
    dialogue = {}
    for name, policy in modes.items():
        t0 = time.perf_counter()
        result = run_reliability_aware_dialogue(
            make_patient(),
            policy=policy,
            initial_observations=case.initial_observations,
        )
        wall = time.perf_counter() - t0
        ranked = sorted(result.belief, key=result.belief.__getitem__, reverse=True)
        verify_turns = [
            (turn.action.report_index, turn.observation.key.name)
            for turn in result.turns
            if turn.action.kind.value == "verify"
        ]
        dialogue[name] = {
            "wall_seconds": round(wall, 3),
            "predicted": result.predicted_diagnosis,
            "correct": result.correct,
            "top3": ranked[:3],
            "top1_probability": round(result.belief[ranked[0]], 4),
            "new_questions": result.new_questions,
            "verification_questions": result.verification_questions,
            "verified_report_indices": [i for i, _ in verify_turns],
            "verified_report_names": [n for _, n in verify_turns],
            "stop_reason": result.stop_reason,
            "n_retrieval_entries": len(result.retrieval_log),
            "queries": [e["query"] for e in result.retrieval_log],
        }

    # ---- Part C: retrieval impact into VerifyOld --------------------------- #
    patient = make_patient()
    tracker, reports = _snapshot(patient, model, case.initial_observations, 4)
    ranked_diseases = tracker.ranked_diseases()

    base_policy = ReliabilityAwareActionPolicy()
    rag_policy = ReliabilityAwareActionPolicy(
        config=ReliabilityAwarePolicyConfig(
            retrieval_mode=RetrievalMode.DYNAMIC_RAG,
            retrieval_impact_weight=RETRIEVAL_IMPACT_WEIGHT,
        ),
        retriever=retriever,
    )
    impacts = rag_policy._retrieval_impacts(tracker, tuple(reports), set())
    base_scores = base_policy._verification_scores(
        tracker,
        initial_observations=case.initial_observations,
        reports=tuple(reports),
        verified_report_indices=set(),
        report_risks=(),
    )
    rag_scores = rag_policy._verification_scores(
        tracker,
        initial_observations=case.initial_observations,
        reports=tuple(reports),
        verified_report_indices=set(),
        report_risks=(),
    )
    base_by_idx = {s.report_index: s for s in base_scores}
    rag_by_idx = {s.report_index: s for s in rag_scores}

    ranking = []
    for idx, report in enumerate(reports):
        ranking.append(
            {
                "report_index": idx,
                "feature": report.key.name,
                "value": report.value,
                "retrieval_impact": round(impacts.get(idx, 0.0), 4),
                "base_score": round(base_by_idx[idx].score, 6),
                "rag_score": round(rag_by_idx[idx].score, 6),
            }
        )
    ranking.sort(key=lambda r: r["rag_score"], reverse=True)
    base_order = sorted(base_by_idx, key=lambda i: base_by_idx[i].score, reverse=True)
    rag_order = sorted(rag_by_idx, key=lambda i: rag_by_idx[i].score, reverse=True)

    # ---- Part D: counterfactual metrics on the top-impact report ----------- #
    top_report = max(ranking, key=lambda r: r["retrieval_impact"])
    top_idx = top_report["report_index"]
    counterfactual = {}
    top_disease = ranked_diseases[0][0]
    graph_features = graph.get(top_disease, set())

    def evidence_score(rep_list):
        # Compare on base evidence codes (r.key.name) so multi-choice atomized
        # keys (name + {"option": ...}) still map onto the condition graph.
        positive = {
            r.key.name
            for r in rep_list
            if str(r.value).lower()
            not in {"absent", "no", "false", "negative", "none", UNKNOWN}
        }
        graph_codes = {f.name for f in graph_features}
        covered = len(positive & graph_codes)
        total = len(graph_codes)
        return covered / total if total else 0.0

    baseline_belief = _replay_belief(model, case.initial_observations, reports)
    for op in ("delete", "weaken", "flip"):
        cf = counterfactual_retrieval(
            retriever,
            tuple(reports),
            ranked_diseases,
            top_idx,
            op,
            top_k=5,
            k=10,
            feature_names=feature_names,
        )
        altered = apply_counterfactual(tuple(reports), top_idx, op)
        altered_belief = _replay_belief(model, case.initial_observations, altered)
        posterior_delta = abs(
            baseline_belief.get(top_disease, 0.0) - altered_belief.get(top_disease, 0.0)
        )
        counterfactual[op] = {
            "before_query": cf["before_query"],
            "after_query": cf["after_query"],
            "topk_jaccard": round(cf["change"]["topk_jaccard"], 4),
            "rbo": round(cf["change"]["rbo"], 4),
            "ndcg_change": round(cf["change"]["ndcg_change"], 4),
            "disease_evidence_score_before": round(evidence_score(reports), 4),
            "disease_evidence_score_after": round(evidence_score(altered), 4),
            "posterior_delta_top1_disease": round(posterior_delta, 6),
        }

    report = {
        "case": {
            "case_id": case.case_id,
            "diagnosis": case.diagnosis,
            "n_initial_observations": len(case.initial_observations),
            "n_states": len(case.states),
        },
        "data_scale": {
            "corpus_snippets": len(retriever),
            "corpus_bytes": sum(p.stat().st_size for p in CORPUS.glob("*.jsonl")),
            "index_build_seconds": round(index_seconds, 2),
            "retrieve_ms": round(retrieve_ms, 1),
        },
        "dialogue_by_mode": dialogue,
        "verifyold_ranking": {
            "top_disease": top_disease,
            "base_order": base_order,
            "rag_order": rag_order,
            "ranking_changed": base_order != rag_order,
            "rows": ranking,
        },
        "counterfactual_on_top_report": {
            "report_index": top_idx,
            "feature": top_report["feature"],
            "ops": counterfactual,
        },
    }
    (OUT / "smoke_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
