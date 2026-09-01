"""VeriMedRAG phase-2A toy smoke: does the retrieval joint gate behave?

Three checks, all on the toy model + a tiny in-memory corpus:

  1. ``rank_only`` does NOT auto-open the gate on high-impact reports.
  2. ``joint_gate`` opens the gate on a high-risk + high-impact report.
  3. when AskNew utility is higher, the system still chooses AskNew.

Red lines honoured here:
  * toy model only — never the DDXPlus test/validate split;
  * the policy never reads latent state (no oracle_selection / oracle_correction);
  * writes only into this new artifacts directory.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from powerful_medrag.belief import BeliefTracker
from powerful_medrag.decision import (
    ActionKind,
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
    RetrievalGateMode,
    RetrievalMode,
)
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.retrieval import MedicalRetriever
from powerful_medrag.schema import CertaintyCue, FeatureKey, Observation

OUT = Path(__file__).resolve().parent

INTEG_CORPUS = [
    {"id": "doc_fever", "title": "Fever", "content": "fever",
     "contents": "fever is elevated body temperature"},
    {"id": "doc_myalgia", "title": "Myalgia", "content": "myalgia",
     "contents": "myalgia is muscle pain"},
]


def _write_corpus(tmpdir: str, docs: list[dict]) -> str:
    path = Path(tmpdir) / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc) + "\n")
    return tmpdir


def _policy(retriever, *, gate_mode, weight, min_util=0.03):
    return ReliabilityAwareActionPolicy(
        config=ReliabilityAwarePolicyConfig(
            retrieval_mode=RetrievalMode.DYNAMIC_RAG,
            retrieval_gate_mode=gate_mode,
            retrieval_impact_weight=weight,
            minimum_action_utility=min_util,
        ),
        retriever=retriever,
    )


def _rank(policy, model, reports):
    tracker = BeliefTracker(model)
    for obs in reports:
        tracker.update(obs)
    return policy.rank_actions(
        tracker,
        initial_observations=(),
        reports=tuple(reports),
        asked={r.key for r in reports},
        verified_report_indices=set(),
        verification_count=0,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory()
    retriever = MedicalRetriever(_write_corpus(tmp.name, INTEG_CORPUS))
    cases, specs = generate_toy_cases(cases_per_disease=30, seed=41)
    model = DiseaseStateModel.fit(cases, specs)
    c = CertaintyCue.CERTAIN

    def present(name: str) -> Observation:
        return Observation(FeatureKey(name), "present", certainty=c)

    # runny_nose + itchy_eyes push belief away from influenza, so fever=present
    # is both surprising (high error probability) and corpus-backed (deleting it
    # wipes retrieval -> impact 1.0).
    high_risk_high_impact = [present("runny_nose"), present("itchy_eyes"),
                             present("fever")]

    checks = {}

    # --- check 1: rank_only does not auto-open the gate -------------------- #
    rank_policy = _policy(retriever, gate_mode=RetrievalGateMode.RANK_ONLY, weight=2.0)
    rank_actions = _rank(rank_policy, model, high_risk_high_impact)
    rank_triggered = [
        a for a in rank_actions
        if a.kind is ActionKind.VERIFY and a.retrieval_triggered
    ]
    rank_log_opened = [
        e for e in rank_policy.last_verification_log if e["opened_by_retrieval"]
    ]
    checks["rank_only_does_not_auto_open"] = {
        "passed": not rank_triggered and not rank_log_opened,
        "n_retrieval_triggered": len(rank_triggered),
        "n_opened_by_retrieval_log": len(rank_log_opened),
        "best_action": rank_actions[0].kind.value if rank_actions else None,
    }

    # --- check 2: joint_gate opens the gate -------------------------------- #
    joint_policy = _policy(retriever, gate_mode=RetrievalGateMode.JOINT_GATE, weight=2.0)
    joint_actions = _rank(joint_policy, model, high_risk_high_impact)
    joint_triggered = [
        a for a in joint_actions
        if a.kind is ActionKind.VERIFY and a.retrieval_triggered
    ]
    fever_log = next(
        (e for e in joint_policy.last_verification_log
         if e["evidence_code"] == "fever"),
        None,
    )
    checks["joint_gate_opens"] = {
        "passed": bool(joint_triggered),
        "n_retrieval_triggered": len(joint_triggered),
        "best_action": joint_actions[0].kind.value if joint_actions else None,
        "best_report_index": joint_actions[0].report_index if joint_actions else None,
        "fever_activation_value": fever_log["retrieval_activation_value"] if fever_log else None,
        "fever_error_probability": fever_log["error_probability"] if fever_log else None,
        "fever_normalized_impact": fever_log["normalized_retrieval_impact"] if fever_log else None,
    }

    # --- check 3: AskNew still wins when its utility is higher ------------- #
    fever_only = [present("fever")]
    asknew_policy = _policy(
        retriever, gate_mode=RetrievalGateMode.JOINT_GATE, weight=0.0
    )
    asknew_actions = _rank(asknew_policy, model, fever_only)
    asknew_triggered = [
        a for a in asknew_actions
        if a.kind is ActionKind.VERIFY and a.retrieval_triggered
    ]
    best_is_new = bool(asknew_actions) and asknew_actions[0].kind is ActionKind.NEW
    checks["asknew_still_wins"] = {
        "passed": bool(asknew_triggered) and best_is_new,
        "gate_opened": bool(asknew_triggered),
        "best_action": asknew_actions[0].kind.value if asknew_actions else None,
        "best_asknew_utility": round(
            max((a.utility for a in asknew_actions if a.kind is ActionKind.NEW), default=0.0),
            6,
        ),
    }

    report = {
        "phase": "VeriMedRAG phase 2A (retrieval joint gate) toy smoke",
        "model": "toy (influenza / common_cold / allergic_rhinitis)",
        "corpus": "tiny in-memory corpus (fever + myalgia snippets only)",
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks.values()),
    }
    (OUT / "smoke_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
