"""Phase 8B mechanism smoke (Cases A-E).

Proves, on the toy model, the five mechanisms the joint report channel is
supposed to deliver -- without touching the test split or running N=5/N=20:

* Case A: a single correct answer concentrates the disease posterior and keeps
  the joint misreport posterior below the suspicious threshold.
* Case B: a conflicting re-ask is *not* compressed to UNKNOWN; it resolves
  probabilistically (the correct re-ask shifts the belief toward the true disease).
* Case C: two consistent answers are a single factor and confirm the (correct)
  mode, so p_mode drops rather than rising.
* Case D: rho=0 makes the independent re-ask the product of the two marginals.
* Case E: the Stop reliability gate is wired to the joint misreport posterior and
  genuinely triggers when a strongly-contradicting answer follows strong evidence.

Writes ``smoke_metrics.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from powerful_medrag.belief import BeliefTracker
from powerful_medrag.channel import AnswerChannel
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.joint_channel_policy import JointChannelBrierAuditPolicy
from powerful_medrag.joint_reliability_belief import JointReliabilityBeliefTracker
from powerful_medrag.joint_report_channel import JointReportChannel, VerificationType
from powerful_medrag.schema import UNKNOWN, CertaintyCue, FeatureKey, Observation

HERE = Path(__file__).resolve().parent


def obs(name, value, certainty=CertaintyCue.CERTAIN):
    return Observation(FeatureKey(name), value, certainty)


def top(belief):
    return max(belief, key=belief.__getitem__)


def main() -> None:
    cases, specs = generate_toy_cases(cases_per_disease=100, seed=41)
    model = DiseaseStateModel.fit(cases, specs)
    chan = AnswerChannel()
    jc = JointReportChannel(chan, repeat_mode_persistence=0.5)
    fever = FeatureKey("fever")

    results: dict = {}

    # ---- Case A: single correct answer ------------------------------------ #
    t = JointReliabilityBeliefTracker(model, jc)
    t.observe_single(obs("fever", "present"))
    results["case_A_single_correct"] = {
        "top_disease": top(t.belief),
        "top_probability": round(t.belief[top(t.belief)], 6),
        "p_mode_misreported_fever": round(t.p_mode_misreported(fever), 6),
        "mechanism_ok": top(t.belief) == "influenza"
        and t.p_mode_misreported(fever) < 0.05,
    }

    # ---- Case B: conflicting re-ask is not compressed --------------------- #
    t = JointReliabilityBeliefTracker(model, jc)
    t.observe_single(obs("fever", "absent"))
    influenza_before = t.belief["influenza"]
    t.observe_verification(fever, obs("fever", "present"), VerificationType.REPEAT)
    bundle = t.memory[fever]
    answers_preserved = [a.value for a in bundle.all_answers] == ["absent", "present"]
    # the old UNKNOWN-compression path would contribute no evidence and leave
    # influenza at its post-single value; the joint path must move it.
    resolves_probabilistically = t.belief["influenza"] > influenza_before
    results["case_B_conflict_resolves"] = {
        "answers_preserved": answers_preserved,
        "all_answers": [a.value for a in bundle.all_answers],
        "influenza_before": round(influenza_before, 6),
        "influenza_after": round(t.belief["influenza"], 6),
        "mechanism_ok": answers_preserved and resolves_probabilistically,
    }

    # ---- Case C: consistent answers confirm the (correct) mode ------------ #
    t = JointReliabilityBeliefTracker(model, jc)
    t.observe_single(obs("fever", "absent"))
    p_mode_before = t.p_mode_misreported(fever)
    t.observe_verification(fever, obs("fever", "absent"), VerificationType.REPEAT)
    p_mode_after = t.p_mode_misreported(fever)
    results["case_C_consistent_confirms_mode"] = {
        "p_mode_before": round(p_mode_before, 6),
        "p_mode_after": round(p_mode_after, 6),
        "all_answers": [a.value for a in t.memory[fever].all_answers],
        "mechanism_ok": p_mode_after < p_mode_before,
    }

    # ---- Case D: rho=0 makes the independent re-ask the product ----------- #
    jc0 = JointReportChannel(chan, repeat_mode_persistence=0.0)
    joint = jc0.joint_probability(
        "present", "absent", "present", ("absent", "present")
    )
    product = jc0.single_probability("present", "present", ("absent", "present")) * (
        jc0.single_probability("absent", "present", ("absent", "present"))
    )
    results["case_D_independent_reask_is_product"] = {
        "joint": round(joint, 6),
        "product": round(product, 6),
        "mechanism_ok": abs(joint - product) < 1e-9,
    }

    # ---- Case E: reliability gate wired + triggers ------------------------ #
    t = JointReliabilityBeliefTracker(model, jc)
    for name in ("fever", "dry_cough", "myalgia"):
        t.observe_single(obs(name, "present"))
    t.observe_single(obs("itchy_eyes", "present"))
    itchy = FeatureKey("itchy_eyes")
    max_pmode = max(t.p_mode_misreported(k) for k in t.memory.keys())
    policy = JointChannelBrierAuditPolicy(model, channel=jc, tracker=t)
    policy.choose_action()
    log = policy.last_decision_log
    results["case_E_reliability_gate"] = {
        "p_mode_misreported_itchy_eyes": round(t.p_mode_misreported(itchy), 6),
        "max_mode_misreport": log["max_mode_misreport"],
        "suspicious_report_threshold": log["suspicious_report_threshold"],
        "reliability_ready": log["reliability_ready"],
        "base_stop_ready": log["base_stop_ready"],
        "mechanism_ok": (
            max_pmode > log["suspicious_report_threshold"]
            and log["reliability_ready"] is False
            and log["base_stop_ready"] is False
        ),
    }

    results["_summary"] = {
        "all_mechanisms_ok": all(
            results[case]["mechanism_ok"]
            for case in (
                "case_A_single_correct",
                "case_B_conflict_resolves",
                "case_C_consistent_confirms_mode",
                "case_D_independent_reask_is_product",
                "case_E_reliability_gate",
            )
        )
    }

    out = HERE / "smoke_metrics.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
