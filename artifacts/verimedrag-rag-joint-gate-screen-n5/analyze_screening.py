"""Phase 2B screening analysis: summaries, paired deltas, Go/No-Go criteria.

Reads ``case_outcomes.csv`` and ``turn_observations.csv`` produced by
``run_screening.py`` and writes:
  * ``summary_by_config_noise.csv``  — per config x noise aggregate metrics
  * ``paired_deltas.csv``            — per case x noise, delta vs baseline
  * ``screening_metrics.json``       — everything the report needs, incl. the
                                      7 Go/No-Go criteria numbers

Baseline is ``rank_only_w0.0``.  Retrieval-triggered verification (the joint
gate actually opening VerifyOld) can only occur in ``joint_gate_*`` configs,
because ``RANK_ONLY`` never lets retrieval open the gate.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASELINE = "rank_only_w0.0"
NOISE_RATES = ("0.2", "0.3")
CONFIGS = [
    "rank_only_w0.0",
    "rank_only_w0.1",
    "rank_only_w0.5",
    "rank_only_w1.0",
    "joint_gate_w0.1",
    "joint_gate_w0.5",
    "joint_gate_w1.0",
]


def load(path: str) -> list[dict]:
    with open(OUT / path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def i(x) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def summarize(cases: list[dict]) -> dict:
    rows = {}
    for config in CONFIGS:
        for noise in NOISE_RATES:
            sub = [c for c in cases if c["config"] == config and c["noise_rate"] == noise]
            if not sub:
                continue
            n = len(sub)
            verifies = sum(i(c["verification_questions"]) for c in sub)
            unnec = sum(i(c["unnecessary_verifications"]) for c in sub)
            resolved = sum(i(c["resolved_wrong_reports"]) for c in sub)
            retrig = sum(i(c["retrieval_triggered_verifications"]) for c in sub)
            retrig_hit = sum(i(c["retrieval_triggered_hits"]) for c in sub)
            gate_open_asknew = sum(i(c["gate_open_but_asknew"]) for c in sub)
            gate_open_verify = sum(i(c["gate_open_and_verify"]) for c in sub)
            ord_r = sum(i(c["ordinary_retrieves"]) for c in sub)
            cf_r = sum(i(c["counterfactual_retrieves"]) for c in sub)
            rows[(config, noise)] = {
                "config": config,
                "noise_rate": noise,
                "cases": n,
                "top1_accuracy": sum(i(c["correct_top1"]) for c in sub) / n,
                "top3_accuracy": sum(i(c["correct_top3"]) for c in sub) / n,
                "brier_score": sum(fnum(c["brier_score"]) for c in sub) / n,
                "new_questions_mean": sum(i(c["new_questions"]) for c in sub) / n,
                "verification_questions_mean": verifies / n,
                "total_atomic_questions_mean": (
                    sum(i(c["total_atomic_questions"]) for c in sub) / n
                ),
                "total_atomic_questions_sum": sum(i(c["total_atomic_questions"]) for c in sub),
                "unnecessary_verification_rate": (unnec / verifies) if verifies else None,
                "conflict_resolution_rate": (resolved / verifies) if verifies else None,
                "premature_stop_rate": sum(i(c["premature_stop"]) for c in sub) / n,
                "uncertain_output_rate": sum(i(c["uncertain_output"]) for c in sub) / n,
                "retrieval_triggered_verifications": retrig,
                "retrieval_triggered_hits": retrig_hit,
                "retrieval_triggered_cases": sum(
                    1 for c in sub if i(c["retrieval_triggered_verifications"]) > 0
                ),
                "triggered_verification_true_hit_rate": (
                    (retrig_hit / retrig) if retrig else None
                ),
                "gate_open_but_asknew": gate_open_asknew,
                "gate_open_and_verify": gate_open_verify,
                "ordinary_retrieves_mean": ord_r / n,
                "counterfactual_retrieves_mean": cf_r / n,
            }
    return rows


def paired_deltas(cases: list[dict], turns: list[dict]) -> tuple[list[dict], dict]:
    """Per (case, noise, config) deltas vs baseline, plus turn-level change counts."""
    # index baseline case rows
    base_rows = {
        (c["case_id"], c["noise_rate"]): c
        for c in cases
        if c["config"] == BASELINE
    }
    # turn maps: (case_id, noise, config) -> {turn_index: row}
    turn_map = defaultdict(dict)
    for t in turns:
        turn_map[(t["case_id"], t["noise_rate"], t["config"])][int(t["turn_index"])] = t

    deltas = []
    change_stats = {}  # config -> {action_change_cases, order_change_cases, action_change_turns, order_change_turns}
    for config in CONFIGS:
        if config == BASELINE:
            continue
        for noise in NOISE_RATES:
            sub = [c for c in cases if c["config"] == config and c["noise_rate"] == noise]
            for c in sub:
                key = (c["case_id"], noise)
                b = base_rows.get(key)
                if b is None:
                    continue
                deltas.append(
                    {
                        "case_id": c["case_id"],
                        "diagnosis": c["diagnosis"],
                        "noise_rate": noise,
                        "config": config,
                        "delta_top1": i(c["correct_top1"]) - i(b["correct_top1"]),
                        "delta_top3": i(c["correct_top3"]) - i(b["correct_top3"]),
                        "delta_brier": round(
                            fnum(c["brier_score"]) - fnum(b["brier_score"]), 8
                        ),
                        "delta_total_questions": i(c["total_atomic_questions"])
                        - i(b["total_atomic_questions"]),
                        "delta_new_questions": i(c["new_questions"])
                        - i(b["new_questions"]),
                        "delta_verification_questions": i(c["verification_questions"])
                        - i(b["verification_questions"]),
                    }
                )

        # turn-level change counting (case-level aggregation)
        action_cases = set()
        order_cases = set()
        action_turns = 0
        order_turns = 0
        for (case_id, noise, cfg), tmap in turn_map.items():
            if cfg != config:
                continue
            bmap = turn_map.get((case_id, noise, BASELINE), {})
            if not bmap:
                continue
            for ti in sorted(set(tmap) & set(bmap)):
                a, bb = tmap[ti], bmap[ti]
                if (
                    a["chosen_action"] != bb["chosen_action"]
                    or (a["report_index"] or "") != (bb["report_index"] or "")
                ):
                    action_turns += 1
                    action_cases.add((case_id, noise))
                if (a["verify_order"] or "") != (bb["verify_order"] or ""):
                    order_turns += 1
                    order_cases.add((case_id, noise))
        change_stats[config] = {
            "action_change_cases": len(action_cases),
            "order_change_cases": len(order_cases),
            "action_change_turns": action_turns,
            "order_change_turns": order_turns,
        }
    return deltas, change_stats


def go_no_go(summary: dict, change_stats: dict) -> dict:
    """Evaluate the 7 screening criteria for each joint_gate config vs baseline."""
    criteria = {}
    for config in ("joint_gate_w0.1", "joint_gate_w0.5", "joint_gate_w1.0"):
        ev = {}
        for noise in NOISE_RATES:
            base = summary[(BASELINE, noise)]
            cfg = summary[(config, noise)]
            # (1) truly triggers new VerifyOld
            trig = cfg["retrieval_triggered_verifications"]
            trig_cases = cfg["retrieval_triggered_cases"]
            # (2) not clearly dominated (same or better Top-1/Brier at same-ish questions)
            # (3) Top-1 drop <= 0.5pp
            d_top1 = cfg["top1_accuracy"] - base["top1_accuracy"]
            # (4) Brier relative degradation <= 5%
            rel_brier = (
                (cfg["brier_score"] - base["brier_score"]) / base["brier_score"]
                if base["brier_score"] > 0
                else 0.0
            )
            # (5) unnecessary verification rate increase <= 5pp
            d_unnec = (
                (cfg["unnecessary_verification_rate"] or 0.0)
                - (base["unnecessary_verification_rate"] or 0.0)
            )
            # (6) conflict resolution rate decrease <= 5pp
            d_conflict = (
                (cfg["conflict_resolution_rate"] or 0.0)
                - (base["conflict_resolution_rate"] or 0.0)
            )
            # (7) total question increase buys risk reduction
            d_questions = cfg["total_atomic_questions_mean"] - base["total_atomic_questions_mean"]
            ev[noise] = {
                "triggers_verify": trig,
                "triggers_verify_cases": trig_cases,
                "delta_top1_pp": round(d_top1 * 100, 4),
                "rel_brier_change": round(rel_brier, 6),
                "delta_unnecessary_rate_pp": round(d_unnec * 100, 4),
                "delta_conflict_rate_pp": round(d_conflict * 100, 4),
                "delta_questions": round(d_questions, 4),
                "triggered_true_hit_rate": cfg["triggered_verification_true_hit_rate"],
            }
        criteria[config] = ev
    return criteria


def write_summary(rows: dict) -> None:
    keys = list(next(iter(rows.values())).keys())
    with open(OUT / "summary_by_config_noise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for (config, noise) in sorted(rows):
            w.writerow(rows[(config, noise)])


def main() -> None:
    cases = load("case_outcomes.csv")
    turns = load("turn_observations.csv")
    summary = summarize(cases)
    deltas, change_stats = paired_deltas(cases, turns)
    criteria = go_no_go(summary, change_stats)

    write_summary(summary)
    with open(OUT / "paired_deltas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(deltas[0].keys()))
        w.writeheader()
        w.writerows(deltas)

    out = {
        "summary": {f"{k[0]}__{k[1]}": v for k, v in summary.items()},
        "change_stats": change_stats,
        "go_no_go": criteria,
        "baseline": BASELINE,
    }
    with open(OUT / "screening_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote summary_by_config_noise.csv ({len(summary)} rows)")
    print(f"wrote paired_deltas.csv ({len(deltas)} rows)")
    print(f"wrote screening_metrics.json")
    print(json.dumps(criteria, indent=2))


if __name__ == "__main__":
    main()
