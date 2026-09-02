"""Phase 5 -- analysis for the N=5 online learned-worthiness screen.

Reads the trajectory CSVs written by ``run_screen.py`` and produces the summary /
paired-delta / verification / harm-gate / RAG-ablation tables plus the baseline
regression check against the frozen Phase 2B ``joint_gate_w1.0`` run (seed 2026).

All quantities are realized outcomes against the true disease (evaluation side);
the deployable prediction path never reads the true state (enforced by policy +
tests).  Red lines: validate split only, no test access, no retraining, no commit.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
PHASE2B = Path("artifacts/verimedrag-rag-joint-gate-screen-n5")

STRATEGIES = (
    "heuristic_verify",
    "model_based_vbayes_verify",
    "learned_worthiness_full_rag",
    "learned_worthiness_no_rag",
)
SHORT = {
    "heuristic_verify": "heuristic",
    "model_based_vbayes_verify": "v_bayes",
    "learned_worthiness_full_rag": "learned_full",
    "learned_worthiness_no_rag": "learned_norag",
}
NOISE_RATES = (0.2, 0.3)

# metrics where a *delta* is meaningful for the paired comparison
DELTA_METRICS = (
    "correct_top1", "correct_top3", "brier_score", "new_questions",
    "verification_questions", "total_atomic_questions",
    "correct_to_wrong_flips", "wrong_to_correct_flips",
    "unnecessary_verifications", "resolved_wrong_reports",
)

# baseline -> candidates for paired deltas
PAIRS = (
    ("heuristic_verify", "model_based_vbayes_verify"),
    ("heuristic_verify", "learned_worthiness_full_rag"),
    ("heuristic_verify", "learned_worthiness_no_rag"),
    ("model_based_vbayes_verify", "learned_worthiness_full_rag"),
    ("model_based_vbayes_verify", "learned_worthiness_no_rag"),
)


def read(name):
    with open(OUT / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    """Safe float; empty string -> None."""
    v = row.get(key, "")
    if v == "" or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def summary_by_strategy_noise(outcomes):
    rows = []
    for strategy in STRATEGIES:
        for noise in NOISE_RATES:
            sub = [r for r in outcomes
                   if r["strategy"] == strategy and float(r["noise_rate"]) == noise]
            n = len(sub)
            verifies = sum(int(r["verification_questions"]) for r in sub)
            resolved = sum(int(r["resolved_wrong_reports"]) for r in sub)
            rows.append({
                "strategy": strategy,
                "strategy_short": SHORT[strategy],
                "noise_rate": noise,
                "n_trajectories": n,
                "top1_accuracy": round(mean([fnum(r, "correct_top1") for r in sub]), 6),
                "top3_accuracy": round(mean([fnum(r, "correct_top3") for r in sub]), 6),
                "brier_score": round(mean([fnum(r, "brier_score") for r in sub]), 6),
                "new_questions_mean": round(mean([fnum(r, "new_questions") for r in sub]), 4),
                "verification_questions_mean": round(mean([fnum(r, "verification_questions") for r in sub]), 4),
                "total_atomic_questions_mean": round(mean([fnum(r, "total_atomic_questions") for r in sub]), 4),
                "unnecessary_verifications_mean": round(mean([fnum(r, "unnecessary_verifications") for r in sub]), 4),
                "resolved_wrong_reports_mean": round(mean([fnum(r, "resolved_wrong_reports") for r in sub]), 4),
                "correct_to_wrong_flips_mean": round(mean([fnum(r, "correct_to_wrong_flips") for r in sub]), 4),
                "wrong_to_correct_flips_mean": round(mean([fnum(r, "wrong_to_correct_flips") for r in sub]), 4),
                "premature_stop_rate": round(mean([fnum(r, "premature_stop") for r in sub]), 6),
                "uncertain_output_rate": round(mean([fnum(r, "uncertain_output") for r in sub]), 6),
                "verification_hit_rate": round(resolved / verifies, 6) if verifies else "",
                "wall_clock_mean": round(mean([fnum(r, "wall_clock") for r in sub]), 4),
            })
    return rows


def paired_deltas(outcomes):
    index = {}
    for r in outcomes:
        index[(r["case_id"], float(r["noise_rate"]), int(r["seed"]), r["strategy"])] = r
    rows = []
    for r in outcomes:
        key = (r["case_id"], float(r["noise_rate"]), int(r["seed"]))
        for base, cand in PAIRS:
            b = index.get((*key, base))
            c = index.get((*key, cand))
            if b is None or c is None:
                continue
            for metric in DELTA_METRICS:
                bv = fnum(b, metric)
                cv = fnum(c, metric)
                if bv is None or cv is None:
                    continue
                rows.append({
                    "case_id": r["case_id"],
                    "diagnosis": r["diagnosis"],
                    "noise_rate": r["noise_rate"],
                    "seed": r["seed"],
                    "baseline": SHORT[base],
                    "candidate": SHORT[cand],
                    "metric": metric,
                    "baseline_value": bv,
                    "candidate_value": cv,
                    "delta": cv - bv,
                })
    return rows


def verification_analysis(outcomes):
    rows = []
    for strategy in STRATEGIES:
        sub = [r for r in outcomes if r["strategy"] == strategy]
        n = len(sub)
        verifies = sum(int(r["verification_questions"]) for r in sub)
        unnecessary = sum(int(r["unnecessary_verifications"]) for r in sub)
        resolved = sum(int(r["resolved_wrong_reports"]) for r in sub)
        ctw = sum(int(r["correct_to_wrong_flips"]) for r in sub)
        wtc = sum(int(r["wrong_to_correct_flips"]) for r in sub)
        rows.append({
            "strategy": strategy,
            "strategy_short": SHORT[strategy],
            "n_trajectories": n,
            "total_verifications": verifies,
            "mean_verifications": round(verifies / n, 6) if n else "",
            "mean_unnecessary_verifications": round(unnecessary / n, 6) if n else "",
            "mean_resolved_wrong_reports": round(resolved / n, 6) if n else "",
            "verification_hit_rate": round(resolved / verifies, 6) if verifies else "",
            "correct_to_wrong_per_verification": round(ctw / verifies, 6) if verifies else "",
            "wrong_to_correct_per_verification": round(wtc / verifies, 6) if verifies else "",
            "total_correct_to_wrong_flips": ctw,
            "total_wrong_to_correct_flips": wtc,
        })
    return rows


def harm_gate_analysis(candidates):
    """Candidate-level view of the learned gain + harm gates.

    ``gain_pass`` = net_value > 0; ``harm_pass`` = harm_hat <= tau_harm;
    ``allowed`` = both.  ``harm_blocked`` = gain-passing candidates the harm
    gate suppresses (predicted high correct->wrong risk, hence not selected).
    """
    rows = []
    for strategy in ("learned_worthiness_full_rag", "learned_worthiness_no_rag"):
        sub = [c for c in candidates if c["strategy"] == strategy]
        gain_pass = [c for c in sub if c.get("passed_gain_gate") == "True"]
        harm_pass = [c for c in sub if c.get("passed_harm_gate") == "True"]
        allowed = [c for c in sub if c.get("passed_gain_gate") == "True"
                   and c.get("passed_harm_gate") == "True"]
        harm_blocked = [c for c in sub if c.get("passed_gain_gate") == "True"
                        and c.get("passed_harm_gate") == "False"]
        selected = [c for c in sub if c.get("selected") == "1"]

        def mh(rows_, key="harm_hat"):
            vals = [fnum(c, key) for c in rows_]
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 6) if vals else ""

        # realized correct->wrong among actually-selected verifications
        sel_ctw = [int(c["realized_correct_to_wrong"]) for c in selected
                   if c.get("realized_correct_to_wrong") not in ("", None)]
        sel_wtc = [int(c["realized_wrong_to_correct"]) for c in selected
                   if c.get("realized_wrong_to_correct") not in ("", None)]

        rows.append({
            "strategy": strategy,
            "strategy_short": SHORT[strategy],
            "n_candidates": len(sub),
            "n_gain_pass": len(gain_pass),
            "n_harm_pass": len(harm_pass),
            "n_allowed": len(allowed),
            "n_harm_blocked": len(harm_blocked),
            "n_selected": len(selected),
            "mean_harm_hat_all": mh(sub),
            "mean_harm_hat_selected": mh(selected),
            "mean_harm_hat_harm_blocked": mh(harm_blocked),
            "mean_net_value_all": mh(sub, "net_value"),
            "mean_net_value_selected": mh(selected, "net_value"),
            "realized_correct_to_wrong_selected": sum(sel_ctw),
            "realized_correct_to_wrong_rate_selected": (
                round(sum(sel_ctw) / len(sel_ctw), 6) if sel_ctw else ""
            ),
            "realized_wrong_to_correct_selected": sum(sel_wtc),
        })
    return rows


def rag_ablation(outcomes):
    """full-RAG vs no-RAG (both learned), paired at the case level."""
    full = {(r["case_id"], float(r["noise_rate"]), int(r["seed"])): r
            for r in outcomes if r["strategy"] == "learned_worthiness_full_rag"}
    norag = {(r["case_id"], float(r["noise_rate"]), int(r["seed"])): r
             for r in outcomes if r["strategy"] == "learned_worthiness_no_rag"}
    rows = []
    for metric in DELTA_METRICS:
        deltas = []
        fv = []
        nv = []
        for key in full:
            f = full[key]
            n = norag.get(key)
            if n is None:
                continue
            a = fnum(f, metric)
            b = fnum(n, metric)
            if a is None or b is None:
                continue
            fv.append(a)
            nv.append(b)
            deltas.append(a - b)
        rows.append({
            "metric": metric,
            "learned_full_rag_mean": round(mean(fv), 6) if fv else "",
            "learned_no_rag_mean": round(mean(nv), 6) if nv else "",
            "mean_delta": round(mean(deltas), 6) if deltas else "",
            "n_pairs": len(deltas),
        })
    return rows


def baseline_regression_check(outcomes):
    """Compare our heuristic_verify (seed 2026) to Phase 2B joint_gate_w1.0."""
    phase2b = list(csv.DictReader(
        open(PHASE2B / "case_outcomes.csv", newline="", encoding="utf-8")
    ))
    ref = {(r["case_id"], float(r["noise_rate"])): r
           for r in phase2b if r["config"] == "joint_gate_w1.0" and int(r["seed"]) == 2026}
    mine = {(r["case_id"], float(r["noise_rate"])): r
            for r in outcomes if r["strategy"] == "heuristic_verify" and int(r["seed"]) == 2026}

    compare_cols = [
        "predicted_diagnosis", "correct_top1", "correct_top3", "new_questions",
        "verification_questions", "total_atomic_questions",
        "unnecessary_verifications", "resolved_wrong_reports",
        "premature_stop", "uncertain_output",
    ]
    brier_col = "brier_score"

    lines = []
    add = lines.append
    add("Baseline regression check: heuristic_verify (seed 2026) vs "
        "Phase 2B joint_gate_w1.0 (seed 2026)")
    add("=" * 78)
    add(f"rows compared: {len(mine)} (expect 245 cases x 2 noise = 490)")
    add(f"Phase 2B reference rows: {len(ref)}")
    add("")

    keys = sorted(set(mine) & set(ref))
    n = len(keys)
    mismatches = []
    brier_diffs = []
    for key in keys:
        m, r = mine[key], ref[key]
        for col in compare_cols:
            if str(m[col]) != str(r[col]):
                mismatches.append((key, col, m[col], r[col]))
        bm = fnum(m, brier_col)
        br = fnum(r, brier_col)
        if bm is not None and br is not None:
            brier_diffs.append(abs(bm - br))

    add(f"matched keys: {n}")
    add(f"integer/bool column mismatches: {len(mismatches)}")
    for key, col, mv, rv in mismatches[:20]:
        add(f"  MISMATCH {key} {col}: mine={mv} ref={rv}")
    if brier_diffs:
        add(f"max |brier_score| diff: {max(brier_diffs):.2e}")
    add("")
    if n == 490 and not mismatches and (not brier_diffs or max(brier_diffs) < 1e-6):
        add("RESULT: PASS -- heuristic_verify is byte-identical to the frozen "
            "Phase 2B joint_gate_w1.0 baseline; the learned integration did not "
            "alter the heuristic path.")
    else:
        add("RESULT: FAIL -- see mismatches above.")
    return "\n".join(lines) + "\n"


def write_csv(rows, path, fieldnames):
    with open(OUT / path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    outcomes = read("case_outcomes.csv")
    candidates = read("verification_candidates.csv")
    print(f"read {len(outcomes)} outcome rows, {len(candidates)} candidate rows")

    summ = summary_by_strategy_noise(outcomes)
    write_csv(summ, "summary_by_strategy_noise.csv",
              ["strategy", "strategy_short", "noise_rate", "n_trajectories",
               "top1_accuracy", "top3_accuracy", "brier_score",
               "new_questions_mean", "verification_questions_mean",
               "total_atomic_questions_mean", "unnecessary_verifications_mean",
               "resolved_wrong_reports_mean", "correct_to_wrong_flips_mean",
               "wrong_to_correct_flips_mean", "premature_stop_rate",
               "uncertain_output_rate", "verification_hit_rate", "wall_clock_mean"])

    delta = paired_deltas(outcomes)
    write_csv(delta, "paired_deltas.csv",
              ["case_id", "diagnosis", "noise_rate", "seed", "baseline",
               "candidate", "metric", "baseline_value", "candidate_value", "delta"])

    verif = verification_analysis(outcomes)
    write_csv(verif, "verification_analysis.csv",
              ["strategy", "strategy_short", "n_trajectories",
               "total_verifications", "mean_verifications",
               "mean_unnecessary_verifications", "mean_resolved_wrong_reports",
               "verification_hit_rate", "correct_to_wrong_per_verification",
               "wrong_to_correct_per_verification", "total_correct_to_wrong_flips",
               "total_wrong_to_correct_flips"])

    harm = harm_gate_analysis(candidates)
    write_csv(harm, "harm_gate_analysis.csv",
              ["strategy", "strategy_short", "n_candidates", "n_gain_pass",
               "n_harm_pass", "n_allowed", "n_harm_blocked", "n_selected",
               "mean_harm_hat_all", "mean_harm_hat_selected",
               "mean_harm_hat_harm_blocked", "mean_net_value_all",
               "mean_net_value_selected", "realized_correct_to_wrong_selected",
               "realized_correct_to_wrong_rate_selected",
               "realized_wrong_to_correct_selected"])

    rag = rag_ablation(outcomes)
    write_csv(rag, "rag_ablation.csv",
              ["metric", "learned_full_rag_mean", "learned_no_rag_mean",
               "mean_delta", "n_pairs"])

    (OUT / "baseline_regression_check.txt").write_text(
        baseline_regression_check(outcomes), encoding="utf-8"
    )

    print("wrote summary_by_strategy_noise.csv, paired_deltas.csv, "
          "verification_analysis.csv, harm_gate_analysis.csv, rag_ablation.csv, "
          "baseline_regression_check.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
