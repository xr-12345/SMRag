"""Phase 6 -- analysis of the VerifyOld drop-in N=5 screen.

Reads ``case_outcomes.csv`` / ``verification_turns.csv`` /
``verification_candidates.csv`` and writes the summary + comparison tables
used by ``write_reports.py``:

  * summary_by_strategy_noise.csv
  * paired_deltas.csv
  * rerank_analysis.csv
  * filter_analysis.csv
  * verification_quality.csv
  * rag_ablation.csv
  * baseline_regression_check.txt
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

from powerful_medrag.worthiness_dropin import DropinStrategy

OUT = Path(__file__).resolve().parent

BASE = DropinStrategy.HEURISTIC_BASELINE.value
RERANK = DropinStrategy.LEARNED_FULL_RERANK.value
FILTER = DropinStrategy.LEARNED_FULL_FILTER.value
NORAG = DropinStrategy.LEARNED_NORAG_FILTER.value
LEARNED = (RERANK, FILTER, NORAG)

PHASE5_OUTCOMES = (
    "artifacts/verimedrag-worthiness-online-screen-n5/case_outcomes.csv"
)

METRIC_LABELS = {
    "brier_score": "Brier",
    "nll_score": "NLL",
    "correct_top1": "Top-1 acc",
    "correct_top3": "Top-3 acc",
    "total_atomic_questions": "total atomic questions",
    "new_questions": "AskNew count",
    "verification_questions": "VerifyOld count",
    "correct_to_wrong_flips": "correct->wrong flips",
    "wrong_to_correct_flips": "wrong->correct flips",
    "unnecessary_verifications": "unnecessary verifications",
    "resolved_wrong_reports": "resolved wrong reports",
    "premature_stop": "premature stop",
}

PAIRED_METRICS = [
    "brier_score", "nll_score", "correct_top1", "correct_top3",
    "total_atomic_questions", "new_questions", "verification_questions",
    "correct_to_wrong_flips", "wrong_to_correct_flips",
    "unnecessary_verifications", "resolved_wrong_reports", "premature_stop",
]


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    try:
        return float(row[key])
    except (ValueError, TypeError):
        return float("nan")


def key_of(row):
    return (row["case_id"], row["noise_rate"], row["seed"])


def mean_of(rows, key):
    vals = [fnum(r, key) for r in rows if not math.isnan(fnum(r, key))]
    return sum(vals) / len(vals) if vals else float("nan")


def ratio(rows, num_key, den_key):
    n = sum(fnum(r, num_key) for r in rows)
    d = sum(fnum(r, den_key) for r in rows)
    return n / d if d else float("nan")


def ece_of(rows):
    conf = np.array([fnum(r, "confidence") for r in rows])
    acc = np.array([fnum(r, "correct_top1") for r in rows])
    if conf.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    n = conf.size
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == 9 else (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def paired_stats(diffs):
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    if d.size < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), d.size
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size))
    t = mean / se if se > 0 else 0.0
    p = float(2 * stats.t.sf(abs(t), df=d.size - 1))
    ci_lo = mean - 1.96 * se
    ci_hi = mean + 1.96 * se
    return mean, ci_lo, ci_hi, p, d.size


def write_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# --------------------------------------------------------------------------- #
# summary_by_strategy_noise.csv
# --------------------------------------------------------------------------- #

def summary_rows(outcomes):
    rows = []
    for strategy in (BASE, RERANK, FILTER, NORAG):
        for noise in ("0.2", "0.3"):
            sub = [r for r in outcomes
                   if r["strategy"] == strategy and r["noise_rate"] == noise]
            if not sub:
                continue
            rows.append({
                "strategy": strategy,
                "noise_rate": noise,
                "n_trajectories": len(sub),
                "correct_top1": round(mean_of(sub, "correct_top1"), 6),
                "correct_top3": round(mean_of(sub, "correct_top3"), 6),
                "brier_score": round(mean_of(sub, "brier_score"), 6),
                "nll_score": round(mean_of(sub, "nll_score"), 6),
                "ece": round(ece_of(sub), 6),
                "new_questions": round(mean_of(sub, "new_questions"), 6),
                "verification_questions": round(
                    mean_of(sub, "verification_questions"), 6),
                "total_atomic_questions": round(
                    mean_of(sub, "total_atomic_questions"), 6),
                "unnecessary_verifications": round(
                    mean_of(sub, "unnecessary_verifications"), 6),
                "resolved_wrong_reports": round(
                    mean_of(sub, "resolved_wrong_reports"), 6),
                "correct_to_wrong_flips": round(
                    mean_of(sub, "correct_to_wrong_flips"), 6),
                "wrong_to_correct_flips": round(
                    mean_of(sub, "wrong_to_correct_flips"), 6),
                "premature_stop": round(mean_of(sub, "premature_stop"), 6),
            })
    return rows


# --------------------------------------------------------------------------- #
# paired_deltas.csv
# --------------------------------------------------------------------------- #

def paired_deltas_rows(outcomes):
    base_by_key = {key_of(r): r for r in outcomes if r["strategy"] == BASE}
    rows = []
    for strategy in LEARNED:
        for metric in PAIRED_METRICS:
            diffs = []
            for r in outcomes:
                if r["strategy"] != strategy:
                    continue
                b = base_by_key.get(key_of(r))
                if b is None:
                    continue
                diffs.append(fnum(r, metric) - fnum(b, metric))
            mean, lo, hi, p, n = paired_stats(diffs)
            rows.append({
                "strategy": strategy,
                "metric": metric,
                "mean_delta": "" if math.isnan(mean) else round(mean, 6),
                "ci_lo": "" if math.isnan(lo) else round(lo, 6),
                "ci_hi": "" if math.isnan(hi) else round(hi, 6),
                "p_value": "" if math.isnan(p) else f"{p:.3g}",
                "n": n,
            })
    return rows


# --------------------------------------------------------------------------- #
# comparison table (shared by rerank / filter / rag_ablation)
# --------------------------------------------------------------------------- #

def comparison_rows(outcomes, a, b):
    """Per-metric paired comparison of strategy ``b`` vs ``a`` (delta = b - a)."""
    a_by_key = {key_of(r): r for r in outcomes if r["strategy"] == a}
    rows = []
    for metric in PAIRED_METRICS:
        diffs = []
        for r in outcomes:
            if r["strategy"] != b:
                continue
            ar = a_by_key.get(key_of(r))
            if ar is None:
                continue
            diffs.append(fnum(r, metric) - fnum(ar, metric))
        mean, lo, hi, p, n = paired_stats(diffs)
        rows.append({
            "metric": metric,
            "label": METRIC_LABELS[metric],
            f"{a}": round(mean_of(
                [r for r in outcomes if r["strategy"] == a], metric), 6),
            f"{b}": round(mean_of(
                [r for r in outcomes if r["strategy"] == b], metric), 6),
            "delta": "" if math.isnan(mean) else round(mean, 6),
            "ci_lo": "" if math.isnan(lo) else round(lo, 6),
            "ci_hi": "" if math.isnan(hi) else round(hi, 6),
            "p_value": "" if math.isnan(p) else f"{p:.3g}",
            "n": n,
        })
    return rows


# --------------------------------------------------------------------------- #
# verification_quality.csv
# --------------------------------------------------------------------------- #

def verification_quality_rows(outcomes, turns, candidates):
    rows = []
    for strategy in (BASE, RERANK, FILTER, NORAG):
        o = [r for r in outcomes if r["strategy"] == strategy]
        t = [r for r in turns if r["strategy"] == strategy]
        c = [r for r in candidates if r["strategy"] == strategy]

        verify_q = sum(fnum(r, "verification_questions") for r in o)
        opportunities = len(t)
        executed = len([r for r in t if r["final_action_type"] == "verify"])
        replaced = len([r for r in t if r["final_action_type"] == "new"])

        # candidate-level gate rates (learned only)
        passed_harm = [r for r in c if fnum(r, "passed_harm_gate") == 1.0]
        blocked_harm = [r for r in c if fnum(r, "passed_harm_gate") == 0.0]
        passed_gain = [r for r in c if fnum(r, "passed_gain_gate") == 1.0]
        selected = [r for r in c if fnum(r, "selected") == 1.0]

        rows.append({
            "strategy": strategy,
            "verify_opportunities": opportunities,
            "executed_verifies": executed,
            "replaced_with_asknew": replaced,
            "unnecessary_rate": round(
                ratio(o, "unnecessary_verifications", "verification_questions"), 6),
            "resolved_rate": round(
                ratio(o, "resolved_wrong_reports", "verification_questions"), 6),
            "correct_to_wrong_rate": round(
                ratio(o, "correct_to_wrong_flips", "verification_questions"), 6),
            "wrong_to_correct_rate": round(
                ratio(o, "wrong_to_correct_flips", "verification_questions"), 6),
            "heuristic_report_wrong_rate": round(
                mean_of(t, "heuristic_report_wrong"), 6) if t else "",
            "learned_report_wrong_rate": round(
                mean_of(
                    [r for r in t if r["final_action_type"] == "verify"],
                    "learned_report_wrong"), 6) if t else "",
            "harm_gate_pass_rate": round(
                mean_of(c, "passed_harm_gate"), 6) if c else "",
            "harm_blocked_wrong_rate": round(
                mean_of(blocked_harm, "report_wrong"), 6) if blocked_harm else "",
            "harm_passed_wrong_rate": round(
                mean_of(passed_harm, "report_wrong"), 6) if passed_harm else "",
            "gain_gate_pass_rate": round(
                mean_of(c, "passed_gain_gate"), 6) if c else "",
            "selected_wrong_rate": round(
                mean_of(selected, "report_wrong"), 6) if selected else "",
        })
    return rows


# --------------------------------------------------------------------------- #
# baseline_regression_check.txt
# --------------------------------------------------------------------------- #

def baseline_regression_check(outcomes):
    p5 = {
        key_of(r): r
        for r in load(Path(PHASE5_OUTCOMES))
        if r["strategy"] == "heuristic_verify"
    }
    base = [r for r in outcomes if r["strategy"] == BASE]
    compare = ("predicted_diagnosis", "correct_top1", "correct_top3",
               "brier_score", "new_questions", "verification_questions",
               "total_atomic_questions", "stop_reason")
    mismatches = []
    missing = 0
    for r in base:
        o = p5.get(key_of(r))
        if o is None:
            missing += 1
            continue
        for k in compare:
            if str(r[k]) != str(o[k]):
                mismatches.append((r["case_id"], r["noise_rate"], r["seed"],
                                   k, r[k], o[k]))
    lines = [
        "Phase 6 -- heuristic_baseline vs Phase 5 heuristic_verify regression",
        f"compared {len(base)} baseline trajectories against Phase 5",
        f"mismatches: {len(mismatches)}; missing in Phase5: {missing}",
    ]
    for m in mismatches[:20]:
        lines.append(f"  mismatch {m}")
    if len(mismatches) > 20:
        lines.append(f"  ... and {len(mismatches)-20} more")
    verdict = "PASS (byte-identical)" if (not mismatches and not missing) else "FAIL"
    lines.append(f"verdict: {verdict}")
    (OUT / "baseline_regression_check.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return verdict, len(mismatches), missing


def main():
    outcomes = load(OUT / "case_outcomes.csv")
    turns = load(OUT / "verification_turns.csv")
    candidates = load(OUT / "verification_candidates.csv")

    write_csv(summary_rows(outcomes), OUT / "summary_by_strategy_noise.csv",
              ["strategy", "noise_rate", "n_trajectories", "correct_top1",
               "correct_top3", "brier_score", "nll_score", "ece",
               "new_questions", "verification_questions",
               "total_atomic_questions", "unnecessary_verifications",
               "resolved_wrong_reports", "correct_to_wrong_flips",
               "wrong_to_correct_flips", "premature_stop"])

    write_csv(paired_deltas_rows(outcomes), OUT / "paired_deltas.csv",
              ["strategy", "metric", "mean_delta", "ci_lo", "ci_hi",
               "p_value", "n"])

    rr = comparison_rows(outcomes, BASE, RERANK)
    write_csv(rr, OUT / "rerank_analysis.csv",
              ["metric", "label", BASE, RERANK, "delta", "ci_lo", "ci_hi",
               "p_value", "n"])

    ff = comparison_rows(outcomes, BASE, FILTER)
    write_csv(ff, OUT / "filter_analysis.csv",
              ["metric", "label", BASE, FILTER, "delta", "ci_lo", "ci_hi",
               "p_value", "n"])

    ra = comparison_rows(outcomes, NORAG, FILTER)
    write_csv(ra, OUT / "rag_ablation.csv",
              ["metric", "label", NORAG, FILTER, "delta", "ci_lo", "ci_hi",
               "p_value", "n"])

    vq = verification_quality_rows(outcomes, turns, candidates)
    write_csv(vq, OUT / "verification_quality.csv",
              ["strategy", "verify_opportunities", "executed_verifies",
               "replaced_with_asknew", "unnecessary_rate", "resolved_rate",
               "correct_to_wrong_rate", "wrong_to_correct_rate",
               "heuristic_report_wrong_rate", "learned_report_wrong_rate",
               "harm_gate_pass_rate", "harm_blocked_wrong_rate",
               "harm_passed_wrong_rate", "gain_gate_pass_rate",
               "selected_wrong_rate"])

    verdict, nm, miss = baseline_regression_check(outcomes)
    print(f"baseline regression: {verdict} ({nm} mismatches, {miss} missing)")
    print(f"summary rows: {len(summary_rows(outcomes))}")
    print(f"paired delta rows: {len(paired_deltas_rows(outcomes))}")
    print(f"rerank/filter/rag rows: {len(rr)}/{len(ff)}/{len(ra)}")
    print(f"verification quality rows: {len(vq)}")


if __name__ == "__main__":
    main()
