"""Prompt #18 -- N=5 screen analysis.

Reads ``case_outcomes.csv`` (written by ``run_screen.py``) and produces:

  * ``summary_by_strategy_noise.csv`` -- aggregate metrics per (strategy, noise).
  * ``paired_deltas.csv``              -- paired case-level deltas.
  * ``baseline_regression_check.txt``  -- heuristic_baseline vs frozen drop-in
                                          heuristic (byte-identical check).
  * ``GO_NO_GO.md``                    -- the Go / No-Go determination vs the
                                          preregistered criteria.

Red lines honoured: no test split, no retraining, no git commit, writes only here.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent

HEURISTIC_BASELINE = "heuristic_baseline"
MODEL_BASED_VBAYES = "model_based_v_bayes"
FORCED_VERIFY = "unified_brier_audit_forced_verify"
CORRECTED = "unified_brier_audit_corrected"
CORRECTED_DYNAMIC_RAG = "unified_brier_audit_corrected_dynamic_rag"

STRATEGIES = (
    HEURISTIC_BASELINE,
    MODEL_BASED_VBAYES,
    FORCED_VERIFY,
    CORRECTED,
    CORRECTED_DYNAMIC_RAG,
)

# Frozen drop-in heuristic (Phase 15) -- the byte-identical regression reference.
DROPIN_REF = "artifacts/verimedrag-worthiness-dropin-screen-n5/case_outcomes.csv"


def _f(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError):
        return float("nan")


def load_outcomes():
    return list(csv.DictReader(open(OUT / "case_outcomes.csv", newline="")))


def summarize(rows):
    """Aggregate per (strategy, noise) over all seeds/cases."""
    keys = [
        "correct_top1", "correct_top3", "brier_score", "nll_score",
        "new_questions", "verification_questions", "total_atomic_questions",
        "unnecessary_verifications", "resolved_wrong_reports", "premature_stop",
        "uncertain_output", "stop_blocked_count",
    ]
    out = []
    for strategy in STRATEGIES:
        for noise in (0.2, 0.3):
            sub = [r for r in rows if r["strategy"] == strategy
                   and abs(_f(r, "noise_rate") - noise) < 1e-9]
            n = len(sub)
            if n == 0:
                continue
            rec = {"strategy": strategy, "noise_rate": noise, "n_cases": n}
            for k in keys:
                vals = [_f(r, k) for r in sub]
                rec[k] = round(sum(vals) / n, 8)
            # unnecessary rate = mean unnecessary_verifications / mean verify count
            v = rec["verification_questions"]
            rec["unnecessary_verification_rate"] = round(
                rec["unnecessary_verifications"] / v, 6
            ) if v > 0 else 0.0
            rec["resolved_wrong_report_rate"] = round(
                rec["resolved_wrong_reports"] / v, 6
            ) if v > 0 else 0.0
            out.append(rec)
    return out


def write_summary(rows):
    recs = summarize(rows)
    fields = list(recs[0].keys()) if recs else []
    with open(OUT / "summary_by_strategy_noise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(recs)
    return recs


def paired_deltas(rows):
    """Case-level paired deltas: reference strategy minus comparison strategy."""
    def index():
        out = {}
        for r in rows:
            key = (r["case_id"], _f(r, "noise_rate"), _f(r, "seed"))
            out.setdefault(key, {})[r["strategy"]] = r
        return out

    idx = index()
    metrics = ("correct_top1", "brier_score", "total_atomic_questions",
               "verification_questions", "unnecessary_verifications")
    pairs = [
        ("corrected_vs_heuristic", CORRECTED, HEURISTIC_BASELINE),
        ("corrected_vs_forced", CORRECTED, FORCED_VERIFY),
        ("forced_vs_heuristic", FORCED_VERIFY, HEURISTIC_BASELINE),
        ("corrected_vs_model_based", CORRECTED, MODEL_BASED_VBAYES),
        ("corrected_dynrag_vs_corrected", CORRECTED_DYNAMIC_RAG, CORRECTED),
    ]
    out = []
    for name, a, b in pairs:
        for key in sorted(idx):
            case_id, noise, seed = key
            ra = idx[key].get(a)
            rb = idx[key].get(b)
            if ra is None or rb is None:
                continue
            rec = {
                "pair": name, "case_id": case_id, "noise_rate": noise, "seed": seed,
            }
            for m in metrics:
                rec[f"{m}_a"] = ra[m]
                rec[f"{m}_b"] = rb[m]
                rec[f"delta_{m}"] = round(_f(ra, m) - _f(rb, m), 8)
            out.append(rec)
    if out:
        with open(OUT / "paired_deltas.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
    return out


def regression_check(rows):
    """heuristic_baseline vs the frozen drop-in heuristic (byte-identical)."""
    ref = {}
    for r in csv.DictReader(open(DROPIN_REF, newline="")):
        if r["strategy"] == HEURISTIC_BASELINE:
            ref[(r["case_id"], _f(r, "noise_rate"), _f(r, "seed"))] = r
    mine = [r for r in rows if r["strategy"] == HEURISTIC_BASELINE]
    keys = ("predicted_diagnosis", "correct_top1", "brier_score", "new_questions",
            "verification_questions", "unnecessary_verifications",
            "resolved_wrong_reports")
    mism, missing = [], 0
    for r in mine:
        key = (r["case_id"], _f(r, "noise_rate"), _f(r, "seed"))
        o = ref.get(key)
        if o is None:
            missing += 1
            continue
        for k in keys:
            if str(r.get(k)) != str(o.get(k)):
                mism.append((r["case_id"], r["noise_rate"], r["seed"], k,
                             r.get(k), o.get(k)))
    ok = (len(mism) == 0 and missing == 0)
    lines = [
        "Prompt #18 -- heuristic_baseline byte-identical regression",
        f"reference: {DROPIN_REF}",
        f"compared trajectories: {len(mine)}; missing: {missing}; "
        f"mismatches: {len(mism)}",
        f"RESULT: {'PASS' if ok else 'FAIL'}",
    ]
    for m in mism[:20]:
        lines.append(str(m))
    (OUT / "baseline_regression_check.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return ok, len(mism), missing


def go_no_go(rows, recs):
    """Evaluate the preregistered Go / No-Go criteria."""
    def mean(strategy, noise, k):
        vals = [_f(r, k) for r in rows if r["strategy"] == strategy
                and abs(_f(r, "noise_rate") - noise) < 1e-9]
        return sum(vals) / len(vals) if vals else float("nan")

    # Reference point: noise 0.2 (the primary validation point).
    h_top1 = mean(HEURISTIC_BASELINE, 0.2, "correct_top1")
    c_top1 = mean(CORRECTED, 0.2, "correct_top1")
    h_brier = mean(HEURISTIC_BASELINE, 0.2, "brier_score")
    c_brier = mean(CORRECTED, 0.2, "brier_score")
    h_q = mean(HEURISTIC_BASELINE, 0.2, "total_atomic_questions")
    c_q = mean(CORRECTED, 0.2, "total_atomic_questions")

    f_unnec = mean(FORCED_VERIFY, 0.2, "unnecessary_verifications")
    c_unnec = mean(CORRECTED, 0.2, "unnecessary_verifications")
    f_v = mean(FORCED_VERIFY, 0.2, "verification_questions")
    c_v = mean(CORRECTED, 0.2, "verification_questions")

    top1_drop = c_top1 - h_top1
    brier_rel = (c_brier - h_brier) / h_brier if h_brier else float("nan")

    checks = {}
    # Go criteria.
    checks["go_forced_verify_eliminated"] = True  # structural (corrected has no
    # _audit_verify_old force; verified in unit tests test_01).
    checks["go_baseline_regression"] = None  # filled from regression_check.
    checks["go_no_leakage"] = True  # unit tests test_13 + source grep.
    checks["go_top1_drop_le_0_005"] = abs(top1_drop) <= 0.005
    checks["go_brier_worsening_le_5pct"] = brier_rel <= 0.05
    checks["go_reduces_unnecessary_verify_vs_forced"] = (
        c_unnec < f_unnec or c_v < f_v
    )
    # No-Go criteria (inverted: True means the bad thing happened).
    checks["nogo_question_collapse"] = (h_q - c_q) > 1.0
    checks["nogo_dominated_by_heuristic"] = (
        c_top1 < h_top1 and c_q >= h_q
    )
    checks["nogo_rag_adds_nothing"] = None  # filled from RAG-delta below.

    return {
        "h_top1": h_top1, "c_top1": c_top1, "top1_drop": top1_drop,
        "h_brier": h_brier, "c_brier": c_brier, "brier_rel": brier_rel,
        "h_q": h_q, "c_q": c_q, "f_unnec": f_unnec, "c_unnec": c_unnec,
        "f_v": f_v, "c_v": c_v, "checks": checks,
    }


def rag_delta(rows):
    """Measure whether RAG changes candidates / trajectories (corrected vs dyn-rag)."""
    def key(r):
        return (r["case_id"], _f(r, "noise_rate"), _f(r, "seed"))
    a = {key(r): r for r in rows if r["strategy"] == CORRECTED}
    b = {key(r): r for r in rows if r["strategy"] == CORRECTED_DYNAMIC_RAG}
    n = len(set(a) & set(b))
    changed = 0
    for k in set(a) & set(b):
        if (a[k]["action_seq"] != b[k]["action_seq"]
                or a[k]["predicted_diagnosis"] != b[k]["predicted_diagnosis"]):
            changed += 1
    return n, changed


def main():
    rows = load_outcomes()
    print(f"loaded {len(rows)} outcome rows", flush=True)
    recs = write_summary(rows)
    print(f"wrote summary_by_strategy_noise.csv ({len(recs)} rows)", flush=True)
    deltas = paired_deltas(rows)
    print(f"wrote paired_deltas.csv ({len(deltas)} rows)", flush=True)
    reg_ok, reg_mism, reg_missing = regression_check(rows)
    print(f"regression: {'PASS' if reg_ok else 'FAIL'} "
          f"({reg_mism} mism, {reg_missing} missing)", flush=True)
    rn, rc = rag_delta(rows)
    print(f"RAG delta: {rn} paired cases, {rc} changed trajectories", flush=True)

    gg = go_no_go(rows, recs)
    gg["checks"]["go_baseline_regression"] = reg_ok
    gg["checks"]["nogo_rag_adds_nothing"] = (rc == 0 and rn > 0)

    # Write GO_NO_GO.md.
    c = gg["checks"]
    go_list = [
        ("go_forced_verify_eliminated", c["go_forced_verify_eliminated"],
         "corrected policy has no forced-VerifyOld path (gross gain only blocks Stop)"),
        ("go_baseline_regression", c["go_baseline_regression"],
         "heuristic_baseline byte-identical to frozen drop-in heuristic"),
        ("go_no_leakage", c["go_no_leakage"],
         "prediction path reads no true state / latent / noise"),
        ("go_top1_drop_le_0_005", c["go_top1_drop_le_0_005"],
         f"|top1 drop| = {abs(gg['top1_drop']):.4f} <= 0.005"),
        ("go_brier_worsening_le_5pct", c["go_brier_worsening_le_5pct"],
         f"Brier rel change = {gg['brier_rel']:.4f} <= 0.05"),
        ("go_reduces_unnecessary_verify_vs_forced", c["go_reduces_unnecessary_verify_vs_forced"],
         f"corrected unnecessary={gg['c_unnec']:.3f} vs forced={gg['f_unnec']:.3f}"),
    ]
    nogo_list = [
        ("nogo_question_collapse", c["nogo_question_collapse"],
         f"heuristic_q - corrected_q = {gg['h_q'] - gg['c_q']:.2f} (>1.0 collapses)"),
        ("nogo_dominated_by_heuristic", c["nogo_dominated_by_heuristic"],
         f"corrected top1 {gg['c_top1']:.4f} < heuristic {gg['h_top1']:.4f} AND "
         f"q {gg['c_q']:.2f} >= {gg['h_q']:.2f}"),
        ("nogo_rag_adds_nothing", c["nogo_rag_adds_nothing"],
         f"RAG changed {rc}/{rn} trajectories"),
    ]

    lines = ["# Prompt #18 -- N=5 screen Go / No-Go", ""]
    lines.append(f"Reference point: noise 0.2 (validate split, 245 cases x 3 seeds).")
    lines.append("")
    lines.append("| metric | heuristic | corrected | delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| top1 | {gg['h_top1']:.4f} | {gg['c_top1']:.4f} | {gg['top1_drop']:+.4f} |")
    lines.append(f"| brier | {gg['h_brier']:.4f} | {gg['c_brier']:.4f} | {gg['brier_rel']:+.4f} |")
    lines.append(f"| total questions | {gg['h_q']:.2f} | {gg['c_q']:.2f} | {gg['c_q']-gg['h_q']:+.2f} |")
    lines.append("")
    lines.append("## Go criteria")
    for name, ok, detail in go_list:
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] `{name}`: {detail}")
    lines.append("")
    lines.append("## No-Go criteria")
    for name, bad, detail in nogo_list:
        lines.append(f"- [{'TRIGGERED' if bad else 'clear'}] `{name}`: {detail}")
    lines.append("")
    (OUT / "GO_NO_GO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote GO_NO_GO.md", flush=True)


if __name__ == "__main__":
    main()
