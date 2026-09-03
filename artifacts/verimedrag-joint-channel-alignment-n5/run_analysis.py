"""Phase 8C -- N=5 screen analysis.

Consumes ``case_outcomes.csv`` and ``reliability_predictions.csv`` (produced by
``run_screen.py``) and derives the remaining analysis artifacts:

* ``rho_sensitivity.csv``      -- matched rho_env=rho_model metric sensitivity
* ``matched_mismatched_deltas.csv`` -- cost of env/model correlation mismatch
* ``reliability_metrics.json`` -- p_mode / p_wrong AUROC / AUPRC / Brier / ECE
* ``scenario_summary.json``    -- per-scenario aggregates for the reports

All aggregates pool across cases x noise rates x seeds within a (strategy,
rho_env, rho_model) cell.  Nothing here reads the test split or fits rho.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent

STRATEGY_LABELS = {
    "heuristic_baseline": "heuristic_baseline",
    "unified_brier_audit_corrected": "unified_brier_audit_corrected",
    "joint_channel_brier_audit": "joint_channel_brier_audit",
}

MATCHED_SCENARIOS = ((0.0, 0.0), (0.5, 0.5), (0.9, 0.9))
# mismatched pairs share rho_model=0.5
MISMATCH_AT_05 = ((0.0, 0.5), (0.5, 0.5), (0.9, 0.5))


def _f(row, name, default=0.0):
    v = row.get(name, "")
    if v == "" or v is None:
        return default
    return float(v)


def _i(row, name, default=0):
    v = row.get(name, "")
    if v == "" or v is None:
        return default
    return int(float(v))


def load_outcomes():
    path = OUT / "case_outcomes.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def ece_from_rows(rows):
    """Expected calibration error over top-1 confidence (10 equal-width bins)."""
    bins = [[] for _ in range(10)]
    for r in rows:
        conf = _f(r, "confidence")
        correct = _i(r, "correct_top1")
        idx = min(9, int(conf * 10))
        bins[idx].append((conf, correct))
    ece = 0.0
    total = len(rows)
    for b in bins:
        if not b:
            continue
        mean_conf = sum(c for c, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += (len(b) / total) * abs(mean_conf - acc)
    return ece


def scenario_key(strategy, rho_env, rho_model):
    env = float(rho_env)
    model = float(rho_model) if rho_model not in ("", None) else -1.0
    return (strategy, env, model)


def aggregate(rows):
    n = len(rows)
    if n == 0:
        return {}
    top1 = sum(_i(r, "correct_top1") for r in rows) / n
    top3 = sum(_i(r, "correct_top3") for r in rows) / n
    brier = sum(_f(r, "brier_score") for r in rows) / n
    nll = sum(_f(r, "nll_score") for r in rows) / n
    entropy = sum(_f(r, "posterior_entropy") for r in rows) / n
    conf = sum(_f(r, "confidence") for r in rows) / n
    newq = sum(_i(r, "new_questions") for r in rows) / n
    verq = sum(_i(r, "verification_questions") for r in rows) / n
    turns = sum(_i(r, "total_atomic_questions") for r in rows) / n
    unnec = sum(_i(r, "unnecessary_verifications") for r in rows)
    resolved = sum(_i(r, "resolved_wrong_reports") for r in rows)
    c2w = sum(_i(r, "correct_to_wrong_flips") for r in rows)
    w2c = sum(_i(r, "wrong_to_correct_flips") for r in rows)
    premat = sum(_i(r, "premature_stop") for r in rows)
    uncert = sum(_i(r, "uncertain_output") for r in rows)
    return {
        "n": n,
        "top1_accuracy": round(top1, 6),
        "top3_accuracy": round(top3, 6),
        "mean_brier": round(brier, 6),
        "mean_nll": round(nll, 6),
        "mean_confidence": round(conf, 6),
        "ece": round(ece_from_rows(rows), 6),
        "mean_posterior_entropy": round(entropy, 6),
        "mean_new_questions": round(newq, 4),
        "mean_verification_questions": round(verq, 4),
        "mean_total_turns": round(turns, 4),
        "unnecessary_verifications": unnec,
        "resolved_wrong_reports": resolved,
        "correct_to_wrong_flips": c2w,
        "wrong_to_correct_flips": w2c,
        "premature_stop": premat,
        "uncertain_output": uncert,
    }


def _auc(scores, labels):
    """Trapezoid AUROC via the Mann-Whitney formulation (label 1 = positive)."""
    import math

    pairs = list(zip(scores, labels))
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    npos, nneg = len(pos), len(neg)
    pos_sorted = sorted(pos)
    rank_sum = 0
    for s in pos_sorted:
        # number of negatives strictly below s + 0.5 * ties
        below = sum(1 for x in neg if x < s)
        ties = sum(1 for x in neg if x == s)
        rank_sum += below + 0.5 * ties
    return rank_sum / (npos * nneg)


def _auprc(scores, labels):
    """Area under precision-recall curve (label 1 = positive)."""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0], reverse=True)
    tp = 0
    fp = 0
    npos = sum(labels)
    if npos == 0:
        return float("nan")
    auc = 0.0
    prev_recall = 0.0
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
            recall = tp / npos
            precision = tp / (tp + fp)
            auc += (recall - prev_recall) * precision
            prev_recall = recall
    return auc


def _brier(scores, labels):
    if not labels:
        return float("nan")
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(labels)


def _ece(scores, labels, nbins=10):
    if not labels:
        return float("nan")
    bins = [[] for _ in range(nbins)]
    for s, y in zip(scores, labels):
        idx = min(nbins - 1, int(s * nbins))
        bins[idx].append((s, y))
    ece = 0.0
    n = len(labels)
    for b in bins:
        if not b:
            continue
        mc = sum(s for s, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += (len(b) / n) * abs(mc - acc)
    return ece


def reliability_metrics(rows):
    """p_mode vs true_misreported, p_wrong vs true_wrong (JOINT strategy only)."""
    out = {}
    for pred, target in (("p_mode", "true_misreported"), ("p_wrong", "true_wrong")):
        sub = [( _f(r, pred), _i(r, target)) for r in rows
               if r.get(pred, "") != "" and r.get(target, "") != ""]
        scores = [s for s, _ in sub]
        labels = [y for _, y in sub]
        out[pred] = {
            "n": len(sub),
            "positives": sum(labels),
            "auroc": round(_auc(scores, labels), 6) if sub else None,
            "auprc": round(_auprc(scores, labels), 6) if sub else None,
            "brier": round(_brier(scores, labels), 6) if sub else None,
            "ece": round(_ece(scores, labels), 6) if sub else None,
        }
    return out


def write_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    outcomes = load_outcomes()
    if not outcomes:
        print("no case_outcomes.csv; run run_screen.py first")
        return 1

    # group by (strategy, rho_env, rho_model)
    groups = defaultdict(list)
    for r in outcomes:
        groups[scenario_key(r["strategy"], r["rho_env"], r["rho_model"])].append(r)

    agg = {k: aggregate(v) for k, v in groups.items()}

    # --- scenario summary ---------------------------------------------------- #
    summary = {}
    for (strategy, rho_env, rho_model), a in sorted(agg.items()):
        summary[f"{strategy}|{rho_env:g}|{rho_model:g}"] = a
    (OUT / "scenario_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # --- rho_sensitivity.csv (matched, JOINT strategy) ----------------------- #
    rho_rows = []
    for strategy in STRATEGY_LABELS:
        for rho_env, rho_model in MATCHED_SCENARIOS:
            key = (strategy, float(rho_env), float(rho_model))
            if key not in agg:
                continue
            a = agg[key]
            rho_rows.append({
                "strategy": strategy,
                "rho_env": rho_env,
                "rho_model": rho_model,
                "scenario": f"matched_{rho_env:g}_{rho_model:g}",
                **a,
            })
    metric_fields = [
        "n", "top1_accuracy", "top3_accuracy", "mean_brier", "mean_nll",
        "mean_confidence", "ece", "mean_posterior_entropy",
        "mean_new_questions", "mean_verification_questions", "mean_total_turns",
        "unnecessary_verifications", "resolved_wrong_reports",
        "correct_to_wrong_flips", "wrong_to_correct_flips", "premature_stop",
        "uncertain_output",
    ]
    rho_fields = ["strategy", "rho_env", "rho_model", "scenario"] + metric_fields
    write_csv(rho_rows, OUT / "rho_sensitivity.csv", rho_fields)

    # --- matched_mismatched_deltas.csv -------------------------------------- #
    # rho_model=0.5: matched (0.5/0.5) vs mismatched (0.0/0.5, 0.9/0.5)
    delta_rows = []
    matched = agg.get(("joint_channel_brier_audit", 0.5, 0.5), {})
    metric_keys = ["top1_accuracy", "top3_accuracy", "mean_brier", "mean_nll",
                   "mean_confidence", "ece", "mean_posterior_entropy",
                   "mean_new_questions", "mean_verification_questions",
                   "mean_total_turns"]
    for rho_env in (0.0, 0.9):
        other = agg.get(("joint_channel_brier_audit", float(rho_env), 0.5), {})
        if not other:
            continue
        for m in metric_keys:
            mv = matched.get(m)
            ov = other.get(m)
            if mv is None or ov is None:
                continue
            delta_rows.append({
                "metric": m,
                "rho_model": 0.5,
                "matched_env": 0.5,
                "mismatched_env": rho_env,
                "matched_value": mv,
                "mismatched_value": ov,
                "delta": round(ov - mv, 6),
            })
    write_csv(
        delta_rows, OUT / "matched_mismatched_deltas.csv",
        ["metric", "rho_model", "matched_env", "mismatched_env",
         "matched_value", "mismatched_value", "delta"],
    )

    # --- reliability metrics (JOINT only) ------------------------------------ #
    rel_path = OUT / "reliability_predictions.csv"
    rel = {}
    if rel_path.exists():
        with open(rel_path, newline="") as f:
            rel_rows = list(csv.DictReader(f))
        for (strategy, rho_env, rho_model), a in sorted(agg.items()):
            sub = [r for r in rel_rows
                   if r["strategy"] == strategy
                   and float(r["rho_env"]) == rho_env
                   and float(r["rho_model"]) == rho_model]
            if sub:
                rel[f"{strategy}|{rho_env:g}|{rho_model:g}"] = reliability_metrics(sub)
        (OUT / "reliability_metrics.json").write_text(
            json.dumps(rel, indent=2), encoding="utf-8"
        )

    # --- console summary ----------------------------------------------------- #
    print("per-scenario aggregates:")
    for (strategy, rho_env, rho_model), a in sorted(agg.items()):
        print(
            f"  {strategy:32s} env={rho_env:g} model={rho_model:g} "
            f"n={a['n']:4d} top1={a['top1_accuracy']:.4f} "
            f"brier={a['mean_brier']:.4f} ece={a['ece']:.4f} "
            f"turns={a['mean_total_turns']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
