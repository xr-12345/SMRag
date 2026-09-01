"""Phase 2C confirmation analysis.

Reads ``case_outcomes.csv`` / ``turn_observations.csv`` / ``candidate_events.csv``
from ``run_confirmation.py`` and produces the Phase-2C products:

  summary_primary.csv, summary_full_n20.csv, paired_deltas_primary.csv,
  paired_deltas_full.csv, diagnosis_transition_counts.csv,
  retrieval_trigger_analysis.csv, calibration_metrics.csv,
  bootstrap_results.csv, confirmation_metrics.json.

Statistics (as preregistered in CONFIG.json): case-level paired deltas with a
disease-stratified cluster bootstrap (unit = case_id, 3 seeds resampled
together, within-disease with replacement, 10 000 reps, seed 2026) for noise
0.2 / 0.3 / pooled 0.2+0.3.  Top-1 gets a wrong->correct / correct->wrong
decomposition plus exact McNemar (supplementary); retrieval-triggered precision
gets a Wilson 95% CI.  ECE uses 15 equal-width bins.

Two bootstrap statistics are used:
  * mean paired delta  (brier, top1, top3, nll, margin, question counts) —
    bootstrap the mean per-case delta.
  * aggregate-rate delta (unnecessary-verification rate, conflict-resolution
    rate) — bootstrap rate(a) - rate(b) on the resampled sample.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from powerful_medrag.gate_analysis import _expected_calibration_error

OUT = Path(__file__).resolve().parent
BASELINE = "rank_only_w0.0"
CONFIGS = ["rank_only_w0.0", "rank_only_w1.0", "joint_gate_w1.0"]
NOISE_LEVELS = ["0.0", "0.1", "0.2", "0.3"]
FOCUS_NOISE = ["0.2", "0.3"]
POOLED_LABEL = "0.2+0.3"
N_BOOT = 10_000
BOOT_SEED = 2026
EBS = 15  # ECE bins

COMPARISONS = [
    ("ranking", "rank_only_w1.0", "rank_only_w0.0"),
    ("gating", "joint_gate_w1.0", "rank_only_w1.0"),
    ("total", "joint_gate_w1.0", "rank_only_w0.0"),
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


def nk(row) -> str:
    return f"{fnum(row['noise_rate']):.1f}"


def sk(row) -> str:
    return str(i(row["seed"]))


def primary_ids() -> set[str]:
    out = set()
    for row in load("confirmation_case_manifest.csv"):
        if i(row["in_primary_confirmation"]) == 1:
            out.add(row["case_id"])
    return out


def diag_of(cases: list[dict]) -> dict[str, str]:
    return {c["case_id"]: c["diagnosis"] for c in cases}


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #


def summarize(cases: list[dict], subset_ids: set[str], subset_label: str) -> list[dict]:
    rows = []
    for config in CONFIGS:
        for noise in NOISE_LEVELS:
            sub = [
                c
                for c in cases
                if c["config"] == config
                and nk(c) == noise
                and c["case_id"] in subset_ids
            ]
            if not sub:
                continue
            n = len(sub)
            verifies = sum(i(c["verification_questions"]) for c in sub)
            unnec = sum(i(c["unnecessary_verifications"]) for c in sub)
            resolved = sum(i(c["resolved_wrong_reports"]) for c in sub)
            retrig = sum(i(c["retrieval_triggered_verifications"]) for c in sub)
            retrig_hit = sum(i(c["retrieval_triggered_hits"]) for c in sub)
            gate_asknew = sum(i(c["gate_open_but_asknew"]) for c in sub)
            gate_verify = sum(i(c["gate_open_and_verify"]) for c in sub)
            labels = [i(c["correct_top1"]) for c in sub]
            confs = [fnum(c["confidence"]) for c in sub]
            ece = _expected_calibration_error(labels, confs, bins=EBS)
            stop_counts = Counter(c["stop_reason"] for c in sub)
            rows.append(
                {
                    "subset": subset_label,
                    "config": config,
                    "noise_rate": noise,
                    "n_case_seed_pairs": n,
                    "top1_accuracy": round(sum(i(c["correct_top1"]) for c in sub) / n, 6),
                    "top3_accuracy": round(sum(i(c["correct_top3"]) for c in sub) / n, 6),
                    "brier_score_mean": round(sum(fnum(c["brier_score"]) for c in sub) / n, 6),
                    "nll_mean": round(sum(fnum(c["nll"]) for c in sub) / n, 6),
                    "posterior_margin_mean": round(
                        sum(fnum(c["posterior_margin"]) for c in sub) / n, 6
                    ),
                    "ece_15bin": round(ece, 6),
                    "premature_stop_rate": round(sum(i(c["premature_stop"]) for c in sub) / n, 6),
                    "uncertain_output_rate": round(sum(i(c["uncertain_output"]) for c in sub) / n, 6),
                    "new_questions_mean": round(sum(i(c["new_questions"]) for c in sub) / n, 6),
                    "verification_questions_mean": round(verifies / n, 6),
                    "total_atomic_questions_mean": round(
                        sum(i(c["total_atomic_questions"]) for c in sub) / n, 6
                    ),
                    "unnecessary_verification_rate": round(
                        (unnec / verifies) if verifies else 0.0, 6
                    ),
                    "conflict_resolution_rate": round(
                        (resolved / verifies) if verifies else 0.0, 6
                    ),
                    "retrieval_triggered_verifications": retrig,
                    "retrieval_triggered_hits": retrig_hit,
                    "retrieval_triggered_cases": sum(
                        1 for c in sub if i(c["retrieval_triggered_verifications"]) > 0
                    ),
                    "triggered_true_hit_rate": round(
                        (retrig_hit / retrig) if retrig else 0.0, 6
                    ),
                    "gate_open_but_asknew": gate_asknew,
                    "gate_open_and_verify": gate_verify,
                    "ordinary_retrieves_mean": round(
                        sum(i(c["ordinary_retrieves"]) for c in sub) / n, 3
                    ),
                    "counterfactual_retrieves_mean": round(
                        sum(i(c["counterfactual_retrieves"]) for c in sub) / n, 3
                    ),
                    "wall_clock_mean": round(sum(fnum(c["wall_clock"]) for c in sub) / n, 4),
                    "stop_reason_counts": json.dumps(dict(sorted(stop_counts.items()))),
                }
            )
    return rows


def write_rows(rows: list[dict], path: str) -> None:
    if not rows:
        return
    with open(OUT / path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# paired arrays + cluster bootstrap
# --------------------------------------------------------------------------- #


def pair_rows(cases, a: str, b: str, subset_ids: set[str], level: str):
    """Return (keys, a_rows, b_rows) for a config pair over a level.

    The analysis unit is (case_id, seed, noise_rate): for the pooled level the
    same case appears once per (seed, noise), so the key MUST include the noise
    to avoid collision.  ``level`` is a single noise or POOLED_LABEL (0.2+0.3).
    """
    levels = FOCUS_NOISE if level == POOLED_LABEL else [level]
    a_rows, b_rows = {}, {}
    for c in cases:
        if c["case_id"] not in subset_ids:
            continue
        if nk(c) not in levels:
            continue
        key = (c["case_id"], sk(c), nk(c))
        if c["config"] == a:
            a_rows[key] = c
        elif c["config"] == b:
            b_rows[key] = c
    keys = sorted(set(a_rows) & set(b_rows))
    return keys, a_rows, b_rows


def _case_sums(d: dict, keys):
    out: dict[str, float] = defaultdict(float)
    for (cid, _s, _n) in keys:
        out[cid] += d.get((cid, _s, _n), 0.0)
    return out


def _case_obs(keys):
    out: dict[str, int] = defaultdict(int)
    for (cid, _s, _n) in keys:
        out[cid] += 1
    return out


def _disease_groups(keys, diag):
    by = defaultdict(set)
    for (cid, _s, _n) in keys:
        by[diag.get(cid, "UNK")].add(cid)
    return {d: sorted(cids) for d, cids in by.items()}


def mean_delta_bootstrap(delta: dict, keys, diag) -> dict:
    case_sum = _case_sums(delta, keys)
    case_obs = _case_obs(keys)
    groups = _disease_groups(keys, diag)
    diseases = sorted(groups)
    rng = random.Random(BOOT_SEED)
    reps = []
    for _ in range(N_BOOT):
        tn = 0.0
        td = 0
        for d in diseases:
            cids = groups[d]
            nd = len(cids)
            for _ in range(nd):
                cid = cids[rng.randrange(nd)]
                tn += case_sum.get(cid, 0.0)
                td += case_obs.get(cid, 0)
        reps.append(tn / td if td else float("nan"))
    reps = sorted(r for r in reps if not math.isnan(r))
    n = len(reps)
    mean = sum(reps) / n
    return {
        "mean": mean,
        "ci_low": reps[int(0.025 * n)],
        "ci_high": reps[min(int(0.975 * n), n - 1)],
        "std": (sum((r - mean) ** 2 for r in reps) / n) ** 0.5,
        "n_reps": n,
    }


def rate_delta_bootstrap(a_num, a_den, b_num, b_den, keys, diag) -> dict:
    """Bootstrap rate(a) - rate(b), where rate = sum(num)/sum(den)."""
    an = _case_sums(a_num, keys)
    ad = _case_sums(a_den, keys)
    bn = _case_sums(b_num, keys)
    bd = _case_sums(b_den, keys)
    groups = _disease_groups(keys, diag)
    diseases = sorted(groups)
    rng = random.Random(BOOT_SEED)
    reps = []
    for _ in range(N_BOOT):
        san = sad = sbn = sbd = 0.0
        for d in diseases:
            cids = groups[d]
            nd = len(cids)
            for _ in range(nd):
                cid = cids[rng.randrange(nd)]
                san += an.get(cid, 0.0)
                sad += ad.get(cid, 0.0)
                sbn += bn.get(cid, 0.0)
                sbd += bd.get(cid, 0.0)
        ra = san / sad if sad else float("nan")
        rb = sbn / sbd if sbd else float("nan")
        reps.append(ra - rb if not (math.isnan(ra) or math.isnan(rb)) else float("nan"))
    reps = sorted(r for r in reps if not math.isnan(r))
    n = len(reps)
    mean = sum(reps) / n
    return {
        "mean": mean,
        "ci_low": reps[int(0.025 * n)],
        "ci_high": reps[min(int(0.975 * n), n - 1)],
        "std": (sum((r - mean) ** 2 for r in reps) / n) ** 0.5,
        "n_reps": n,
    }


def bootstrap_table(cases, diag, subset_ids, subset_label) -> list[dict]:
    rows = []
    for name, a, b in COMPARISONS:
        for level in NOISE_LEVELS + [POOLED_LABEL]:
            keys, a_rows, b_rows = pair_rows(cases, a, b, subset_ids, level)
            if not keys:
                continue
            # mean-delta metrics
            deltas = {
                "top1": {k: i(a_rows[k]["correct_top1"]) - i(b_rows[k]["correct_top1"]) for k in keys},
                "top3": {k: i(a_rows[k]["correct_top3"]) - i(b_rows[k]["correct_top3"]) for k in keys},
                "brier": {k: fnum(a_rows[k]["brier_score"]) - fnum(b_rows[k]["brier_score"]) for k in keys},
                "nll": {k: fnum(a_rows[k]["nll"]) - fnum(b_rows[k]["nll"]) for k in keys},
                "margin": {k: fnum(a_rows[k]["posterior_margin"]) - fnum(b_rows[k]["posterior_margin"]) for k in keys},
                "total_questions": {k: i(a_rows[k]["total_atomic_questions"]) - i(b_rows[k]["total_atomic_questions"]) for k in keys},
            }
            for mname, delta in deltas.items():
                pt = sum(delta.values()) / len(delta)
                boot = mean_delta_bootstrap(delta, keys, diag)
                rows.append(
                    {
                        "subset": subset_label,
                        "comparison": name,
                        "a": a,
                        "b": b,
                        "level": level,
                        "metric": mname,
                        "statistic": "mean_paired_delta",
                        "point_estimate": round(pt, 8),
                        "boot_mean": round(boot["mean"], 8),
                        "ci_low": round(boot["ci_low"], 8),
                        "ci_high": round(boot["ci_high"], 8),
                        "boot_std": round(boot["std"], 8),
                        "n_pairs": len(delta),
                    }
                )
            # rate-delta metrics
            unnec = {
                "a_num": {k: i(a_rows[k]["unnecessary_verifications"]) for k in keys},
                "a_den": {k: i(a_rows[k]["verification_questions"]) for k in keys},
                "b_num": {k: i(b_rows[k]["unnecessary_verifications"]) for k in keys},
                "b_den": {k: i(b_rows[k]["verification_questions"]) for k in keys},
            }
            conflict = {
                "a_num": {k: i(a_rows[k]["resolved_wrong_reports"]) for k in keys},
                "a_den": {k: i(a_rows[k]["verification_questions"]) for k in keys},
                "b_num": {k: i(b_rows[k]["resolved_wrong_reports"]) for k in keys},
                "b_den": {k: i(b_rows[k]["verification_questions"]) for k in keys},
            }
            for mname, d in (("unnecessary_verification_rate", unnec), ("conflict_resolution_rate", conflict)):
                ra = sum(d["a_num"].values()) / sum(d["a_den"].values()) if sum(d["a_den"].values()) else 0.0
                rb = sum(d["b_num"].values()) / sum(d["b_den"].values()) if sum(d["b_den"].values()) else 0.0
                pt = ra - rb
                boot = rate_delta_bootstrap(d["a_num"], d["a_den"], d["b_num"], d["b_den"], keys, diag)
                rows.append(
                    {
                        "subset": subset_label,
                        "comparison": name,
                        "a": a,
                        "b": b,
                        "level": level,
                        "metric": mname,
                        "statistic": "rate_delta",
                        "point_estimate": round(pt, 8),
                        "boot_mean": round(boot["mean"], 8),
                        "ci_low": round(boot["ci_low"], 8),
                        "ci_high": round(boot["ci_high"], 8),
                        "boot_std": round(boot["std"], 8),
                        "n_pairs": len(keys),
                    }
                )
    return rows


def paired_delta_rows(cases, subset_ids, subset_label) -> list[dict]:
    rows = []
    for name, a, b in COMPARISONS:
        for level in NOISE_LEVELS + [POOLED_LABEL]:
            keys, a_rows, b_rows = pair_rows(cases, a, b, subset_ids, level)
            for (cid, s, nz) in keys:
                ra, rb = a_rows[(cid, s, nz)], b_rows[(cid, s, nz)]
                rows.append(
                    {
                        "subset": subset_label,
                        "comparison": name,
                        "a": a,
                        "b": b,
                        "level": level,
                        "noise_rate": nz,
                        "case_id": cid,
                        "seed": s,
                        "delta_top1": i(ra["correct_top1"]) - i(rb["correct_top1"]),
                        "delta_top3": i(ra["correct_top3"]) - i(rb["correct_top3"]),
                        "delta_brier": round(fnum(ra["brier_score"]) - fnum(rb["brier_score"]), 8),
                        "delta_nll": round(fnum(ra["nll"]) - fnum(rb["nll"]), 8),
                        "delta_margin": round(fnum(ra["posterior_margin"]) - fnum(rb["posterior_margin"]), 8),
                        "delta_total_questions": i(ra["total_atomic_questions"]) - i(rb["total_atomic_questions"]),
                        "delta_new_questions": i(ra["new_questions"]) - i(rb["new_questions"]),
                        "delta_verification_questions": i(ra["verification_questions"]) - i(rb["verification_questions"]),
                    }
                )
    return rows


# --------------------------------------------------------------------------- #
# diagnosis transitions + McNemar
# --------------------------------------------------------------------------- #


def transitions(cases, a, b, subset_ids, level: str) -> dict:
    keys, a_rows, b_rows = pair_rows(cases, a, b, subset_ids, level)
    w2c = c2w = cc = ww = pred_changed = 0
    brier_only = 0
    for k in keys:
        ra, rb = a_rows[k], b_rows[k]
        ta, tb = i(ra["correct_top1"]), i(rb["correct_top1"])
        if tb == 0 and ta == 1:
            w2c += 1
        elif tb == 1 and ta == 0:
            c2w += 1
        elif ta == 1 and tb == 1:
            cc += 1
        elif ta == 0 and tb == 0:
            ww += 1
        if ra["predicted_diagnosis"] != rb["predicted_diagnosis"]:
            pred_changed += 1
        if fnum(ra["brier_score"]) < fnum(rb["brier_score"]) and ta == tb:
            brier_only += 1
    return {
        "n_pairs": len(keys),
        "wrong_to_correct": w2c,
        "correct_to_wrong": c2w,
        "correct_to_correct": cc,
        "wrong_to_wrong": ww,
        "final_prediction_changed": pred_changed,
        "brier_only_improved_no_top1_change": brier_only,
    }


def trajectory_changed(turns, a, b, subset_ids, levels) -> dict:
    def sig(cfg_rows):
        return tuple(
            (t["chosen_action"], (t["report_index"] or ""))
            for t in sorted(cfg_rows, key=lambda t: int(t["turn_index"]))
        )

    a_map = defaultdict(list)
    b_map = defaultdict(list)
    for t in turns:
        if t["case_id"] not in subset_ids or nk(t) not in levels:
            continue
        key = (t["case_id"], sk(t), nk(t))
        if t["config"] == a:
            a_map[key].append(t)
        elif t["config"] == b:
            b_map[key].append(t)
    keys = sorted(set(a_map) & set(b_map))
    changed = sum(1 for k in keys if sig(a_map[k]) != sig(b_map[k]))
    return {"n_pairs": len(keys), "trajectory_changed": changed}


def mc_nemar(cases, a, b, subset_ids, level: str):
    n10 = n01 = 0
    keys, a_rows, b_rows = pair_rows(cases, a, b, subset_ids, level)
    for k in keys:
        if i(a_rows[k]["correct_top1"]) == 1 and i(b_rows[k]["correct_top1"]) == 0:
            n10 += 1
        elif i(a_rows[k]["correct_top1"]) == 0 and i(b_rows[k]["correct_top1"]) == 1:
            n01 += 1
    return {"wrong_to_correct": n10, "correct_to_wrong": n01, "mcnemar_p": exact_mcnemar(n10, n01)}


def exact_mcnemar(n10, n01):
    n = n10 + n01
    if n == 0:
        return 1.0
    lo = min(n10, n01)
    p_lo = sum(math.comb(n, k) * 0.5 ** n for k in range(lo + 1))
    return min(1.0, 2.0 * p_lo)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - margin, center + margin)


# --------------------------------------------------------------------------- #
# retrieval trigger analysis
# --------------------------------------------------------------------------- #


def trigger_analysis(cands, cases, subset_ids, levels) -> list[dict]:
    rows = []
    for config in CONFIGS:
        sub_cands = [
            c for c in cands
            if c["config"] == config
            and c["case_id"] in subset_ids
            and nk(c) in levels
        ]
        comparable = [c for c in sub_cands if i(c["is_comparable"]) == 1]
        wrong_all = sum(i(c["is_actually_wrong"]) for c in comparable)
        base_rate = wrong_all / len(comparable) if comparable else 0.0
        triggered = [c for c in sub_cands if i(c["was_triggered"]) == 1]
        triggered_comp = [c for c in triggered if i(c["is_comparable"]) == 1]
        trig_wrong = sum(i(c["is_actually_wrong"]) for c in triggered_comp)
        trig_prec = trig_wrong / len(triggered_comp) if triggered_comp else 0.0
        lo, hi = wilson(trig_wrong, len(triggered_comp))
        sub_cases = [
            c for c in cases
            if c["config"] == config
            and c["case_id"] in subset_ids
            and nk(c) in levels
        ]
        trig_verif = sum(i(c["retrieval_triggered_verifications"]) for c in sub_cases)
        trig_hit = sum(i(c["retrieval_triggered_hits"]) for c in sub_cases)
        avg_impact = (
            sum(fnum(c["normalized_retrieval_impact"]) for c in sub_cands) / len(sub_cands)
            if sub_cands
            else 0.0
        )
        rows.append(
            {
                "config": config,
                "n_candidates": len(sub_cands),
                "n_comparable_candidates": len(comparable),
                "candidate_misreport_base_rate": round(base_rate, 6),
                "n_triggered_verifications": len(triggered),
                "n_triggered_comparable": len(triggered_comp),
                "triggered_precision": round(trig_prec, 6),
                "triggered_precision_wilson_low": round(lo, 6),
                "triggered_precision_wilson_high": round(hi, 6),
                "precision_enrichment": round(
                    (trig_prec / base_rate) if base_rate else 0.0, 4
                ),
                "case_outcomes_triggered_verifications": trig_verif,
                "case_outcomes_triggered_hits": trig_hit,
                "avg_normalized_retrieval_impact": round(avg_impact, 6),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# calibration (ECE per config x noise)
# --------------------------------------------------------------------------- #


def calibration(cases, subset_ids, subset_label) -> list[dict]:
    rows = []
    for config in CONFIGS:
        for noise in NOISE_LEVELS:
            sub = [
                c for c in cases
                if c["config"] == config and nk(c) == noise and c["case_id"] in subset_ids
            ]
            if not sub:
                continue
            labels = [i(c["correct_top1"]) for c in sub]
            confs = [fnum(c["confidence"]) for c in sub]
            ece = _expected_calibration_error(labels, confs, bins=EBS)
            rows.append(
                {
                    "subset": subset_label,
                    "config": config,
                    "noise_rate": noise,
                    "n": len(sub),
                    "ece_15bin": round(ece, 6),
                    "mean_confidence": round(sum(confs) / len(confs), 6),
                    "mean_accuracy": round(sum(labels) / len(labels), 6),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    cases = load("case_outcomes.csv")
    turns = load("turn_observations.csv")
    cands = load("candidate_events.csv")
    diag = diag_of(cases)
    prim = primary_ids()
    full = {c["case_id"] for c in cases}

    print(f"loaded: {len(cases)} case rows, {len(turns)} turn rows, {len(cands)} candidate rows")
    print(f"primary subset: {len(prim)} cases; full subset: {len(full)} cases")

    sp = summarize(cases, prim, "primary")
    sf = summarize(cases, full, "full_n20")
    write_rows(sp, "summary_primary.csv")
    write_rows(sf, "summary_full_n20.csv")

    cal = calibration(cases, prim, "primary") + calibration(cases, full, "full_n20")
    write_rows(cal, "calibration_metrics.csv")

    boot = bootstrap_table(cases, diag, prim, "primary")
    boot += bootstrap_table(cases, diag, full, "full_n20")
    write_rows(boot, "bootstrap_results.csv")

    write_rows(paired_delta_rows(cases, prim, "primary"), "paired_deltas_primary.csv")
    write_rows(paired_delta_rows(cases, full, "full_n20"), "paired_deltas_full.csv")

    trans_rows = []
    for subset_ids, slab in ((prim, "primary"), (full, "full_n20")):
        for name, a, b in COMPARISONS:
            for level in FOCUS_NOISE + [POOLED_LABEL]:
                levels = FOCUS_NOISE if level == POOLED_LABEL else [level]
                t = transitions(cases, a, b, subset_ids, level)
                tr = trajectory_changed(turns, a, b, subset_ids, levels)
                m = mc_nemar(cases, a, b, subset_ids, level)
                trans_rows.append(
                    {
                        "subset": slab,
                        "comparison": name,
                        "a": a,
                        "b": b,
                        "level": level,
                        "n_pairs": t["n_pairs"],
                        "wrong_to_correct": t["wrong_to_correct"],
                        "correct_to_wrong": t["correct_to_wrong"],
                        "correct_to_correct": t["correct_to_correct"],
                        "wrong_to_wrong": t["wrong_to_wrong"],
                        "final_prediction_changed": t["final_prediction_changed"],
                        "trajectory_changed": tr["trajectory_changed"],
                        "trajectory_changed_but_prediction_unchanged": (
                            tr["trajectory_changed"] - t["final_prediction_changed"]
                        ),
                        "brier_only_improved_no_top1_change": t["brier_only_improved_no_top1_change"],
                        "mcnemar_wrong_to_correct": m["wrong_to_correct"],
                        "mcnemar_correct_to_wrong": m["correct_to_wrong"],
                        "mcnemar_p": round(m["mcnemar_p"], 6),
                    }
                )
    write_rows(trans_rows, "diagnosis_transition_counts.csv")

    trig = trigger_analysis(cands, cases, prim, FOCUS_NOISE)
    write_rows(trig, "retrieval_trigger_analysis.csv")

    print("wrote summaries, calibration, bootstrap, paired deltas, transitions, triggers")

    metrics = {
        "summary_primary": sp,
        "summary_full": sf,
        "bootstrap": boot,
        "transitions": trans_rows,
        "trigger_analysis": trig,
        "calibration": cal,
    }
    with open(OUT / "confirmation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("wrote confirmation_metrics.json")


if __name__ == "__main__":
    main()
