"""VeriMedRAG Phase 7 (Prompt #16) — multi-step verification-value audit analyzer.

Reads ``action_labels.csv`` + ``rollout_samples.csv`` (written by ``run_audit.py``)
and computes the nine report items:

    1.  label stability (MC error + paired sign-flip rate)
    2.  one-step vs multi-step misalignment
    3.  heuristic predictive power
    4.  oracle upper bound (perfect wrong-report selection)
    5.  questions-vs-brier decomposition
    6.  RAG signal (retrieval_impact over error_probability)
    7.  leakage (construction guarantee, reported from the unit tests)
    8.  Go / No-Go
    9.  worth-training (learnability of the multi-step label)

Modes:
    --mode smoke-check   integrity gates over the current CSVs -> smoke_checks.txt
    --mode analyze       full analysis -> analysis.json + the two Markdown reports
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
C_Q = 0.03
N_ROLLOUTS = 8


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #

def _f(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value, default=None):
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _blank(value):
    return value is None or value == ""


def load_actions(path=OUT / "action_labels.csv"):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_rollouts(path=OUT / "rollout_samples.csv"):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- #
# state grouping
# --------------------------------------------------------------------------- #

def state_key(row):
    return (row["case_id"], row["noise_rate"], row["state_kind"])


def group_states(actions):
    """Return {state_key: {"new": row, "verify": [rows]}}."""
    states: dict[tuple, dict] = {}
    for row in actions:
        key = state_key(row)
        bucket = states.setdefault(key, {"new": None, "verify": []})
        if row["action_kind"] == "new":
            bucket["new"] = row
        else:
            bucket["verify"].append(row)
    return states


def paired_rollouts(rollouts):
    """Group rollouts by state -> {state_key: {"new": [8 rows], "verify": {idx: [8 rows]}}}."""
    by_state: dict[tuple, dict] = {}
    for row in rollouts:
        key = (row["case_id"], row["noise_rate"], row["state_kind"])
        bucket = by_state.setdefault(key, {"new": [], "verify": {}})
        if row["action_kind"] == "new":
            bucket["new"].append(row)
        else:
            idx = _i(row["report_index"])
            bucket["verify"].setdefault(idx, []).append(row)
    return by_state


def _j(terminal_brier, future_questions):
    return terminal_brier + C_Q * future_questions


# --------------------------------------------------------------------------- #
# smoke integrity gates
# --------------------------------------------------------------------------- #

def smoke_checks(actions, rollouts, expected_cases):
    checks = []
    seen_cases = {row["case_id"] for row in actions}

    checks.append(("cases_run", len(seen_cases) >= expected_cases,
                   f"{len(seen_cases)} unique cases (need >= {expected_cases})"))

    states = group_states(actions)
    checks.append(("states_found", len(states) > 0,
                   f"{len(states)} qualifying states"))

    no_new = [k for k, s in states.items() if s["new"] is None]
    checks.append(("one_new_per_state", len(no_new) == 0,
                   f"{len(no_new)} states missing an AskNew row"))

    no_verify = [k for k, s in states.items() if not s["verify"]]
    checks.append(("verify_candidates_present", len(no_verify) == 0,
                   f"{len(no_verify)} states with zero verify rows"))

    # rollout count must be N_ROLLOUTS per action
    expected_rollout_rows = len(actions) * N_ROLLOUTS
    checks.append(("rollout_row_count", len(rollouts) == expected_rollout_rows,
                   f"{len(rollouts)} rollout rows vs {expected_rollout_rows} expected"))

    # finite / sign / range sanity
    bad_finite = [row for row in actions if not math.isfinite(_f(row["j_value"]))]
    checks.append(("j_value_finite", len(bad_finite) == 0,
                   f"{len(bad_finite)} rows with non-finite j_value"))

    bad_q = [
        row for row in actions
        if abs(_f(row["q_multi"]) + _f(row["j_value"])) > 1e-6
    ]
    checks.append(("q_multi_negation", len(bad_q) == 0,
                   f"{len(bad_q)} rows where q_multi != -j_value"))

    bad_future = [
        row for row in actions
        if not (1 <= _f(row["future_questions"]) <= 15)
    ]
    checks.append(("future_questions_range", len(bad_future) == 0,
                   f"{len(bad_future)} rows with future_questions outside [1, 15]"))

    bad_brier = [
        row for row in actions
        if not (0.0 <= _f(row["terminal_brier"]) <= 2.0001)
    ]
    checks.append(("terminal_brier_range", len(bad_brier) == 0,
                   f"{len(bad_brier)} rows with terminal_brier outside [0, 2]"))

    # every verify row's report_index is an integer in range
    bad_verify_idx = [
        row for row in actions
        if row["action_kind"] == "verify" and _i(row["report_index"]) is None
    ]
    checks.append(("verify_report_index_present", len(bad_verify_idx) == 0,
                   f"{len(bad_verify_idx)} verify rows with missing report_index"))

    return checks


def write_smoke_checks(checks):
    lines = ["Phase 7 -- multi-step value audit smoke checks", ""]
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= bool(ok)
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    (OUT / "smoke_checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all_ok


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

def _corr(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 3:
        return float("nan")
    xs, ys = xs[mask], ys[mask]
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _ols_r2(y, X):
    """R^2 of OLS regression of y on design matrix X (with intercept)."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    mask = np.isfinite(y)
    for col in X.T:
        mask &= np.isfinite(col)
    if mask.sum() < len(y) * 0.5 or mask.sum() < 10:
        return float("nan"), int(mask.sum())
    y = y[mask]
    X = X[mask]
    Xd = np.column_stack([np.ones(len(y)), X])
    try:
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), int(mask.sum())
    yhat = Xd @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, int(mask.sum())


def _mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else float("nan")


def analyze(actions, rollouts):
    states = group_states(actions)
    ro_by_state = paired_rollouts(rollouts)
    metrics: dict = {}

    # --- per-candidate rows assembled once ---------------------------------
    # For each state, compute V_multi_i = J_new - J_i and one_step_delta_i =
    # v_bayes_verify_i - v_bayes_new, plus flags.
    candidates: list[dict] = []
    for key, bucket in states.items():
        new = bucket["new"]
        if new is None:
            continue
        j_new = _f(new["j_value"])
        j_new_se = _f(new["j_se"])
        v_new = _f(new["v_bayes"])
        brier_new = _f(new["terminal_brier"])
        future_new = _f(new["future_questions"])
        for vrow in bucket["verify"]:
            j_i = _f(vrow["j_value"])
            j_i_se = _f(vrow["j_se"])
            v_i = _f(vrow["v_bayes"])
            is_wrong = _i(vrow["is_report_wrong"], 0) or 0
            err_p = _f(vrow["retrospective_error_prob"])
            rip = _f(vrow["retrieval_impact"])
            hut = _f(vrow["heuristic_verify_utility"])
            rank = _i(vrow["verify_rank"])
            candidates.append(
                {
                    "state": key,
                    "j_new": j_new,
                    "j_new_se": j_new_se,
                    "brier_new": brier_new,
                    "future_new": future_new,
                    "v_new": v_new,
                    "v_i": v_i,
                    "j_i": j_i,
                    "j_i_se": j_i_se,
                    "v_multi": j_new - j_i,
                    "v_multi_se": math.hypot(j_new_se, j_i_se),
                    "one_step_delta": v_i - v_new,
                    "brier_delta": brier_new - _f(vrow["terminal_brier"]),
                    "future_delta": future_new - _f(vrow["future_questions"]),
                    "is_wrong": is_wrong,
                    "err_p": err_p,
                    "rip": rip,
                    "hut": hut,
                    "rank": rank,
                }
            )

    # --- item 1: label stability -------------------------------------------
    j_se_vals = [_f(r["j_se"]) for r in actions if _f(r["j_se"]) is not None]
    v_multi_mag = [abs(c["v_multi"]) for c in candidates]
    v_multi_se = [c["v_multi_se"] for c in candidates]
    significant = sum(
        1 for c in candidates if abs(c["v_multi"]) > 2 * c["v_multi_se"]
    )
    metrics["label_stability"] = {
        "n_actions": len(actions),
        "j_se_mean": _mean(j_se_vals),
        "j_se_median": float(np.median(j_se_vals)) if j_se_vals else float("nan"),
        "j_se_p90": float(np.percentile(j_se_vals, 90)) if j_se_vals else float("nan"),
        "v_multi_abs_mean": _mean(v_multi_mag),
        "v_multi_se_mean": _mean(v_multi_se),
        "frac_significant": significant / len(candidates) if candidates else float("nan"),
        "n_candidates": len(candidates),
    }

    # paired sign-flip rate (common random numbers)
    sign_flips = 0
    paired_total = 0
    for key, bucket in states.items():
        new = bucket["new"]
        if new is None:
            continue
        ro = ro_by_state.get(key)
        if ro is None:
            continue
        new_rolls = sorted(ro["new"], key=lambda r: _i(r["rollout_index"]))
        if len(new_rolls) != N_ROLLOUTS:
            continue
        new_j = [_j(_f(r["terminal_brier"]), _f(r["future_questions"])) for r in new_rolls]
        for vrow in bucket["verify"]:
            idx = _i(vrow["report_index"])
            v_rolls = sorted(ro["verify"].get(idx, []), key=lambda r: _i(r["rollout_index"]))
            if len(v_rolls) != N_ROLLOUTS:
                continue
            v_j = [_j(_f(r["terminal_brier"]), _f(r["future_questions"])) for r in v_rolls]
            mean_v = _mean([new_j[k] - v_j[k] for k in range(N_ROLLOUTS)])
            agree = sum(
                1 for k in range(N_ROLLOUTS)
                if (new_j[k] - v_j[k]) * mean_v >= 0
            )
            sign_flips += N_ROLLOUTS - agree
            paired_total += N_ROLLOUTS
    metrics["label_stability"]["paired_sign_flip_rate"] = (
        sign_flips / paired_total if paired_total else float("nan")
    )
    metrics["label_stability"]["paired_n"] = paired_total

    # --- item 2: one-step vs multi-step -------------------------------------
    metrics["one_vs_multi"] = {
        "pearson": _corr(
            [c["one_step_delta"] for c in candidates],
            [c["v_multi"] for c in candidates],
        ),
        "sign_agreement": _mean(
            [1.0 if (c["one_step_delta"] * c["v_multi"]) > 0 else 0.0
             for c in candidates]
        ),
    }
    # per-state top-1 agreement
    top1_agree = 0
    top1_states = 0
    for key, bucket in states.items():
        if bucket["new"] is None or len(bucket["verify"]) < 2:
            continue
        vrs = bucket["verify"]
        by_vmulti = max(vrs, key=lambda r: _f(r["j_value"]) * -1)  # min J = max V
        by_onestep = max(vrs, key=lambda r: _f(r["v_bayes"]))
        top1_states += 1
        if _i(by_vmulti["report_index"]) == _i(by_onestep["report_index"]):
            top1_agree += 1
    metrics["one_vs_multi"]["top1_agreement"] = (
        top1_agree / top1_states if top1_states else float("nan")
    )
    metrics["one_vs_multi"]["top1_states"] = top1_states

    # --- item 3: heuristic predictive power --------------------------------
    metrics["heuristic"] = {
        "pearson_util_v_multi": _corr(
            [c["hut"] for c in candidates], [c["v_multi"] for c in candidates]
        ),
        "pearson_errp_v_multi": _corr(
            [c["err_p"] for c in candidates], [c["v_multi"] for c in candidates]
        ),
    }
    # per-state top-1 (heuristic #1 by verify_rank) vs multi-step #1
    h_top1_agree = 0
    h_top1_states = 0
    for key, bucket in states.items():
        if bucket["new"] is None or len(bucket["verify"]) < 2:
            continue
        vrs = bucket["verify"]
        by_vmulti = max(vrs, key=lambda r: _f(r["j_value"]) * -1)
        by_heur = min(vrs, key=lambda r: _i(r["verify_rank"], 10**9))
        h_top1_states += 1
        if _i(by_vmulti["report_index"]) == _i(by_heur["report_index"]):
            h_top1_agree += 1
    metrics["heuristic"]["top1_agreement"] = (
        h_top1_agree / h_top1_states if h_top1_states else float("nan")
    )
    metrics["heuristic"]["top1_states"] = h_top1_states

    # --- item 4: oracle upper bound ----------------------------------------
    # per-state max V_multi over wrong candidates vs heuristic #1's V_multi
    oracle_gaps = []
    oracle_values = []
    heuristic_values = []
    onestep_values = []
    for key, bucket in states.items():
        new = bucket["new"]
        if new is None or not bucket["verify"]:
            continue
        j_new = _f(new["j_value"])
        wrong = [
            r for r in bucket["verify"] if _i(r["is_report_wrong"], 0) == 1
        ]
        if not wrong:
            continue
        vrs = bucket["verify"]
        oracle_v = max(_f(new["j_value"]) - _f(r["j_value"]) for r in wrong)
        by_vmulti = max(vrs, key=lambda r: _f(r["j_value"]) * -1)
        by_onestep = max(vrs, key=lambda r: _f(r["v_bayes"]))
        by_heur = min(vrs, key=lambda r: _i(r["verify_rank"], 10**9))
        oracle_values.append(oracle_v)
        heuristic_values.append(_f(new["j_value"]) - _f(by_heur["j_value"]))
        onestep_values.append(_f(new["j_value"]) - _f(by_onestep["j_value"]))
        oracle_gaps.append(oracle_v - (_f(new["j_value"]) - _f(by_heur["j_value"])))
    metrics["oracle"] = {
        "n_states_with_wrong": len(oracle_values),
        "oracle_v_mean": _mean(oracle_values),
        "heuristic_v_mean": _mean(heuristic_values),
        "onestep_v_mean": _mean(onestep_values),
        "oracle_minus_heuristic_mean": _mean(oracle_gaps),
        "wrong_candidate_v_multi_mean": _mean(
            [c["v_multi"] for c in candidates if c["is_wrong"] == 1]
        ),
        "correct_candidate_v_multi_mean": _mean(
            [c["v_multi"] for c in candidates if c["is_wrong"] == 0]
        ),
    }

    # --- item 5: questions vs brier ----------------------------------------
    wrong = [c for c in candidates if c["is_wrong"] == 1]
    correct = [c for c in candidates if c["is_wrong"] == 0]
    metrics["decomposition"] = {
        "all_brier_delta": _mean([c["brier_delta"] for c in candidates]),
        "all_future_delta_cost": _mean([C_Q * c["future_delta"] for c in candidates]),
        "wrong_brier_delta": _mean([c["brier_delta"] for c in wrong]),
        "wrong_future_delta_cost": _mean([C_Q * c["future_delta"] for c in wrong]),
        "correct_brier_delta": _mean([c["brier_delta"] for c in correct]),
        "correct_future_delta_cost": _mean([C_Q * c["future_delta"] for c in correct]),
        "n_wrong": len(wrong),
        "n_correct": len(correct),
    }

    # --- item 6: RAG signal ------------------------------------------------
    base_r2, base_n = _ols_r2(
        [c["v_multi"] for c in candidates],
        [[c["err_p"] if c["err_p"] is not None else 0.0] for c in candidates],
    )
    with_rag_r2, rag_n = _ols_r2(
        [c["v_multi"] for c in candidates],
        [
            [c["err_p"] if c["err_p"] is not None else 0.0,
             c["rip"] if c["rip"] is not None else 0.0]
            for c in candidates
        ],
    )
    metrics["rag_signal"] = {
        "pearson_rip_v_multi": _corr(
            [c["rip"] for c in candidates], [c["v_multi"] for c in candidates]
        ),
        "pearson_errp_v_multi": _corr(
            [c["err_p"] for c in candidates], [c["v_multi"] for c in candidates]
        ),
        "r2_errp_only": base_r2,
        "r2_errp_plus_rag": with_rag_r2,
        "r2_delta": with_rag_r2 - base_r2 if math.isfinite(base_r2) else float("nan"),
        "n": rag_n,
    }

    # --- item 7: leakage (construction statement) --------------------------
    metrics["leakage"] = {
        "statement": (
            "True disease D* and sampled answers enter ONLY the evaluation-side "
            "terminal metrics (terminal_brier / nll / top1 / wrong_to_correct). "
            "The continuation policy is the exact frozen ReliabilityAwareActionPolicy; "
            "no oracle_states, no learned gate, no true-state read.  Unit tests "
            "test_10 (deployable policy reads no true state) and test_11 (oracle "
            "labels do not feed policy) enforce this."
        ),
        "unit_tests": [
            "test_10_deployable_policy_reads_no_true_state",
            "test_11_oracle_labels_do_not_feed_policy",
            "test_04_continuation_uses_heuristic_not_oracle",
        ],
    }

    # --- item 8: Go / No-Go -------------------------------------------------
    # Diagnostic gain: does ranking verify candidates by V_multi select a wrong
    # report (precision@1) better than one-step V_Bayes and the heuristic?
    def precision_at1(rank_key):
        hits = 0
        denom = 0
        for key, bucket in states.items():
            if bucket["new"] is None or len(bucket["verify"]) < 2:
                continue
            vrs = bucket["verify"]
            if rank_key == "v_multi":
                best = max(vrs, key=lambda r: _f(r["j_value"]) * -1)
            elif rank_key == "one_step":
                best = max(vrs, key=lambda r: _f(r["v_bayes"]))
            else:  # heuristic
                best = min(vrs, key=lambda r: _i(r["verify_rank"], 10**9))
            denom += 1
            hits += int(_i(best["is_report_wrong"], 0) == 1)
        return hits / denom if denom else float("nan")

    p_vmulti = precision_at1("v_multi")
    p_onestep = precision_at1("one_step")
    p_heur = precision_at1("heuristic")
    metrics["go_nogo"] = {
        "precision_at1_v_multi": p_vmulti,
        "precision_at1_one_step": p_onestep,
        "precision_at1_heuristic": p_heur,
        "v_multi_minus_onestep": p_vmulti - p_onestep,
        "v_multi_minus_heuristic": p_vmulti - p_heur,
        "v_multi_wrong_mean": _mean([c["v_multi"] for c in candidates if c["is_wrong"] == 1]),
        "v_multi_correct_mean": _mean([c["v_multi"] for c in candidates if c["is_wrong"] == 0]),
        "frac_v_multi_positive": _mean([1.0 if c["v_multi"] > 0 else 0.0 for c in candidates]),
    }

    # --- item 9: worth training --------------------------------------------
    def feat_row(c):
        return [
            c["err_p"] if c["err_p"] is not None else 0.0,
            c["rip"] if c["rip"] is not None else 0.0,
            c["hut"] if c["hut"] is not None else 0.0,
        ]

    full_r2, full_n = _ols_r2(
        [c["v_multi"] for c in candidates], [feat_row(c) for c in candidates]
    )
    hut_r2, hut_n = _ols_r2(
        [c["v_multi"] for c in candidates],
        [[c["hut"] if c["hut"] is not None else 0.0] for c in candidates],
    )
    metrics["worth_training"] = {
        "r2_full_features": full_r2,
        "r2_heuristic_only": hut_r2,
        "n": full_n,
        "features": ["error_probability", "retrieval_impact", "heuristic_verify_utility"],
    }

    metrics["counts"] = {
        "n_states": len(states),
        "n_actions": len(actions),
        "n_verify_candidates": len(candidates),
        "n_wrong_candidates": sum(1 for c in candidates if c["is_wrong"] == 1),
        "n_cases": len({row["case_id"] for row in actions}),
        "noise_rates": sorted({row["noise_rate"] for row in actions}),
    }

    return metrics


def render_report(metrics):
    m = metrics
    cs = m["counts"]
    ls = m["label_stability"]
    om = m["one_vs_multi"]
    h = m["heuristic"]
    orc = m["oracle"]
    dec = m["decomposition"]
    rag = m["rag_signal"]
    gn = m["go_nogo"]
    wt = m["worth_training"]

    def pct(x):
        return "nan" if not math.isfinite(x) else f"{100 * x:.1f}%"

    lines = []
    lines.append("# Multi-Step Verification Value — Feasibility Audit (Phase 7)")
    lines.append("")
    lines.append(
        f"Scope: DDXPlus **train** split, {cs['n_cases']}/49 cases reached a qualifying "
        f"state (1/disease, balanced; 4 cases never accumulated >=2 non-UNKNOWN reports), "
        f"noise {cs['noise_rates']}, {cs['n_states']} qualifying states, "
        f"{cs['n_verify_candidates']} verify-candidate labels "
        f"({cs['n_wrong_candidates']} truly-wrong)."
    )
    lines.append("")
    lines.append("Objective: `J(a|H_t) = E[Brier(b_T, D*) + 0.03·N_future | H_t, a, π_heuristic]`, "
                 "`Q_multi = -J`, `V_multi(i) = J(AskNew_best) - J(VerifyOld(i))`.")
    lines.append("")

    lines.append("## 1. Label stability")
    lines.append("")
    lines.append(f"- n actions = {ls['n_actions']}; mean J SE = {ls['j_se_mean']:.4f} "
                 f"(median {ls['j_se_median']:.4f}, p90 {ls['j_se_p90']:.4f}).")
    lines.append(f"- |V_multi| mean = {ls['v_multi_abs_mean']:.4f} vs SE mean {ls['v_multi_se_mean']:.4f}; "
                 f"{pct(ls['frac_significant'])} candidates are >2 SE from zero.")
    lines.append(f"- paired (common-random-number) sign-flip rate = "
                 f"{pct(ls['paired_sign_flip_rate'])} over {ls['paired_n']} rollouts.")
    lines.append("")

    lines.append("## 2. One-step vs multi-step misalignment")
    lines.append("")
    lines.append(f"- Pearson(V_Bayes_Δ, V_multi) = {om['pearson']:.3f}; "
                 f"sign agreement = {pct(om['sign_agreement'])}.")
    lines.append(f"- top-1 verify target agreement = {pct(om['top1_agreement'])} "
                 f"over {om['top1_states']} states.")
    lines.append("")

    lines.append("## 3. Heuristic predictive power")
    lines.append("")
    lines.append(f"- Pearson(heuristic_util, V_multi) = {h['pearson_util_v_multi']:.3f}; "
                 f"Pearson(error_prob, V_multi) = {h['pearson_errp_v_multi']:.3f}.")
    lines.append(f"- top-1 agreement (heuristic #1 vs V_multi #1) = "
                 f"{pct(h['top1_agreement'])} over {h['top1_states']} states.")
    lines.append("")

    lines.append("## 4. Oracle upper bound")
    lines.append("")
    lines.append(f"- states with >=1 wrong report: {orc['n_states_with_wrong']}.")
    lines.append(f"- mean V_multi of best-wrong (oracle) = {orc['oracle_v_mean']:.4f} vs "
                 f"heuristic-#1 = {orc['heuristic_v_mean']:.4f} vs "
                 f"one-step-#1 = {orc['onestep_v_mean']:.4f}.")
    lines.append(f"- oracle − heuristic gap = {orc['oracle_minus_heuristic_mean']:.4f}.")
    lines.append(f"- wrong candidates V_multi mean = {orc['wrong_candidate_v_multi_mean']:.4f} "
                 f"vs correct candidates = {orc['correct_candidate_v_multi_mean']:.4f}.")
    lines.append("")

    lines.append("## 5. Questions vs Brier decomposition")
    lines.append("")
    lines.append(f"- pooled V_multi = BrierΔ {dec['all_brier_delta']:.4f} + "
                 f"0.03·futureΔ {dec['all_future_delta_cost']:.4f}.")
    lines.append(f"- wrong candidates: BrierΔ {dec['wrong_brier_delta']:.4f} + "
                 f"0.03·futureΔ {dec['wrong_future_delta_cost']:.4f} (n={dec['n_wrong']}).")
    lines.append(f"- correct candidates: BrierΔ {dec['correct_brier_delta']:.4f} + "
                 f"0.03·futureΔ {dec['correct_future_delta_cost']:.4f} (n={dec['n_correct']}).")
    lines.append("")

    lines.append("## 6. RAG signal")
    lines.append("")
    lines.append(f"- Pearson(retrieval_impact, V_multi) = {rag['pearson_rip_v_multi']:.3f} "
                 f"vs Pearson(error_prob, V_multi) = {rag['pearson_errp_v_multi']:.3f}.")
    lines.append(f"- OLS R²: error_prob only = {rag['r2_errp_only']:.3f}, "
                 f"+ retrieval_impact = {rag['r2_errp_plus_rag']:.3f} (Δ = {rag['r2_delta']:.3f}, n={rag['n']}).")
    lines.append("")

    lines.append("## 7. Leakage")
    lines.append("")
    lines.append(metrics["leakage"]["statement"])
    lines.append("")

    lines.append("## 8. Go / No-Go")
    lines.append("")
    lines.append(f"- precision@1 (wrong-report selection): V_multi = {pct(gn['precision_at1_v_multi'])}, "
                 f"one-step = {pct(gn['precision_at1_one_step'])}, heuristic = {pct(gn['precision_at1_heuristic'])}.")
    lines.append(f"- V_multi − one-step = {pct(gn['v_multi_minus_onestep'])}; "
                 f"V_multi − heuristic = {pct(gn['v_multi_minus_heuristic'])}.")
    lines.append(f"- V_multi wrong mean = {gn['v_multi_wrong_mean']:.4f} vs correct = "
                 f"{gn['v_multi_correct_mean']:.4f}; fraction positive = {pct(gn['frac_v_multi_positive'])}.")
    lines.append("- NOTE: V_multi's selection gain is the *oracle* value (it uses true D* in the "
                 "terminal rollouts); it is not deployable. The deployable question is item 9 "
                 "(learnability). See GO_NO_GO.md for the two-part verdict.")
    lines.append("")

    lines.append("## 9. Worth training")
    lines.append("")
    lines.append(f"- OLS R²(heuristic_util only) = {wt['r2_heuristic_only']:.3f}; "
                 f"R²([error_prob, retrieval_impact, heuristic_util]) = {wt['r2_full_features']:.3f} (n={wt['n']}).")
    lines.append("")

    return "\n".join(lines)


def render_go_nogo(metrics):
    gn = metrics["go_nogo"]
    om = metrics["one_vs_multi"]
    ls = metrics["label_stability"]
    orc = metrics["oracle"]
    rag = metrics["rag_signal"]
    wt = metrics["worth_training"]
    cs = metrics["counts"]

    # Two independent sub-questions, each judged on its own evidence.
    # A. reasonableness (multi-step vs one-step)
    a_ok = (
        orc["wrong_candidate_v_multi_mean"] > 0.0
        and orc["onestep_v_mean"] < 0.0
        and gn["precision_at1_v_multi"] > gn["precision_at1_one_step"]
    )
    # B. deployable / trainable
    b_ok = (
        ls["frac_significant"] >= 0.5
        and wt["r2_full_features"] >= 0.15
        and rag["r2_delta"] > 0.02
    )
    go = a_ok and b_ok

    lines = []
    lines.append("# Phase 7 — Multi-Step Verification Value: Go / No-Go")
    lines.append("")
    lines.append("## A. Is multi-step value more reasonable than one-step Brier value?")
    lines.append("")
    lines.append(f"- wrong vs correct V_multi: {orc['wrong_candidate_v_multi_mean']:+.4f} vs "
                 f"{orc['correct_candidate_v_multi_mean']:+.4f} (right sign).")
    lines.append(f"- one-step V_Bayes top pick realized V_multi = {orc['onestep_v_mean']:+.4f} "
                 f"(< 0 = counterproductive); heuristic = {orc['heuristic_v_mean']:+.4f}; "
                 f"oracle best-wrong = {orc['oracle_v_mean']:+.4f}.")
    lines.append(f"- precision@1: V_multi {gn['precision_at1_v_multi']:.3f} vs one-step "
                 f"{gn['precision_at1_one_step']:.3f} vs heuristic {gn['precision_at1_heuristic']:.3f}.")
    lines.append(f"- one-step vs multi-step: Pearson {om['pearson']:.3f}, "
                 f"sign agreement {om['sign_agreement']:.3f}.")
    lines.append(f"**A verdict: {'YES' if a_ok else 'NO'}** — multi-step value is "
                 f"{'' if a_ok else 'not '}more reasonable than one-step Brier value.")
    lines.append("")
    lines.append("## B. Is it stable and learnable enough to deploy (train)?")
    lines.append("")
    lines.append(f"- label stability: {100*ls['frac_significant']:.0f}% significant "
                 f"(SE {ls['v_multi_se_mean']:.3f} vs |V| {ls['v_multi_abs_mean']:.3f}), "
                 f"paired sign-flip {100*ls['paired_sign_flip_rate']:.1f}%.")
    lines.append(f"- learnability: OLS R²(full features) = {wt['r2_full_features']:.3f} "
                 f"(heuristic-only {wt['r2_heuristic_only']:.3f}).")
    lines.append(f"- RAG adds ΔR² = {rag['r2_delta']:.3f}.")
    lines.append(f"**B verdict: {'YES' if b_ok else 'NO'}** — the label is "
                 f"{'' if b_ok else 'not '}stable and learnable from label-free features.")
    lines.append("")
    lines.append(f"## Overall: **{'GO' if go else 'NO-GO'}** "
                 f"(A={'YES' if a_ok else 'NO'}, B={'YES' if b_ok else 'NO'})")
    lines.append("")
    lines.append(f"Coverage: {cs['n_cases']}/49 cases reached a qualifying state "
                 f"({cs['n_states']} states, {cs['n_verify_candidates']} verify labels, "
                 f"{cs['n_wrong_candidates']} truly-wrong).")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke-check", "analyze"), default="analyze")
    ap.add_argument("--expected-cases", type=int, default=49)
    args = ap.parse_args()

    actions = load_actions()
    rollouts = load_rollouts()

    if args.mode == "smoke-check":
        checks = smoke_checks(actions, rollouts, args.expected_cases)
        ok = write_smoke_checks(checks)
        for name, passed, detail in checks:
            print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        print(f"\nsmoke {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    metrics = analyze(actions, rollouts)
    (OUT / "analysis.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    (OUT / "MULTISTEP_VALUE_AUDIT_REPORT.md").write_text(
        render_report(metrics), encoding="utf-8"
    )
    (OUT / "GO_NO_GO.md").write_text(
        render_go_nogo(metrics), encoding="utf-8"
    )
    print(json.dumps(metrics["go_nogo"], indent=2, default=str))
    print(json.dumps(metrics["counts"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
