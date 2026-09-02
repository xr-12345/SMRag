"""Phase 3A analysis.

Reads ``action_value_samples.csv`` (one row per state-action, 7 comparison
scores) and writes the six analysis products required by CONFIG.json:

    value_correlations.csv        Spearman/Kendall + cluster-bootstrap CI
    value_calibration.csv         V_Bayes binned vs E[V_real]
    pairwise_action_accuracy.csv  AskNew-vs-VerifyOld choice accuracy
    action_regret.csv             top-action regret per policy
    selected_action_returns.csv   realized returns of each policy's chosen action
    counterexamples.csv           representative failure modes
    cost_sensitivity.csv          V_Bayes / V_real direction across C in {0,.03,.06}

All statistics are label-free on the *deployable* side; V_real is the
evaluation-only target.  Nothing here is trained; nothing is committed.
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent
SAMPLE = OUT / "action_value_samples.csv"

BASE_COST = 0.03  # the cost at which the audit was run

# --- load / clean ---------------------------------------------------------- #


def load_rows() -> list[dict]:
    with SAMPLE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --- correlation helpers --------------------------------------------------- #


def spearman(x, y) -> tuple[float, float] | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def kendall(x, y) -> tuple[float, float] | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    tau, p = stats.kendalltau(x, y)
    return float(tau), float(p)


def cluster_bootstrap(
    rows: list[dict],
    cluster_key: str,
    statistic,
    n_boot: int = 1000,
    seed: int = 3031,
) -> dict:
    """Cluster (resample whole clusters) bootstrap of a per-row statistic.

    Returns {mean, lo, hi, n_boot} where lo/hi are the 2.5/97.5 percentiles.
    ``statistic(rows_subset) -> float`` (may return None).
    """
    rng = random.Random(seed)
    clusters: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        clusters[row[cluster_key]].append(row)
    keys = list(clusters.keys())
    if not keys:
        return {"mean": None, "lo": None, "hi": None, "n_boot": 0}
    values = []
    for _ in range(n_boot):
        chosen = rng.choices(keys, k=len(keys))
        subset = [row for key in chosen for row in clusters[key]]
        value = statistic(subset)
        if value is not None:
            values.append(value)
    if not values:
        return {"mean": None, "lo": None, "hi": None, "n_boot": 0}
    values.sort()
    lo = values[int(0.025 * (len(values) - 1))]
    hi = values[int(0.975 * (len(values) - 1))]
    return {"mean": sum(values) / len(values), "lo": lo, "hi": hi, "n_boot": len(values)}


def _corr_statistic(score_key: str, corr_fn):
    def statistic(rows):
        xs = [num(r, score_key) for r in rows]
        ys = [num(r, "v_real") for r in rows]
        paired = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
        if len(paired) < 3:
            return None
        px, py = zip(*paired)
        out = corr_fn(list(px), list(py))
        return out[0] if out is not None else None

    return statistic


# --- policies -------------------------------------------------------------- #


def _best(rows: list[dict], key: str):
    """argmax row by score key among rows with a non-None score."""
    best = None
    for row in rows:
        value = num(row, key)
        if value is None:
            continue
        if best is None or value > num(best, key):
            best = row
    return best


def policy_choice2(rows: list[dict], policy: str) -> dict | None:
    """Joint choice over new+verify on a single comparable score."""
    if policy == "eig":
        return _best([r for r in rows if r["action_kind"] == "new"], "eig")
    if policy == "v_bayes":
        return _best(rows, "v_bayes")
    if policy == "heuristic":
        return _best(rows, "_heuristic_util")
    if policy == "retrieval_joint_gate":
        return _best(rows, "_joint_util")
    raise ValueError(policy)


def add_derived(rows: list[dict]) -> list[dict]:
    """Attach per-row composite scores used by the joint-choice policies."""
    for row in rows:
        eig = num(row, "eig")
        hvu = num(row, "heuristic_verify_utility")
        epti = num(row, "error_prob_times_impact")
        row["_heuristic_util"] = hvu if row["action_kind"] == "verify" else eig
        row["_joint_util"] = epti if row["action_kind"] == "verify" else eig
    return rows


# --- cost sensitivity ------------------------------------------------------ #


def shifted_cost(row: dict, new_cost: float, key: str) -> float | None:
    """V(key) recomputed at ``new_cost`` from the gross reduction.

    V = gross_reduction - C, so V(C') = V(C) + (C - C').  For V_real the gross
    reduction is ``gross_brier_reduction``; for V_bayes the reduction is
    ``v_bayes + BASE_COST`` (the risk unit differs but the additive cost shift
    is exact in either unit).
    """
    if key == "v_real":
        gross = num(row, "gross_brier_reduction")
        if gross is None:
            return None
        return gross - new_cost
    if key == "v_bayes":
        v = num(row, "v_bayes")
        if v is None:
            return None
        return v + BASE_COST - new_cost
    raise ValueError(key)


# --- main ------------------------------------------------------------------ #


def correlations(rows: list[dict]) -> list[dict]:
    score_scopes = [
        ("eig", "AskNew"),
        ("v_bayes", "AskNew"),
        ("retrospective_error_prob", "VerifyOld"),
        ("retrieval_impact", "VerifyOld"),
        ("heuristic_verify_utility", "VerifyOld"),
        ("error_prob_times_impact", "VerifyOld"),
        ("v_bayes", "VerifyOld"),
        ("v_bayes", "both"),
    ]
    out = []
    scope_action = {"AskNew": "new", "VerifyOld": "verify"}
    for score_key, scope in score_scopes:
        if scope == "both":
            subset = rows
        else:
            subset = [r for r in rows if r["action_kind"] == scope_action[scope]]
        for corr_name, corr_fn in (("spearman", spearman), ("kendall", kendall)):
            point = _corr_statistic(score_key, corr_fn)(subset)
            boot = cluster_bootstrap(subset, "case_id", _corr_statistic(score_key, corr_fn))
            out.append(
                {
                    "score": score_key,
                    "scope": scope,
                    "metric": corr_name,
                    "point": _fmt(point),
                    "bootstrap_mean": _fmt(boot["mean"]),
                    "ci_low": _fmt(boot["lo"]),
                    "ci_high": _fmt(boot["hi"]),
                    "n": len(subset),
                    "n_boot": boot["n_boot"],
                }
            )
    return out


def calibration(rows: list[dict]) -> list[dict]:
    both = [r for r in rows if num(r, "v_bayes") is not None and num(r, "v_real") is not None]
    both.sort(key=lambda r: num(r, "v_bayes"))
    n_bins = 10
    bins: list[list[dict]] = [[] for _ in range(n_bins)]
    edges = [num(r, "v_bayes") for r in both]
    lo, hi = min(edges), max(edges)
    span = (hi - lo) or 1.0
    for r in both:
        v = num(r, "v_bayes")
        idx = int((v - lo) / span * n_bins)
        idx = min(idx, n_bins - 1)
        bins[idx].append(r)
    out = []
    for idx, group in enumerate(bins):
        if not group:
            continue
        v_bayes = [num(r, "v_bayes") for r in group]
        v_real = [num(r, "v_real") for r in group]
        out.append(
            {
                "bin": idx,
                "v_bayes_mean": _fmt(sum(v_bayes) / len(v_bayes)),
                "v_real_mean": _fmt(sum(v_real) / len(v_real)),
                "v_real_se": _fmt(stats.sem(v_real) if len(v_real) > 1 else 0.0),
                "n": len(group),
                "frac_v_real_positive": _fmt(sum(1 for v in v_real if v > 0) / len(v_real)),
            }
        )
    return out


def pairwise_accuracy(rows: list[dict]) -> list[dict]:
    states = defaultdict(list)
    for row in rows:
        states[(row["case_id"], row["noise_rate"], row["state_index"])].append(row)
    out = []
    for policy in ("v_bayes", "heuristic", "retrieval_joint_gate"):
        agree = 0
        total = 0
        new_to_verify = 0
        verify_to_new = 0
        for _, group in states.items():
            new_rows = [r for r in group if r["action_kind"] == "new"]
            verify_rows = [r for r in group if r["action_kind"] == "verify"]
            if not new_rows or not verify_rows:
                continue
            # oracle choice (V_real) between best-new and best-verify
            best_new_real = _best(new_rows, "v_real")
            best_verify_real = _best(verify_rows, "v_real")
            oracle_new = num(best_new_real, "v_real") if best_new_real else None
            oracle_verify = num(best_verify_real, "v_real") if best_verify_real else None
            if oracle_new is None or oracle_verify is None:
                continue
            oracle_verify_kind = oracle_verify > oracle_new  # True -> verify
            # policy choice
            if policy == "v_bayes":
                bn = _best(new_rows, "v_bayes")
                bv = _best(verify_rows, "v_bayes")
                pn = num(bn, "v_bayes") if bn else None
                pv = num(bv, "v_bayes") if bv else None
                policy_verify_kind = (pv is not None and pn is not None and pv > pn)
                if pn is None or pv is None:
                    continue
            elif policy == "heuristic":
                bn = _best(new_rows, "eig")
                bv = _best(verify_rows, "heuristic_verify_utility")
                pn = num(bn, "eig") if bn else None
                pv = num(bv, "heuristic_verify_utility") if bv else None
                if pn is None or pv is None:
                    continue
                policy_verify_kind = pv > pn
            else:  # retrieval_joint_gate
                bn = _best(new_rows, "eig")
                bv = _best(verify_rows, "error_prob_times_impact")
                pn = num(bn, "eig") if bn else None
                pv = num(bv, "error_prob_times_impact") if bv else None
                if pn is None or pv is None:
                    continue
                policy_verify_kind = pv > pn
            total += 1
            if policy_verify_kind == oracle_verify_kind:
                agree += 1
            elif policy_verify_kind and not oracle_verify_kind:
                verify_to_new += 1  # policy verifies, oracle asks new
            else:
                new_to_verify += 1  # policy asks new, oracle verifies
        out.append(
            {
                "policy": policy,
                "n_states": total,
                "accuracy": _fmt(agree / total if total else None),
                "asknew_to_verifyold_miscount": new_to_verify,
                "verifyold_to_asknew_miscount": verify_to_new,
            }
        )
    return out


def action_regret(rows: list[dict]) -> list[dict]:
    states = defaultdict(list)
    for row in rows:
        states[(row["case_id"], row["noise_rate"], row["state_index"])].append(row)
    policies = ("eig", "heuristic", "retrieval_joint_gate", "v_bayes")
    out = []
    for policy in policies:
        regrets = []
        for _, group in states.items():
            real_values = [num(r, "v_real") for r in group if num(r, "v_real") is not None]
            if not real_values:
                continue
            max_real = max(real_values)
            chosen = policy_choice2(group, policy)
            if chosen is None:
                continue
            chosen_real = num(chosen, "v_real")
            if chosen_real is None:
                continue
            regrets.append(max_real - chosen_real)
        out.append(
            {
                "policy": policy,
                "n_states": len(regrets),
                "mean_regret": _fmt(sum(regrets) / len(regrets) if regrets else None),
                "median_regret": _fmt(statistics_median(regrets) if regrets else None),
                "frac_zero_regret": _fmt(sum(1 for r in regrets if r < 1e-9) / len(regrets) if regrets else None),
                "max_regret": _fmt(max(regrets) if regrets else None),
            }
        )
    return out


def selected_action_returns(rows: list[dict]) -> list[dict]:
    states = defaultdict(list)
    for row in rows:
        states[(row["case_id"], row["noise_rate"], row["state_index"])].append(row)
    out = []
    for policy in ("eig", "heuristic", "retrieval_joint_gate", "v_bayes"):
        brier = []
        nll = []
        wrong_to_correct = []
        correct_to_wrong = []
        net = []
        for _, group in states.items():
            chosen = policy_choice2(group, policy)
            if chosen is None:
                continue
            b = num(chosen, "gross_brier_reduction")
            n = num(chosen, "gross_nll_reduction")
            v = num(chosen, "v_real")
            w = num(chosen, "wrong_to_correct")
            c = num(chosen, "correct_to_wrong")
            if b is not None:
                brier.append(b)
            if n is not None:
                nll.append(n)
            if v is not None:
                net.append(v)
            if w is not None:
                wrong_to_correct.append(w)
            if c is not None:
                correct_to_wrong.append(c)
        out.append(
            {
                "policy": policy,
                "n_states": len(net),
                "mean_gross_brier_reduction": _fmt(mean(brier)),
                "mean_gross_nll_reduction": _fmt(mean(nll)),
                "mean_net_realized_return": _fmt(mean(net)),
                "mean_wrong_to_correct": _fmt(mean(wrong_to_correct)),
                "mean_correct_to_wrong": _fmt(mean(correct_to_wrong)),
            }
        )
    return out


def counterexamples(rows: list[dict]) -> list[dict]:
    out = []
    # 1. high error-prob but V_real ~ 0 (verify rows)
    verify = [r for r in rows if r["action_kind"] == "verify"]
    verify_sorted = sorted(
        [r for r in verify if num(r, "retrospective_error_prob") is not None and num(r, "v_real") is not None],
        key=lambda r: -num(r, "retrospective_error_prob"),
    )
    for r in verify_sorted[:5]:
        out.append(_counterexample_row("high_error_prob_low_vreal", r))
    # 2. high retrieval-impact but V_real ~ 0
    imp_sorted = sorted(
        [r for r in verify if num(r, "retrieval_impact") is not None and num(r, "v_real") is not None],
        key=lambda r: -num(r, "retrieval_impact"),
    )
    for r in imp_sorted[:5]:
        out.append(_counterexample_row("high_impact_low_vreal", r))
    # 3. medium error-prob but high V_real
    med = [r for r in verify if num(r, "retrospective_error_prob") is not None and num(r, "v_real") is not None]
    med = [r for r in med if num(r, "retrospective_error_prob") > 0.03]
    med_sorted = sorted(med, key=lambda r: -num(r, "v_real"))
    for r in med_sorted[:5]:
        out.append(_counterexample_row("medium_error_prob_high_vreal", r))
    # 4. V_Bayes and V_real sign disagree (both action kinds)
    disagree = [
        r for r in rows
        if num(r, "v_bayes") is not None and num(r, "v_real") is not None
        and (num(r, "v_bayes") > 0) != (num(r, "v_real") > 0)
    ]
    disagree = sorted(disagree, key=lambda r: -abs(num(r, "v_bayes") - num(r, "v_real")))
    for r in disagree[:5]:
        out.append(_counterexample_row("sign_disagreement", r))
    # 5/6 require state-level context (V_real favors verify but EIG/heuristic picks new, etc.)
    states = defaultdict(list)
    for row in rows:
        states[(row["case_id"], row["noise_rate"], row["state_index"])].append(row)
    cat5 = []  # V_real favors VerifyOld but EIG (always) picks AskNew
    cat6 = []  # V_real favors AskNew but heuristic picks VerifyOld
    for _, group in states.items():
        new_rows = [r for r in group if r["action_kind"] == "new"]
        verify_rows = [r for r in group if r["action_kind"] == "verify"]
        if not new_rows or not verify_rows:
            continue
        best_new_real = _best(new_rows, "v_real")
        best_verify_real = _best(verify_rows, "v_real")
        if best_new_real is None or best_verify_real is None:
            continue
        gap = num(best_verify_real, "v_real") - num(best_new_real, "v_real")
        if gap > 0:
            cat5.append((gap, best_verify_real))
        else:
            bv = _best(verify_rows, "heuristic_verify_utility")
            bn = _best(new_rows, "eig")
            if bv is not None and bn is not None and num(bv, "heuristic_verify_utility") > num(bn, "eig"):
                cat6.append((-gap, bn))
    cat5.sort(key=lambda t: -t[0])
    cat6.sort(key=lambda t: -t[0])
    for _, row in cat5[:10]:
        out.append(_counterexample_row("eig_picks_new_but_real_favors_verify", row))
    for _, row in cat6[:10]:
        out.append(_counterexample_row("heuristic_picks_verify_but_real_favors_new", row))
    return out


def cost_sensitivity(rows: list[dict]) -> list[dict]:
    both = [r for r in rows if num(r, "v_bayes") is not None and num(r, "v_real") is not None]
    out = []
    for key in ("v_bayes", "v_real"):
        for cost in (0.00, 0.03, 0.06):
            values = [shifted_cost(r, cost, key) for r in both]
            values = [v for v in values if v is not None]
            out.append(
                {
                    "quantity": key,
                    "cost": cost,
                    "n": len(values),
                    "mean": _fmt(mean(values)),
                    "frac_positive": _fmt(sum(1 for v in values if v > 0) / len(values) if values else None),
                }
            )
    return out


# --- small utilities ------------------------------------------------------- #


def _fmt(value) -> float | str:
    if value is None:
        return ""
    return round(float(value), 6)


def mean(values):
    return sum(values) / len(values) if values else None


def statistics_median(values):
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    if n % 2:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / 2


def _counterexample_row(kind: str, row: dict) -> dict:
    return {
        "kind": kind,
        "case_id": row["case_id"],
        "diagnosis": row["diagnosis"],
        "noise_rate": row["noise_rate"],
        "state_kind": row["state_kind"],
        "action_kind": row["action_kind"],
        "evidence_key": row["evidence_key"],
        "retrospective_error_prob": num(row, "retrospective_error_prob"),
        "retrieval_impact": num(row, "retrieval_impact"),
        "eig": num(row, "eig"),
        "v_bayes": num(row, "v_bayes"),
        "v_real": num(row, "v_real"),
        "gross_brier_reduction": num(row, "gross_brier_reduction"),
        "gross_nll_reduction": num(row, "gross_nll_reduction"),
    }


def write_csv(rows: list[dict], name: str):
    path = OUT / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = add_derived(load_rows())
    print(f"loaded {len(rows)} sample rows")

    products = [
        ("value_correlations.csv", correlations(rows)),
        ("value_calibration.csv", calibration(rows)),
        ("pairwise_action_accuracy.csv", pairwise_accuracy(rows)),
        ("action_regret.csv", action_regret(rows)),
        ("selected_action_returns.csv", selected_action_returns(rows)),
        ("counterexamples.csv", counterexamples(rows)),
        ("cost_sensitivity.csv", cost_sensitivity(rows)),
    ]
    for name, product in products:
        write_csv(product, name)
        print(f"wrote {name}: {len(product)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
