"""Phase 4 -- evaluate the frozen learned worthiness model against 8 baselines.

Reads ``validation_features.csv`` + the frozen models, computes the 8 scores,
and writes the evaluation products (value / benefit / harm / matched-budget /
regret / rag-ablation / bootstrap / calibration / feature-importance /
counterexamples).  Validation is used *only* for evaluation -- no retraining.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy import stats

from powerful_medrag import verification_worthiness as vw

OUT = Path(__file__).resolve().parent
MODELS = OUT / "models"

METHODS = (
    "heuristic_verify_utility",
    "retrospective_error_probability",
    "retrieval_impact",
    "error_prob_times_impact",
    "v_bayes",
    "learned_norag",
    "learned_real",
    "learned_shuffled",
)
LEARNED = ("learned_norag", "learned_real", "learned_shuffled")


def load_rows():
    return list(csv.DictReader(open(OUT / "validation_features.csv", newline="", encoding="utf-8")))


def _state_key(r):
    return (r["case_id"], r["noise_rate"], r["state_index"])


def compute_scores(rows):
    scores = {m: np.full(len(rows), np.nan) for m in METHODS}
    h_hat = np.full(len(rows), np.nan)
    allowed = np.zeros(len(rows), dtype=bool)
    allowed_noharm = np.zeros(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        for m in METHODS:
            if m in LEARNED:
                continue
            scores[m][i] = vw.baseline_score(m, r)
    # learned models
    models = {}
    for rag_mode in ("real", "none", "shuffled"):
        models[rag_mode] = joblib.load(MODELS / f"worthiness_{rag_mode}.pkl")
    for rag_mode, col in (("real", "learned_real"), ("none", "learned_norag"),
                          ("shuffled", "learned_shuffled")):
        X = vw.build_feature_matrix(rows, rag_mode=rag_mode)
        mdl = models[rag_mode]
        g_hat, h = mdl.predict(X)
        scores[col] = g_hat - vw.C_VERIFY
        if rag_mode == "real":
            h_hat = h
            allowed = mdl.allowed(X)
            allowed_noharm = (g_hat - vw.C_VERIFY > 0.0)
    return scores, h_hat, allowed, allowed_noharm, models


def cluster_bootstrap_spearman(rows, scores, metric_col, n_boot=500, seed=2027):
    """Case-cluster bootstrap CI for Spearman(score, metric)."""
    by_case = defaultdict(list)
    for i, r in enumerate(rows):
        by_case[r["case_id"]].append(i)
    case_ids = list(by_case)
    rng = np.random.default_rng(seed)
    y = np.asarray([float(r[metric_col]) for r in rows])
    rhos = []
    for _ in range(n_boot):
        sample_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
        idx = [i for cid in sample_ids for i in by_case[cid]]
        x = scores[idx]
        yy = y[idx]
        if np.unique(x).size < 2 or np.unique(yy).size < 2:
            continue
        rhos.append(float(stats.spearmanr(x, yy).statistic))
    rhos = np.asarray(rhos)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(np.mean(rhos)), float(lo), float(hi)


def cluster_bootstrap_spearman_diff(rows, scores, a, b, n_boot=500, seed=2027):
    """Case-cluster bootstrap CI for Spearman(a) - Spearman(b)."""
    by_case = defaultdict(list)
    for i, r in enumerate(rows):
        by_case[r["case_id"]].append(i)
    case_ids = list(by_case)
    rng = np.random.default_rng(seed)
    y = np.asarray([float(r["v_real"]) for r in rows])
    diffs = []
    for _ in range(n_boot):
        sample_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
        idx = [i for cid in sample_ids for i in by_case[cid]]
        xa, xb, yy = scores[a][idx], scores[b][idx], y[idx]
        if (np.unique(xa).size < 2 or np.unique(xb).size < 2
                or np.unique(yy).size < 2):
            continue
        ra = float(stats.spearmanr(xa, yy).statistic)
        rb = float(stats.spearmanr(xb, yy).statistic)
        diffs.append(ra - rb)
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def main() -> int:
    rows = load_rows()
    print(f"validation rows: {len(rows)}")
    scores, h_hat, allowed, allowed_noharm, models = compute_scores(rows)
    value = np.asarray([float(r["v_real"]) for r in rows])
    gain = np.asarray([float(r["gross_brier_reduction"]) for r in rows])
    ctw = np.asarray([float(r["correct_to_wrong"]) for r in rows])
    groups = np.asarray([r["case_id"] for r in rows])

    # ---- validation_predictions.csv ------------------------------------- #
    with (OUT / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["case_id", "diagnosis", "noise_rate", "state_kind", "state_index",
                "turn_index"] + list(METHODS) + ["h_hat", "allowed",
                "v_real", "gross_brier_reduction", "correct_to_wrong"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows):
            row = {c: r.get(c, "") for c in ("case_id", "diagnosis", "noise_rate",
                                             "state_kind", "state_index", "turn_index")}
            for m in METHODS:
                row[m] = round(float(scores[m][i]), 10)
            row["h_hat"] = round(float(h_hat[i]), 10) if not np.isnan(h_hat[i]) else ""
            row["allowed"] = int(bool(allowed[i]))
            row["v_real"] = r["v_real"]
            row["gross_brier_reduction"] = r["gross_brier_reduction"]
            row["correct_to_wrong"] = r["correct_to_wrong"]
            w.writerow(row)

    # ---- value_metrics.csv (Spearman + Kendall + bootstrap CI) ----------- #
    value_rows = []
    for m in METHODS:
        x = scores[m]
        valid = ~np.isnan(x)
        rho = float(stats.spearmanr(x[valid], value[valid]).statistic)
        tau = float(stats.kendalltau(x[valid], value[valid]).statistic)
        rho_mean, lo, hi = cluster_bootstrap_spearman(rows, scores[m], "v_real")
        value_rows.append({"method": m, "n": int(valid.sum()),
                           "spearman": round(rho, 4), "spearman_ci_lo": round(lo, 4),
                           "spearman_ci_hi": round(hi, 4), "kendall": round(tau, 4)})
    write_dicts(value_rows, OUT / "value_metrics.csv")

    # ---- matched-budget (learned real's natural trigger count) ----------- #
    n_trigger = int(allowed.sum())
    print(f"learned real natural triggers: {n_trigger}")
    budget_rows = []
    for m in METHODS:
        if m in LEARNED and m != "learned_real":
            continue
        if m == "learned_real":
            mask = allowed
        else:
            order = np.argsort(-scores[m])
            mask = np.zeros(len(rows), dtype=bool)
            mask[order[:n_trigger]] = True
        sel_gain = gain[mask]
        sel_value = value[mask]
        sel_ctw = ctw[mask]
        budget_rows.append({
            "method": m, "triggers": int(mask.sum()),
            "mean_net_brier": round(float(sel_value.mean()), 5),
            "mean_gross_brier": round(float(sel_gain.mean()), 5),
            "mean_correct_to_wrong": round(float(sel_ctw.mean()), 5),
            "frac_value_gt0": round(float((sel_value > 0).mean()), 5),
            "sum_net_brier": round(float(sel_value.sum()), 5),
        })
    write_dicts(budget_rows, OUT / "matched_budget_results.csv")

    # ---- benefit + harm metrics (matched budget) ------------------------- #
    write_dicts(
        [{k: r[k] for k in ("method", "triggers", "mean_net_brier", "frac_value_gt0")}
         for r in budget_rows],
        OUT / "benefit_metrics.csv",
    )
    # harm metrics: matched-budget correct->wrong + harm-head AUC + gate effect
    harm_labels = np.asarray([vw.derive_labels(r)["harm"] for r in rows])
    harm_auc = vw._roc_auc(harm_labels, h_hat)
    with_gate_ctw = float(ctw[allowed].mean()) if allowed.sum() else float("nan")
    noharm_ctw = float(ctw[allowed_noharm].mean()) if allowed_noharm.sum() else float("nan")
    harm_rows = [
        {"metric": "harm_head_validation_auc", "value": round(harm_auc, 4)},
        {"metric": "with_gate_triggers", "value": int(allowed.sum())},
        {"metric": "without_gate_triggers", "value": int(allowed_noharm.sum())},
        {"metric": "with_gate_mean_correct_to_wrong", "value": round(with_gate_ctw, 5)},
        {"metric": "without_gate_mean_correct_to_wrong", "value": round(noharm_ctw, 5)},
        {"metric": "harm_gate_ctw_reduction", "value": round(noharm_ctw - with_gate_ctw, 5)},
    ]
    for r in budget_rows:
        harm_rows.append({"metric": f"{r['method']}_mean_correct_to_wrong",
                          "value": r["mean_correct_to_wrong"]})
    write_dicts(harm_rows, OUT / "harm_metrics.csv")

    # ---- action regret (top-1 selection, per state) ---------------------- #
    regret_rows = []
    states = defaultdict(list)
    for i, r in enumerate(rows):
        states[_state_key(r)].append(i)
    for m in METHODS:
        regrets = []
        for idxs in states.values():
            if len(idxs) < 2:
                continue
            vals = value[idxs]
            s = scores[m][idxs]
            if np.any(np.isnan(s)):
                continue
            best = vals.max()
            top = int(np.argmax(s))  # index into the state's subset
            regrets.append(best - vals[top])
        regret_rows.append({"method": m, "n_states": len(regrets),
                            "mean_regret": round(float(np.mean(regrets)), 5),
                            "median_regret": round(float(np.median(regrets)), 5),
                            "frac_zero_regret": round(float(np.mean(np.asarray(regrets) == 0)), 5)})
    write_dicts(regret_rows, OUT / "action_regret.csv")

    # ---- rag ablation (Spearman + matched-budget net Brier) -------------- #
    rag_rows = []
    for m in ("learned_norag", "learned_real", "learned_shuffled"):
        x = scores[m]
        rho = float(stats.spearmanr(x, value).statistic)
        rho_mean, lo, hi = cluster_bootstrap_spearman(rows, scores[m], "v_real")
        # matched budget to real's count
        order = np.argsort(-x)
        mask = np.zeros(len(rows), dtype=bool)
        mask[order[:n_trigger]] = True
        rag_rows.append({"method": m, "spearman": round(rho, 4),
                         "spearman_ci_lo": round(lo, 4), "spearman_ci_hi": round(hi, 4),
                         "mean_net_brier_at_k": round(float(value[mask].mean()), 5),
                         "mean_correct_to_wrong_at_k": round(float(ctw[mask].mean()), 5)})
    write_dicts(rag_rows, OUT / "rag_ablation.csv")

    # ---- bootstrap: main advantages ------------------------------------- #
    boot_rows = []
    for a, b, label in (
        ("learned_real", "v_bayes", "spearman learned_real - v_bayes"),
        ("learned_real", "heuristic_verify_utility", "spearman learned_real - heuristic"),
        ("learned_real", "learned_shuffled", "spearman learned_real - shuffled"),
        ("learned_real", "learned_norag", "spearman learned_real - norag"),
    ):
        ra = cluster_bootstrap_spearman(rows, scores[a], "v_real")
        rb = cluster_bootstrap_spearman(rows, scores[b], "v_real")
        dmean, dlo, dhi = cluster_bootstrap_spearman_diff(rows, scores, a, b)
        boot_rows.append({"comparison": label,
                          "a_spearman": round(ra[0], 4),
                          "b_spearman": round(rb[0], 4),
                          "a_ci": f"[{ra[1]:.4f},{ra[2]:.4f}]",
                          "b_ci": f"[{rb[1]:.4f},{rb[2]:.4f}]",
                          "diff_mean": round(dmean, 4),
                          "diff_ci": f"[{dlo:.4f},{dhi:.4f}]",
                          "diff_sig_gt0": int(dlo > 0)})
    write_dicts(boot_rows, OUT / "bootstrap_results.csv")

    # ---- feature importance --------------------------------------------- #
    real_model = models["real"]
    fi_rows = []
    names = list(vw.ALL_FEATURES) if real_model.n_features == len(vw.ALL_FEATURES) \
        else list(vw.BASE_FEATURES)
    if real_model.gain_kind == "hgb":
        from sklearn.inspection import permutation_importance
        X_fi = vw.build_feature_matrix(rows, rag_mode="real")
        imp = permutation_importance(
            real_model.gain_model, X_fi, gain, n_repeats=5, random_state=0
        ).importances_mean
    else:
        imp = np.abs(real_model.gain_model.coef_)
    for name, v in zip(names, imp):
        fi_rows.append({"feature": name, "gain_importance": round(float(v), 6)})
    write_dicts(fi_rows, OUT / "feature_importance.csv")

    # ---- calibration curves --------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        X = vw.build_feature_matrix(rows, rag_mode="real")
        g_hat = real_model.predict(X)[0]
        bins = np.quantile(g_hat, np.linspace(0, 1, 11))
        idx = np.digitize(g_hat, bins[1:-1])
        fig, ax = plt.subplots(figsize=(5, 4))
        pts = []
        for b in range(len(bins) - 1):
            m = idx == b
            if m.sum() < 20:
                continue
            pts.append((g_hat[m].mean(), gain[m].mean()))
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o-", label="gain head")
        ax.plot([gain.min(), gain.max()], [gain.min(), gain.max()], "k--", lw=0.8)
        ax.set_xlabel("predicted gain g_hat")
        ax.set_ylabel("realized gross Brier gain")
        ax.set_title("gain head calibration (validation)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "calibration_curves.png", dpi=110)
        plt.close(fig)
        print("wrote calibration_curves.png")
    except Exception as exc:  # noqa: BLE001
        print(f"calibration plot skipped: {exc}")

    # ---- counterexamples ------------------------------------------------ #
    ce = counterexamples(rows, scores, h_hat, allowed, value)
    write_dicts(ce, OUT / "counterexamples.csv")

    print("evaluation products written")
    return 0


def counterexamples(rows, scores, h_hat, allowed, value):
    s_real = scores["learned_real"]
    harm = np.asarray([vw.derive_labels(r)["harm"] for r in rows])
    out = []
    # high S but negative value
    hi = np.argsort(-s_real)
    for i in hi:
        if value[i] < 0:
            out.append({"category": "high_S_negative_value", "case_id": rows[i]["case_id"],
                        "S": round(float(s_real[i]), 4), "v_real": round(float(value[i]), 4),
                        "h_hat": round(float(h_hat[i]), 4)})
            break
    # low S but high value
    lo = np.argsort(s_real)
    for i in lo:
        if value[i] > 0.05:
            out.append({"category": "low_S_high_value", "case_id": rows[i]["case_id"],
                        "S": round(float(s_real[i]), 4), "v_real": round(float(value[i]), 4),
                        "h_hat": round(float(h_hat[i]), 4)})
            break
    # harm false negative (h_hat low but harm=1)
    hl = np.argsort(h_hat)
    for i in hl:
        if harm[i] == 1:
            out.append({"category": "harm_fn_low_h_hat_but_harm", "case_id": rows[i]["case_id"],
                        "S": round(float(s_real[i]), 4), "v_real": round(float(value[i]), 4),
                        "h_hat": round(float(h_hat[i]), 4)})
            break
    # harm false positive (h_hat high but harm=0)
    hh = np.argsort(-h_hat)
    for i in hh:
        if harm[i] == 0 and value[i] > 0:
            out.append({"category": "harm_fp_high_h_hat_but_beneficial",
                        "case_id": rows[i]["case_id"], "S": round(float(s_real[i]), 4),
                        "v_real": round(float(value[i]), 4),
                        "h_hat": round(float(h_hat[i]), 4)})
            break
    return out


def write_dicts(rows, path):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
