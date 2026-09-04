"""Phase 8E -- analyze the N=5 screen (diagnostic + reliability calibration).

Consumes ``case_outcomes.csv`` and ``reliability_predictions.csv`` (from
``run_n5_screen.py``) and derives:

  * calibration_metrics.csv   -- p_wrong ECE/Brier (explicit answers only),
                                 p_mode ECE/Brier, UNKNOWN rate, per config,
                                 with case-clustered bootstrap 95% CIs
  * subgroup_metrics.csv      -- p_wrong ECE/Brier sliced by value class
                                 (present/absent/categorical) and noise
  * paired_deltas.csv         -- train_fixed - legacy deltas for headline metrics
  * bootstrap_results.csv     -- bootstrap CI for the headline deltas
  * baseline_regression_check.txt -- byte-identical trajectory regression vs the
                                 Phase 8C joint matched-0/0 legacy run

Label conventions (matching Phase 8D):
  true_wrong         = 1[Z != Y AND Y != UNKNOWN]  (content mismatch; UNKNOWN=0)
  true_misreported   = 1[E = MISREPORTED]
  is_nonresponse     = 1[Y = UNKNOWN]

p_wrong is calibrated against ``true_wrong`` over EXPLICIT answers only
(UNKNOWN rows excluded because p_wrong is None there); p_mode against
``true_misreported`` over all first answers.

Red lines: read-only over the CSVs; writes only here; no fitting to
validation/test labels; no git commit.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
OUTCOMES = OUT / "case_outcomes.csv"
RELIABILITY = OUT / "reliability_predictions.csv"
PHASE8C_OUTCOMES = (
    Path(__file__).resolve().parents[2]
    / "artifacts/verimedrag-joint-channel-alignment-n5/case_outcomes.csv"
)

UNKNOWN = "__unknown__"
BINARY_VALUES = ("present", "absent")
LEGACY = "joint_legacy_prior"
TRAIN_FIXED = "joint_train_fixed_prior"
CONFIGS = (LEGACY, TRAIN_FIXED)

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 2026
ECE_BINS = 10


# --------------------------------------------------------------------------- #
# Metrics (numpy closed form, no sklearn)
# --------------------------------------------------------------------------- #
def _brier(labels, probs):
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def _ece(labels, probs, bins=ECE_BINS):
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    bucket = np.minimum((probs * bins).astype(int), bins - 1)
    total = len(labels)
    ece = 0.0
    for b in range(bins):
        mask = bucket == b
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / total) * abs(float(labels[mask].mean()) - float(probs[mask].mean()))
    return float(ece)


def _case_bootstrap(metric_fn, labels, probs, case_ids, rng):
    """Case-clustered bootstrap CI for a scalar metric."""
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    case_ids = np.asarray(case_ids, dtype=object)
    idx: dict[str, list[int]] = defaultdict(list)
    for j, c in enumerate(case_ids):
        idx[c].append(j)
    case_list = list(idx.keys())
    vals = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(case_list, size=len(case_list), replace=True)
        sel = np.concatenate([idx[c] for c in sampled])
        vals.append(metric_fn(labels[sel], probs[sel]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def _ci(vals):
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_rows(path):
    if not path.exists():
        raise SystemExit(f"missing {path}; run run_n5_screen.py first")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def value_class(value):
    if value == UNKNOWN:
        return "unknown"
    if value in BINARY_VALUES:
        return "binary"
    return "categorical"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    outcomes = load_rows(OUTCOMES)
    reliab = load_rows(RELIABILITY)
    print(f"loaded {len(outcomes)} outcome rows, {len(reliab)} reliability rows")

    # ---- (1) diagnostic metrics per config ------------------------------- #
    diag_rows = _diagnostic_metrics(outcomes, rng)
    _write_rows(diag_rows, OUT / "paired_deltas.csv",
                ["metric", LEGACY, TRAIN_FIXED, "delta", "delta_pct", "note"])
    print("wrote paired_deltas.csv")

    # ---- (2) reliability calibration per config -------------------------- #
    calib_rows = _calibration_metrics(reliab, rng)
    _write_rows(calib_rows, OUT / "calibration_metrics.csv",
                ["signal", "config", "subset", "n", "prevalence", "Brier",
                 "ECE", "ECE_ci", "note"])
    print("wrote calibration_metrics.csv")

    # ---- (3) subgroup metrics -------------------------------------------- #
    subgroup_rows = _subgroup_metrics(reliab, rng)
    _write_rows(subgroup_rows, OUT / "subgroup_metrics.csv",
                ["stratum", "config", "signal", "n", "prevalence", "Brier",
                 "ECE", "note"])
    print("wrote subgroup_metrics.csv")

    # ---- (4) bootstrap results for headline deltas ----------------------- #
    bootstrap_rows = _bootstrap_deltas(reliab, outcomes, rng)
    _write_rows(bootstrap_rows, OUT / "bootstrap_results.csv",
                ["metric", "legacy_value", "train_fixed_value", "delta",
                 "delta_ci_low", "delta_ci_high", "note"])
    print("wrote bootstrap_results.csv")

    # ---- (5) byte-identical regression vs Phase 8C ----------------------- #
    _regression_check(outcomes)

    print("ANALYSIS COMPLETE")
    return 0


# --------------------------------------------------------------------------- #
# Diagnostic metrics
# --------------------------------------------------------------------------- #
def _diagnostic_metrics(outcomes, rng):
    """Per-config diagnostic means + case-clustered bootstrap CI for deltas."""
    by_config = {c: [r for r in outcomes if r["config"] == c] for c in CONFIGS}

    def mean(rows, key, cast=float):
        return float(np.mean([cast(r[key]) for r in rows]))

    def rate(rows, num_key, den_key):
        num = sum(int(r[num_key]) for r in rows)
        den = sum(int(r[den_key]) for r in rows)
        return (num / den) if den else 0.0

    def per_case(rows, key, cast=float):
        d = defaultdict(list)
        for r in rows:
            d[r["case_id"]].append(cast(r[key]))
        return np.asarray([float(np.mean(v)) for v in d.values()])

    metrics = {
        "top1_acc": ("correct_top1", "mean"),
        "top3_acc": ("correct_top3", "mean"),
        "diagnostic_brier": ("brier_score", "mean"),
        "nll": ("nll_score", "mean"),
        "total_atomic_questions": ("total_atomic_questions", "mean"),
        "new_questions": ("new_questions", "mean"),
        "verification_questions": ("verification_questions", "mean"),
        "early_stop_rate": ("early_stop", "mean"),
        "premature_stop_rate": ("premature_stop", "mean"),
    }

    rows = []
    for metric, (key, _) in metrics.items():
        legacy = mean(by_config[LEGACY], key, int if "questions" in metric or "rate" in metric else float)
        fixed = mean(by_config[TRAIN_FIXED], key, int if "questions" in metric or "rate" in metric else float)
        delta = fixed - legacy
        pct = (delta / legacy * 100.0) if legacy else 0.0
        rows.append({
            "metric": metric, LEGACY: round(legacy, 6), TRAIN_FIXED: round(fixed, 6),
            "delta": round(delta, 6), "delta_pct": round(pct, 4),
            "note": "",
        })

    # rate metrics (need paired num/den)
    for metric, (num_key, den_key) in {
        "unnecessary_verification_rate": ("unnecessary_verifications", "verification_questions"),
        "conflict_resolution_rate": ("resolved_wrong_reports", "verification_questions"),
    }.items():
        legacy = rate(by_config[LEGACY], num_key, den_key)
        fixed = rate(by_config[TRAIN_FIXED], num_key, den_key)
        delta = fixed - legacy
        rows.append({
            "metric": metric, LEGACY: round(legacy, 6), TRAIN_FIXED: round(fixed, 6),
            "delta": round(delta, 6), "delta_pct": round(delta / legacy * 100, 4) if legacy else 0.0,
            "note": "per-verification rate",
        })

    return rows


def _bootstrap_deltas(reliab, outcomes, rng):
    """Case-clustered bootstrap CI for the headline paired deltas."""
    rows = []

    # p_wrong ECE on explicit answers
    explicit = [r for r in reliab if r["is_nonresponse"] == "0"]
    legacy_pw = _per_case_arrays(explicit, LEGACY, "p_wrong", "true_wrong")
    fixed_pw = _per_case_arrays(explicit, TRAIN_FIXED, "p_wrong", "true_wrong")
    rows.append(_bootstrap_delta_row(
        "p_wrong_ECE_explicit", _ece, legacy_pw, fixed_pw, rng,
        "ECE on explicit answers (UNKNOWN excluded)"))

    # p_mode ECE on all first answers
    all_first = reliab
    legacy_pm = _per_case_arrays(all_first, LEGACY, "p_mode", "true_misreported")
    fixed_pm = _per_case_arrays(all_first, TRAIN_FIXED, "p_mode", "true_misreported")
    rows.append(_bootstrap_delta_row(
        "p_mode_ECE", _ece, legacy_pm, fixed_pm, rng, "ECE on all first answers"))

    # diagnostic brier (per-case means)
    rows.append(_bootstrap_outcome_delta(
        "diagnostic_brier", "brier_score", outcomes, rng))
    rows.append(_bootstrap_outcome_delta(
        "top1_acc", "correct_top1", outcomes, rng))
    rows.append(_bootstrap_outcome_delta(
        "total_atomic_questions", "total_atomic_questions", outcomes, rng))

    return rows


def _per_case_arrays(rows, config, prob_key, label_key):
    d_p = defaultdict(list)
    d_l = defaultdict(list)
    for r in rows:
        if r["config"] != config:
            continue
        pv = r[prob_key]
        if pv == "":
            continue
        d_p[r["case_id"]].append(float(pv))
        d_l[r["case_id"]].append(int(r[label_key]))
    return d_p, d_l


def _bootstrap_delta_row(name, metric_fn, legacy, fixed, rng, note):
    lp, ll = legacy
    fp, fl = fixed
    cases = sorted(set(lp) & set(fp))
    lc = np.asarray(cases, dtype=object)
    deltas = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(lc, size=len(lc), replace=True)
        l_sel = np.concatenate([np.asarray(lp[c], dtype=float) for c in sampled])
        l_lab = np.concatenate([np.asarray(ll[c], dtype=float) for c in sampled])
        f_sel = np.concatenate([np.asarray(fp[c], dtype=float) for c in sampled])
        f_lab = np.concatenate([np.asarray(fl[c], dtype=float) for c in sampled])
        deltas.append(metric_fn(f_lab, f_sel) - metric_fn(l_lab, l_sel))
    l_val = metric_fn(np.concatenate([np.asarray(ll[c], dtype=float) for c in cases]),
                      np.concatenate([np.asarray(lp[c], dtype=float) for c in cases]))
    f_val = metric_fn(np.concatenate([np.asarray(fl[c], dtype=float) for c in cases]),
                      np.concatenate([np.asarray(fp[c], dtype=float) for c in cases]))
    lo, hi = _ci(deltas)
    return {
        "metric": name, "legacy_value": round(l_val, 6),
        "train_fixed_value": round(f_val, 6), "delta": round(f_val - l_val, 6),
        "delta_ci_low": round(lo, 6), "delta_ci_high": round(hi, 6), "note": note,
    }


def _bootstrap_outcome_delta(name, key, outcomes, rng):
    by_config = {c: defaultdict(list) for c in CONFIGS}
    for r in outcomes:
        by_config[r["config"]][r["case_id"]].append(float(r[key]))
    lp = {c: np.asarray(v) for c, v in by_config[LEGACY].items()}
    fp = {c: np.asarray(v) for c, v in by_config[TRAIN_FIXED].items()}
    cases = sorted(set(lp) & set(fp))
    lc = np.asarray(cases, dtype=object)
    deltas = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(lc, size=len(lc), replace=True)
        l_val = float(np.mean(np.concatenate([lp[c] for c in sampled])))
        f_val = float(np.mean(np.concatenate([fp[c] for c in sampled])))
        deltas.append(f_val - l_val)
    l_mean = float(np.mean(np.concatenate([lp[c] for c in cases])))
    f_mean = float(np.mean(np.concatenate([fp[c] for c in cases])))
    lo, hi = _ci(deltas)
    return {
        "metric": name, "legacy_value": round(l_mean, 6),
        "train_fixed_value": round(f_mean, 6), "delta": round(f_mean - l_mean, 6),
        "delta_ci_low": round(lo, 6), "delta_ci_high": round(hi, 6),
        "note": "case-clustered bootstrap",
    }


# --------------------------------------------------------------------------- #
# Reliability calibration
# --------------------------------------------------------------------------- #
def _calibration_metrics(reliab, rng):
    rows = []
    explicit = [r for r in reliab if r["is_nonresponse"] == "0"]
    all_first = reliab

    for config in CONFIGS:
        # p_wrong vs true_wrong on explicit answers (UNKNOWN excluded)
        exp_cfg = [r for r in explicit if r["config"] == config]
        y = np.asarray([int(r["true_wrong"]) for r in exp_cfg], dtype=float)
        p = np.asarray([float(r["p_wrong"]) for r in exp_cfg], dtype=float)
        cid = np.asarray([r["case_id"] for r in exp_cfg], dtype=object)
        rows.append({
            "signal": "p_wrong", "config": config, "subset": "explicit",
            "n": int(len(y)), "prevalence": float(y.mean()),
            "Brier": _brier(y, p), "ECE": _ece(y, p),
            "ECE_ci": _case_bootstrap(_ece, y, p, cid, rng),
            "note": "UNKNOWN excluded (p_wrong=None)",
        })

        # p_mode vs true_misreported on all first answers
        all_cfg = [r for r in all_first if r["config"] == config]
        y2 = np.asarray([int(r["true_misreported"]) for r in all_cfg], dtype=float)
        p2 = np.asarray([float(r["p_mode"]) for r in all_cfg], dtype=float)
        cid2 = np.asarray([r["case_id"] for r in all_cfg], dtype=object)
        rows.append({
            "signal": "p_mode", "config": config, "subset": "all_first",
            "n": int(len(y2)), "prevalence": float(y2.mean()),
            "Brier": _brier(y2, p2), "ECE": _ece(y2, p2),
            "ECE_ci": _case_bootstrap(_ece, y2, p2, cid2, rng),
            "note": "all first answers (no UNKNOWN conflation)",
        })

    # UNKNOWN rate per config
    for config in CONFIGS:
        cfg = [r for r in all_first if r["config"] == config]
        unk_rate = np.mean([int(r["is_nonresponse"]) for r in cfg])
        rows.append({
            "signal": "is_nonresponse", "config": config, "subset": "all_first",
            "n": len(cfg), "prevalence": float(unk_rate),
            "Brier": "", "ECE": "", "ECE_ci": "",
            "note": "UNKNOWN rate (counted separately)",
        })

    return rows


def _subgroup_metrics(reliab, rng):
    rows = []
    explicit = [r for r in reliab if r["is_nonresponse"] == "0"]

    for config in CONFIGS:
        cfg = [r for r in explicit if r["config"] == config]
        # value class (present/absent/categorical)
        for val in ("present", "absent"):
            sub = [r for r in cfg if r["value"] == val]
            if not sub:
                continue
            y = np.asarray([int(r["true_wrong"]) for r in sub], dtype=float)
            p = np.asarray([float(r["p_wrong"]) for r in sub], dtype=float)
            rows.append({
                "stratum": f"value={val}", "config": config, "signal": "p_wrong",
                "n": len(sub), "prevalence": float(y.mean()),
                "Brier": _brier(y, p), "ECE": _ece(y, p),
                "note": "",
            })
        cat = [r for r in cfg if value_class(r["value"]) == "categorical"]
        if cat:
            y = np.asarray([int(r["true_wrong"]) for r in cat], dtype=float)
            p = np.asarray([float(r["p_wrong"]) for r in cat], dtype=float)
            rows.append({
                "stratum": "value=categorical", "config": config, "signal": "p_wrong",
                "n": len(cat), "prevalence": float(y.mean()),
                "Brier": _brier(y, p), "ECE": _ece(y, p),
                "note": "",
            })
        # noise
        for nrate in ("0.2", "0.3"):
            sub = [r for r in cfg if r["noise_rate"] == nrate]
            if not sub:
                continue
            y = np.asarray([int(r["true_wrong"]) for r in sub], dtype=float)
            p = np.asarray([float(r["p_wrong"]) for r in sub], dtype=float)
            rows.append({
                "stratum": f"noise={nrate}", "config": config, "signal": "p_wrong",
                "n": len(sub), "prevalence": float(y.mean()),
                "Brier": _brier(y, p), "ECE": _ece(y, p),
                "note": "",
            })

    return rows


# --------------------------------------------------------------------------- #
# Byte-identical regression vs Phase 8C
# --------------------------------------------------------------------------- #
def _regression_check(outcomes):
    lines = []
    ok = True
    lines.append("Phase 8E -- baseline byte-identical regression check")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Compare joint_legacy_prior action sequences against the Phase 8C")
    lines.append("joint strategy at matched 0/0 (rho_env=0, rho_model=0). The UNKNOWN")
    lines.append("semantic fix must NOT change the trajectory.")
    lines.append("")

    legacy = [r for r in outcomes if r["config"] == LEGACY]
    if not PHASE8C_OUTCOMES.exists():
        lines.append(f"WARNING: Phase 8C outcomes not found at {PHASE8C_OUTCOMES}")
        lines.append("Cannot run the cross-phase byte-identical comparison.")
        ok = False
    else:
        with open(PHASE8C_OUTCOMES, newline="") as f:
            p8c = list(csv.DictReader(f))
        p8c_joint = [r for r in p8c if r["strategy"] == "joint_channel_brier_audit"
                     and r["rho_env"] == "0.0" and r["rho_model"] == "0.0"]
        p8c_map = {
            (r["case_id"], r["noise_rate"], r["seed"]): r["action_seq"]
            for r in p8c_joint
        }
        mine_map = {
            (r["case_id"], r["noise_rate"], r["seed"]): r["action_seq"]
            for r in legacy
        }
        mismatch = []
        for key in sorted(set(p8c_map) | set(mine_map)):
            if p8c_map.get(key) != mine_map.get(key):
                mismatch.append((key, p8c_map.get(key), mine_map.get(key)))
        lines.append(f"Phase 8C joint matched-0/0 rows : {len(p8c_map)}")
        lines.append(f"Phase 8E legacy config rows    : {len(mine_map)}")
        lines.append(f"Mismatched action sequences   : {len(mismatch)}")
        if mismatch:
            ok = False
            lines.append("")
            lines.append("MISMATCHES (case_id, noise, seed): phase8c_seq -> phase8e_seq")
            for key, a, b in mismatch[:50]:
                lines.append(f"  {key}: {a!r} -> {b!r}")
        else:
            lines.append("RESULT: byte-identical (0 mismatches) -- UNKNOWN fix does not "
                         "change the trajectory.")
    lines.append("")
    lines.append(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    (OUT / "baseline_regression_check.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote baseline_regression_check.txt (PASS={ok})")


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def _write_rows(rows, path, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt_cell(r.get(k)) for k in fields})
    print(f"wrote {path.name} ({len(rows)} rows)")


def _fmt_cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    if isinstance(v, tuple):
        return f"{v[0]:.6f},{v[1]:.6f}"
    return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
