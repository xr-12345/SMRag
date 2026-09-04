"""Phase 22B -- analyze the 4-group N=5 screen.

Consumes ``case_outcomes.csv``, ``reliability_predictions.csv`` and
``turn_decision_logs.csv`` (from ``run_n5_screen.py``) and derives:

  * summary_by_strategy_noise.csv  -- headline metrics per strategy (+ per noise)
  * paired_deltas.csv              -- the four paired comparisons (P-V-P-H,
                                      P-V-L-H, P-V-L-V, L-V-L-H)
  * bootstrap_results.csv          -- case-clustered bootstrap 95% CI for the
                                      headline deltas + Top-1 transitions
  * calibration_metrics.csv        -- p_wrong / p_mode ECE per strategy
                                      (UNKNOWN excluded from p_wrong)
  * stop_audit_events.csv          -- per-turn stop-audit classification
                                      (blocker type: hard-gate-only / value-only
                                      / both / none) for the reliability-relevant
                                      turns
  * baseline_regression_check.txt  -- byte-identical regression vs Phase 8E
                                      (L-H vs joint_legacy_prior, P-H vs
                                      joint_train_fixed_prior)

Red lines: read-only over the CSVs; writes only here; no fitting to labels; no
git commit.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
OUTCOMES = OUT / "case_outcomes.csv"
RELIABILITY = OUT / "reliability_predictions.csv"
TURN_LOGS = OUT / "turn_decision_logs.csv"
PHASE8E_OUTCOMES = (
    Path(__file__).resolve().parents[2]
    / "artifacts/verimedrag-unknown-semantics-train-fixed-n5/case_outcomes.csv"
)

UNKNOWN = "__unknown__"
CONFIGS = ("L-H", "P-H", "L-V", "P-V")
# phase 8e frozen config names -> phase 22b strategy ids
P8E_MAP = {"L-H": "joint_legacy_prior", "P-H": "joint_train_fixed_prior"}
COMPARISONS = (("P-V", "P-H"), ("P-V", "L-H"), ("P-V", "L-V"), ("L-V", "L-H"))

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 2026
ECE_BINS = 10
TAU_P = 0.05
TAU_V = 0.03
TAU_A = 0.03


# --------------------------------------------------------------------------- #
# Metrics
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
        ece += (count / total) * abs(
            float(labels[mask].mean()) - float(probs[mask].mean())
        )
    return float(ece)


def _ci(vals):
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def _f(v, default=0.0):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_rows(path):
    if not path.exists():
        raise SystemExit(f"missing {path}; run run_n5_screen.py first")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _per_case(rows, key, cast=float):
    d = defaultdict(list)
    for r in rows:
        d[r["case_id"]].append(cast(r[key]))
    return {c: np.asarray(v) for c, v in d.items()}


def _mean_of_case_means(rows, key, cast=float):
    pc = _per_case(rows, key, cast)
    return float(np.mean([v.mean() for v in pc.values()]))


def _rate(rows, num_key, den_key):
    num = sum(int(r[num_key]) for r in rows)
    den = sum(int(r[den_key]) for r in rows)
    return (num / den) if den else 0.0


# --------------------------------------------------------------------------- #
# (1) Summary per strategy (and per strategy x noise)
# --------------------------------------------------------------------------- #
def _summary(rows, rng):
    out = []
    for cfg in CONFIGS:
        sub = [r for r in rows if r["config"] == cfg]
        if not sub:
            continue
        row = {"strategy": cfg, "noise_rate": "all", "n_dialogues": len(sub)}
        row.update(_headline_metrics(sub))
        out.append(row)
        for nrate in ("0.2", "0.3"):
            s = [r for r in sub if r["noise_rate"] == nrate]
            if not s:
                continue
            r2 = {"strategy": cfg, "noise_rate": nrate, "n_dialogues": len(s)}
            r2.update(_headline_metrics(s))
            out.append(r2)
    return out


def _headline_metrics(sub):
    # diagnostic calibration on predicted top probability vs correctness
    conf = np.asarray([_f(r["confidence"]) for r in sub])
    top1 = np.asarray([_f(r["correct_top1"]) for r in sub])
    return {
        "top1_acc": float(np.mean([_f(r["correct_top1"]) for r in sub])),
        "top3_acc": float(np.mean([_f(r["correct_top3"]) for r in sub])),
        "diagnostic_brier": float(np.mean([_f(r["brier_score"]) for r in sub])),
        "nll": float(np.mean([_f(r["nll_score"]) for r in sub])),
        "diagnostic_ece": _ece(top1, conf),
        "new_questions": float(np.mean([_f(r["new_questions"]) for r in sub])),
        "verification_questions": float(
            np.mean([_f(r["verification_questions"]) for r in sub])
        ),
        "total_atomic_questions": float(
            np.mean([_f(r["total_atomic_questions"]) for r in sub])
        ),
        "extra_questions_after_threshold": float(
            np.mean([_f(r["extra_questions_after_threshold"]) for r in sub])
        ),
        "unnecessary_verification_rate": _rate(
            sub, "unnecessary_verifications", "verification_questions"
        ),
        "conflict_resolution_rate": _rate(
            sub, "resolved_wrong_reports", "verification_questions"
        ),
        "early_stop_rate": float(np.mean([_f(r["early_stop"]) for r in sub])),
        "premature_stop_rate": float(np.mean([_f(r["premature_stop"]) for r in sub])),
        "stop_audit_triggers_per_dialogue": float(
            np.mean([_f(r["stop_audit_triggers"]) for r in sub])
        ),
        "mean_wall_seconds": float(np.mean([_f(r["wall_clock"]) for r in sub])),
    }


# --------------------------------------------------------------------------- #
# (2) Paired deltas
# --------------------------------------------------------------------------- #
DELTA_METRICS = {
    "top1_acc": ("correct_top1", "mean"),
    "top3_acc": ("correct_top3", "mean"),
    "diagnostic_brier": ("brier_score", "mean"),
    "nll": ("nll_score", "mean"),
    "new_questions": ("new_questions", "mean"),
    "verification_questions": ("verification_questions", "mean"),
    "total_atomic_questions": ("total_atomic_questions", "mean"),
    "extra_questions_after_threshold": ("extra_questions_after_threshold", "mean"),
    "early_stop_rate": ("early_stop", "mean"),
    "premature_stop_rate": ("premature_stop", "mean"),
}


def _paired_deltas(rows):
    by = {c: [r for r in rows if r["config"] == c] for c in CONFIGS}
    out = []
    for a, b in COMPARISONS:
        ra, rb = by[a], by[b]
        for metric, (key, kind) in DELTA_METRICS.items():
            cast = float
            if kind == "mean":
                va = float(np.mean([cast(r[key]) for r in ra]))
                vb = float(np.mean([cast(r[key]) for r in rb]))
            delta = vb - va
            out.append({
                "comparison": f"{a}-{b}", "metric": metric,
                a: round(va, 6), b: round(vb, 6),
                "delta": round(delta, 6),
                "delta_pct": round(delta / va * 100, 4) if va else 0.0,
            })
    return out


# --------------------------------------------------------------------------- #
# (3) Bootstrap results + Top-1 transitions
# --------------------------------------------------------------------------- #
def _bootstrap_outcome_delta(a, b, rows, key, rng, cast=float):
    da = _per_case([r for r in rows if r["config"] == a], key, cast)
    db = _per_case([r for r in rows if r["config"] == b], key, cast)
    cases = sorted(set(da) & set(db))
    lc = np.asarray(cases, dtype=object)
    deltas = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(lc, size=len(lc), replace=True)
        av = float(np.mean(np.concatenate([da[c] for c in sampled])))
        bv = float(np.mean(np.concatenate([db[c] for c in sampled])))
        deltas.append(bv - av)
    amean = float(np.mean(np.concatenate([da[c] for c in cases])))
    bmean = float(np.mean(np.concatenate([db[c] for c in cases])))
    lo, hi = _ci(deltas)
    return amean, bmean, bmean - amean, lo, hi


def _top1_transitions(a, b, rows, rng):
    da, db = {}, {}
    for r in rows:
        k = (r["case_id"], r["noise_rate"], r["seed"])
        if r["config"] == a:
            da[k] = int(r["correct_top1"])
        if r["config"] == b:
            db[k] = int(r["correct_top1"])
    keys = sorted(set(da) & set(db))
    wc = sum(1 for k in keys if da[k] == 0 and db[k] == 1)
    cw = sum(1 for k in keys if da[k] == 1 and db[k] == 0)
    net = wc - cw
    # case-clustered bootstrap of the net transition rate (per case)
    by_case_a = defaultdict(int)
    by_case_b = defaultdict(int)
    case_keys = defaultdict(list)
    for k in keys:
        case_keys[k[0]].append(k)
    cases = sorted(case_keys)
    deltas = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        wc_s = cw_s = 0
        for c in sampled:
            for k in case_keys[c]:
                if da[k] == 0 and db[k] == 1:
                    wc_s += 1
                elif da[k] == 1 and db[k] == 0:
                    cw_s += 1
        deltas.append(wc_s - cw_s)
    lo, hi = _ci(deltas)
    return wc, cw, net, lo, hi


def _bootstrap_results(rows, rng):
    out = []
    for a, b in COMPARISONS:
        for metric, key in (
            ("top1_acc", "correct_top1"),
            ("diagnostic_brier", "brier_score"),
            ("total_atomic_questions", "total_atomic_questions"),
            ("premature_stop_rate", "premature_stop"),
            ("early_stop_rate", "early_stop"),
            ("new_questions", "new_questions"),
        ):
            va, vb, delta, lo, hi = _bootstrap_outcome_delta(a, b, rows, key, rng)
            out.append({
                "comparison": f"{a}-{b}", "metric": metric,
                "a_value": round(va, 6), "b_value": round(vb, 6),
                "delta": round(delta, 6),
                "delta_ci_low": round(lo, 6), "delta_ci_high": round(hi, 6),
            })
        wc, cw, net, lo, hi = _top1_transitions(a, b, rows, rng)
        out.append({
            "comparison": f"{a}-{b}", "metric": "top1_wrong_to_correct",
            "a_value": "", "b_value": "", "delta": wc,
            "delta_ci_low": "", "delta_ci_high": "",
        })
        out.append({
            "comparison": f"{a}-{b}", "metric": "top1_correct_to_wrong",
            "a_value": "", "b_value": "", "delta": cw,
            "delta_ci_low": "", "delta_ci_high": "",
        })
        out.append({
            "comparison": f"{a}-{b}", "metric": "top1_net_improvement",
            "a_value": "", "b_value": "", "delta": net,
            "delta_ci_low": round(lo, 6), "delta_ci_high": round(hi, 6),
        })
    return out


# --------------------------------------------------------------------------- #
# (4) Calibration (p_wrong / p_mode ECE)
# --------------------------------------------------------------------------- #
def _calibration(reliab, rng):
    out = []
    explicit = [r for r in reliab if r["is_nonresponse"] == "0"]
    for cfg in CONFIGS:
        # p_wrong vs true_wrong on explicit answers
        sub = [r for r in explicit if r["config"] == cfg]
        y = np.asarray([int(r["true_wrong"]) for r in sub], dtype=float)
        p = np.asarray([_f(r["p_wrong"]) for r in sub], dtype=float)
        out.append({
            "signal": "p_wrong", "strategy": cfg, "subset": "explicit",
            "n": int(len(y)), "prevalence": float(y.mean()),
            "brier": _brier(y, p), "ece": _ece(y, p),
        })
        # p_mode vs true_misreported on all first answers
        sub2 = [r for r in reliab if r["config"] == cfg]
        y2 = np.asarray([int(r["true_misreported"]) for r in sub2], dtype=float)
        p2 = np.asarray([_f(r["p_mode"]) for r in sub2], dtype=float)
        out.append({
            "signal": "p_mode", "strategy": cfg, "subset": "all_first",
            "n": int(len(y2)), "prevalence": float(y2.mean()),
            "brier": _brier(y2, p2), "ece": _ece(y2, p2),
        })
        # UNKNOWN rate
        unk = np.mean([int(r["is_nonresponse"]) for r in sub2])
        out.append({
            "signal": "is_nonresponse", "strategy": cfg, "subset": "all_first",
            "n": int(len(sub2)), "prevalence": float(unk),
            "brier": "", "ece": "",
        })
    return out


# --------------------------------------------------------------------------- #
# (5) Stop-audit events (per-turn blocker classification)
# --------------------------------------------------------------------------- #
def _blocker_class(max_p_mode, max_gross):
    p_mode_blocks = max_p_mode > TAU_P
    gross_blocks = max_gross > TAU_V
    if p_mode_blocks and gross_blocks:
        return "both", p_mode_blocks, gross_blocks
    if p_mode_blocks:
        return "hard_gate_only", p_mode_blocks, gross_blocks
    if gross_blocks:
        return "value_audit_only", p_mode_blocks, gross_blocks
    return "none", p_mode_blocks, gross_blocks


def _stop_audit_events(turn_rows):
    """One row per reliability-relevant turn + per-strategy blocker counts."""
    out = []
    counts = {c: {"hard_gate_only": 0, "value_audit_only": 0, "both": 0,
                  "reliability_blocked": 0} for c in CONFIGS}
    verify_vs_new = {c: {"gross_block_turns": 0, "gross_block_verify": 0,
                         "gross_block_new": 0} for c in CONFIGS}
    for r in turn_rows:
        cfg = r["strategy"]
        pm = _f(r["max_p_mode"])
        mg = _f(r["max_gross_verify_gain"])
        block, p_mode_blocks, gross_blocks = _blocker_class(pm, mg)
        blocked = r["stop_block_reason"] == "reliability"
        if blocked:
            counts[cfg]["reliability_blocked"] += 1
        if block == "hard_gate_only":
            counts[cfg]["hard_gate_only"] += 1
        elif block == "value_audit_only":
            counts[cfg]["value_audit_only"] += 1
        elif block == "both":
            counts[cfg]["both"] += 1
        if gross_blocks:
            v = verify_vs_new[cfg]
            v["gross_block_turns"] += 1
            if r["chosen_action"] == "verify":
                v["gross_block_verify"] += 1
            elif r["chosen_action"] == "new":
                v["gross_block_new"] += 1
        if block != "none" or blocked:
            out.append({
                "case_id": r["case_id"], "seed": r["seed"],
                "noise_rate": r["noise_rate"], "strategy": cfg,
                "turn": r["turn"], "max_p_mode": r["max_p_mode"],
                "max_gross_verify_gain": r["max_gross_verify_gain"],
                "p_mode_blocks": int(p_mode_blocks),
                "gross_blocks": int(gross_blocks),
                "blocker_type": block,
                "stop_block_reason": r["stop_block_reason"],
                "chosen_action": r["chosen_action"],
            })
    return out, counts, verify_vs_new


# --------------------------------------------------------------------------- #
# (6) Baseline regression vs Phase 8E
# --------------------------------------------------------------------------- #
def _regression_check(rows):
    lines = []
    ok = True
    lines.append("Phase 22B -- baseline byte-identical regression check")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Compare L-H / P-H action sequences against Phase 8E frozen")
    lines.append("joint_legacy_prior / joint_train_fixed_prior (matched 0/0).")
    lines.append("")

    if not PHASE8E_OUTCOMES.exists():
        lines.append(f"WARNING: Phase 8E outcomes not found at {PHASE8E_OUTCOMES}")
        ok = False
    else:
        with open(PHASE8E_OUTCOMES, newline="") as f:
            p8e = list(csv.DictReader(f))
        for strat, p8e_cfg in P8E_MAP.items():
            mine = [r for r in rows if r["config"] == strat]
            ref = [r for r in p8e if r["config"] == p8e_cfg]
            mine_map = {
                (r["case_id"], r["noise_rate"], r["seed"]): r["action_seq"]
                for r in mine
            }
            ref_map = {
                (r["case_id"], r["noise_rate"], r["seed"]): r["action_seq"]
                for r in ref
            }
            mismatch = [
                k for k in sorted(set(mine_map) | set(ref_map))
                if mine_map.get(k) != ref_map.get(k)
            ]
            lines.append(f"{strat} (mine) vs {p8e_cfg} (Phase 8E):")
            lines.append(f"  rows: mine={len(mine_map)} ref={len(ref_map)} "
                         f"mismatch={len(mismatch)}")
            if mismatch:
                ok = False
                for k in mismatch[:20]:
                    lines.append(f"    {k}: ref={ref_map.get(k)!r} "
                                 f"mine={mine_map.get(k)!r}")
            else:
                lines.append("  RESULT: byte-identical (0 mismatches)")
            lines.append("")
    lines.append(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    (OUT / "baseline_regression_check.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote baseline_regression_check.txt (PASS={ok})")
    return ok


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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    outcomes = load_rows(OUTCOMES)
    reliab = load_rows(RELIABILITY)
    turns = load_rows(TURN_LOGS)
    print(f"loaded {len(outcomes)} outcomes, {len(reliab)} reliability, "
          f"{len(turns)} turn rows")

    summary = _summary(outcomes, rng)
    _write_rows(summary, OUT / "summary_by_strategy_noise.csv",
                ["strategy", "noise_rate", "n_dialogues", "top1_acc", "top3_acc",
                 "diagnostic_brier", "nll", "diagnostic_ece", "new_questions",
                 "verification_questions", "total_atomic_questions",
                 "extra_questions_after_threshold",
                 "unnecessary_verification_rate", "conflict_resolution_rate",
                 "early_stop_rate", "premature_stop_rate",
                 "stop_audit_triggers_per_dialogue", "mean_wall_seconds"])

    deltas = _paired_deltas(outcomes)
    _write_rows(deltas, OUT / "paired_deltas.csv",
                ["comparison", "metric", "L-H", "P-H", "L-V", "P-V",
                 "delta", "delta_pct"])

    boot = _bootstrap_results(outcomes, rng)
    _write_rows(boot, OUT / "bootstrap_results.csv",
                ["comparison", "metric", "a_value", "b_value", "delta",
                 "delta_ci_low", "delta_ci_high"])

    calib = _calibration(reliab, rng)
    _write_rows(calib, OUT / "calibration_metrics.csv",
                ["signal", "strategy", "subset", "n", "prevalence", "brier", "ece"])

    events, counts, verify_vs_new = _stop_audit_events(turns)
    _write_rows(events, OUT / "stop_audit_events.csv",
                ["case_id", "seed", "noise_rate", "strategy", "turn",
                 "max_p_mode", "max_gross_verify_gain", "p_mode_blocks",
                 "gross_blocks", "blocker_type", "stop_block_reason",
                 "chosen_action"])

    # stop-audit mechanism counters as JSON for the report
    mechanism = {
        "blocker_counts": counts,
        "gross_block_action_split": verify_vs_new,
    }
    (OUT / "stop_audit_summary.json").write_text(
        json.dumps(mechanism, indent=2), encoding="utf-8")

    ok = _regression_check(outcomes)

    # ---- print headline summary ----------------------------------------- #
    print("\n=== HEADLINE SUMMARY (strategy x all-noise means) ===")
    for r in summary:
        if r["noise_rate"] == "all":
            print(f"{r['strategy']}: top1={r['top1_acc']:.4f} "
                  f"brier={r['diagnostic_brier']:.4f} nll={r['nll']:.4f} "
                  f"q={r['total_atomic_questions']:.2f} "
                  f"early_stop={r['early_stop_rate']:.3f} "
                  f"premature={r['premature_stop_rate']:.4f}")
    print("\n=== PAIRED DELTAS (bootstrap) ===")
    for r in boot:
        if r["metric"] in ("top1_acc", "diagnostic_brier",
                           "total_atomic_questions", "premature_stop_rate"):
            print(f"{r['comparison']} {r['metric']}: {r['a_value']} -> "
                  f"{r['b_value']} delta={r['delta']} "
                  f"CI=[{r['delta_ci_low']}, {r['delta_ci_high']}]")
    print("\n=== STOP-AUDIT BLOCKER COUNTS ===")
    for c in CONFIGS:
        print(f"{c}: {counts[c]}")
    print("\n=== GROSS-BLOCK ACTION SPLIT (is verify forced?) ===")
    for c in CONFIGS:
        v = verify_vs_new[c]
        print(f"{c}: {v}")

    print(f"\nANALYSIS COMPLETE (regression PASS={ok})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
