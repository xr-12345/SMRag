"""Phase 8D -- analyze the offline paired audit predictions.

Consumes ``paired_audit_predictions.csv`` (produced by ``run_offline_paired_audit.py``)
and derives:

  * ``paired_audit_metrics.csv``    -- AUROC / AUPRC / Brier / ECE / adaptive ECE /
                                        calibration slope+intercept per (label, config,
                                        subset), with case-clustered bootstrap 95% CIs
  * ``stratification_metrics.csv``  -- ECE / Brier sliced by noise / value / certainty /
                                        verified / timepoint / case-vs-seed
  * ``reproduction_sanity.json``    -- reproduces Phase 8C's p_wrong ECE ~= 0.156 on the
                                        legacy config (sanity that the pipeline matches)
  * ``reliability_diagrams.png``    -- reliability curves, legacy vs oracle vs train_fixed
                                        vs cue_conditioned

Label conventions (spec section 3, separated):
  wrong_report      = 1[Z_i != Y_i]      content mismatch (main analysis: Y in {present, absent})
  latent_misreport  = 1[E_i = MISREPORTED]
  harmful_misreport = wrong_report AND latent_misreport

p_wrong is calibrated against ``wrong_report``; p_mode against ``latent_misreport``.
``harmful_misreport`` is not a p_wrong/p_mode target (it is the learned-gate label, Step 5).

Phase 8C reproduction: Phase 8C recorded one p_wrong per feature at first answer with
label ``true_wrong = true_state != UNKNOWN and value != UNKNOWN and value != true_state``
(i.e. ``wrong_report AND value != UNKNOWN``), over ALL feature types (binary + categorical
+ unknown).  We reproduce exactly that on the legacy config as the sanity check.

Red lines: read-only over the predictions CSV; writes only into this artifact directory;
no fitting to validation/test labels; no git commit.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent
CSV_PATH = OUT / "paired_audit_predictions.csv"

CONFIGS = ("legacy", "oracle", "train_fixed", "cue_conditioned")
CONFIG_LABELS = {
    "legacy": "legacy (default cue_priors)",
    "oracle": "oracle (env P(E|cue), privileged)",
    "train_fixed": "train_fixed (cross-noise avg, no cue)",
    "cue_conditioned": "cue_conditioned (cross-noise avg + cue mech)",
}
UNKNOWN = "__unknown__"
BINARY_VALUES = ("present", "absent")

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 2026
ECE_BINS = 10


# --------------------------------------------------------------------------- #
# Metrics (numpy/scipy closed form, no sklearn)
# --------------------------------------------------------------------------- #
def _auroc(labels, scores):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n_pos = float(labels.sum())
    n_neg = float(len(labels)) - n_pos
    if n_pos == 0.0 or n_neg == 0.0:
        return math.nan
    ranks = rankdata(scores, method="average")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _auprc(labels, scores):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n_pos = float(labels.sum())
    if n_pos == 0.0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    positions = np.arange(1, len(sorted_labels) + 1, dtype=float)
    precision = np.cumsum(sorted_labels) / positions
    return float(precision[sorted_labels == 1].sum() / n_pos)


def _brier(labels, probs):
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def _ece(labels, probs, bins=ECE_BINS):
    """Equal-width 10-bin ECE, matching Phase 8C run_analysis._ece exactly."""
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


def _adaptive_ece(labels, probs, bins=ECE_BINS):
    """Equal-mass ECE: equal number of samples per bin, each bin weighted 1/B."""
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    n = len(labels)
    if n == 0:
        return math.nan
    order = np.argsort(probs, kind="mergesort")
    sl = labels[order]
    sp = probs[order]
    ece = 0.0
    for b in range(bins):
        lo = int(round(b * n / bins))
        hi = int(round((b + 1) * n / bins))
        if lo >= hi:
            continue
        ece += (1.0 / bins) * abs(float(sl[lo:hi].mean()) - float(sp[lo:hi].mean()))
    return float(ece)


def _calibration_slope_intercept(labels, probs):
    """Fit logit(y) = intercept + slope * logit(p) by gradient descent."""
    labels = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    logit_p = np.log(p / (1.0 - p))
    intercept = 0.0
    slope = 1.0
    n = len(labels)
    for iteration in range(2000):
        z = intercept + slope * logit_p
        prob = 1.0 / (1.0 + np.exp(-z))
        residual = prob - labels
        rate = 0.05 / math.sqrt(1.0 + iteration / 100.0)
        intercept -= rate * float(residual.mean())
        slope -= rate * float((residual * logit_p).mean())
    return float(intercept), float(slope)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_rows():
    if not CSV_PATH.exists():
        raise SystemExit(f"missing {CSV_PATH}; run run_offline_paired_audit.py first")
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def value_class(value):
    if value == UNKNOWN:
        return "unknown"
    if value in BINARY_VALUES:
        return "binary"
    return "categorical"


def label_wrong_report(row):
    """Phase 8C true_wrong convention: content mismatch where the answer is
    not UNKNOWN (a report of 'unknown' is not a wrong content statement)."""
    return 1 if (int(row["wrong_report"]) == 1 and row["value"] != UNKNOWN) else 0


def p_wrong_col(config):
    return f"p_wrong_{config}"


def p_mode_col(config):
    return f"p_mode_{config}"


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def metric_block(y, probs, case_ids, rng):
    """One (labels, probs, case_ids) -> full metric dict with bootstrap CI."""
    y = np.asarray(y, dtype=float)
    probs = np.asarray(probs, dtype=float)
    case_ids = np.asarray(case_ids, dtype=object)
    if len(y) == 0:
        return None
    auroc = _auroc(y, probs)
    auprc = _auprc(y, probs)
    brier = _brier(y, probs)
    ece = _ece(y, probs)
    adap = _adaptive_ece(y, probs)
    slope, intercept = _calibration_slope_intercept(y, probs)

    # case-clustered bootstrap
    case_idx: dict[str, list[int]] = defaultdict(list)
    for j, c in enumerate(case_ids):
        case_idx[c].append(j)
    case_list = list(case_idx.keys())
    auroc_b, auprc_b, brier_b, ece_b, adap_b = [], [], [], [], []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(case_list, size=len(case_list), replace=True)
        idx = np.concatenate([case_idx[c] for c in sampled])
        auroc_b.append(_auroc(y[idx], probs[idx]))
        auprc_b.append(_auprc(y[idx], probs[idx]))
        brier_b.append(_brier(y[idx], probs[idx]))
        ece_b.append(_ece(y[idx], probs[idx]))
        adap_b.append(_adaptive_ece(y[idx], probs[idx]))

    def ci(vals):
        return (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))

    return {
        "n": int(len(y)),
        "cases": len(case_list),
        "prevalence": float(y.mean()),
        "AUROC": auroc, "AUROC_ci": ci(auroc_b),
        "AUPRC": auprc, "AUPRC_ci": ci(auprc_b),
        "Brier": brier, "Brier_ci": ci(brier_b),
        "ECE": ece, "ECE_ci": ci(ece_b),
        "adaptive_ECE": adap, "adaptive_ECE_ci": ci(adap_b),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    rows = load_rows()
    print(f"loaded {len(rows)} rows from {CSV_PATH.name}")

    # Precompute arrays once
    first = np.asarray([r["timepoint"] == "first" for r in rows], dtype=bool)
    case_ids = np.asarray([r["case_id"] for r in rows], dtype=object)
    noise = np.asarray([float(r["noise_rate"]) for r in rows], dtype=float)
    value = np.asarray([r["value"] for r in rows], dtype=object)
    vc = np.asarray([value_class(v) for v in value], dtype=object)
    certainty = np.asarray([r["certainty"] for r in rows], dtype=object)
    verified = np.asarray([int(r["verified"]) for r in rows], dtype=int)
    seed = np.asarray([r["seed"] for r in rows], dtype=object)
    timepoint = np.asarray([r["timepoint"] for r in rows], dtype=object)

    wrong_report = np.asarray([int(r["wrong_report"]) for r in rows], dtype=float)
    latent = np.asarray([int(r["latent_misreport"]) for r in rows], dtype=float)
    harmful = np.asarray([int(r["harmful_misreport"]) for r in rows], dtype=float)
    # Phase 8C-compatible content-mismatch label (answer != UNKNOWN)
    true_wrong_8c = np.asarray([label_wrong_report(r) for r in rows], dtype=float)

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # -- (1) reproduction sanity ------------------------------------------- #
    # Phase 8C computed ECE over ALL first-answer rows (INCLUDING unknown-valued
    # answers): label true_wrong = 0 for unknown answers, while p_wrong = 1.0 for
    # them -- the unknown conflation is part of the 0.156.  So reproduce on `first`.
    repro_mask = first
    repro = metric_block(
        true_wrong_8c[repro_mask],
        np.asarray([float(r[p_wrong_col("legacy")]) for r in rows], dtype=float)[repro_mask],
        case_ids[repro_mask], rng,
    )
    repro["note"] = (
        "Phase 8C reproduction: ALL first-answer rows (incl. unknown), label = "
        "wrong_report AND value != UNKNOWN, legacy p_wrong.  Phase 8C matched 0/0 "
        "p_wrong ECE = 0.1555."
    )
    print(f"  reproduction legacy p_wrong ECE = {repro['ECE']:.6f} "
          f"(Phase 8C = 0.1555; n={repro['n']})")

    # -- (2) main metric table --------------------------------------------- #
    metric_rows: list[dict] = []
    # p_wrong vs wrong_report (binary features, main analysis)
    binary_first = first & (vc == "binary")
    for config in CONFIGS:
        probs = np.asarray([float(r[p_wrong_col(config)]) for r in rows], dtype=float)
        for subset_name, mask in (("binary_first", binary_first),):
            m = metric_block(wrong_report[mask], probs[mask], case_ids[mask], rng)
            metric_rows.append({
                "signal": "p_wrong", "config": config,
                "label": "wrong_report", "subset": subset_name, **m,
            })

    # p_mode vs latent_misreport (all first-answer features)
    for config in CONFIGS:
        probs = np.asarray([float(r[p_mode_col(config)]) for r in rows], dtype=float)
        m = metric_block(latent[first], probs[first], case_ids[first], rng)
        metric_rows.append({
            "signal": "p_mode", "config": config,
            "label": "latent_misreport", "subset": "all_first", **m,
        })

    # p_wrong vs true_wrong over ALL first-answer features (binary+categorical+
    # unknown) -- the exact Phase 8C metric, for every config.  This is the number
    # that reproduces 0.1555 for legacy.
    for config in CONFIGS:
        probs = np.asarray([float(r[p_wrong_col(config)]) for r in rows], dtype=float)
        m = metric_block(true_wrong_8c[first], probs[first], case_ids[first], rng)
        metric_rows.append({
            "signal": "p_wrong", "config": config,
            "label": "wrong_report", "subset": "all_first_incl_unknown", **m,
        })

    # p_wrong vs wrong_report over first-answer features EXCLUDING unknown answers
    # (binary + categorical) -- isolates the prior/cue mismatch from the unknown
    # conflation.
    nonunknown_first = first & (value != UNKNOWN)
    for config in CONFIGS:
        probs = np.asarray([float(r[p_wrong_col(config)]) for r in rows], dtype=float)
        m = metric_block(true_wrong_8c[nonunknown_first], probs[nonunknown_first],
                         case_ids[nonunknown_first], rng)
        metric_rows.append({
            "signal": "p_wrong", "config": config,
            "label": "wrong_report", "subset": "all_first_nonunknown", **m,
        })

    metric_fields = [
        "signal", "config", "label", "subset", "n", "cases", "prevalence",
        "AUROC", "AUROC_ci", "AUPRC", "AUPRC_ci", "Brier", "Brier_ci",
        "ECE", "ECE_ci", "adaptive_ECE", "adaptive_ECE_ci",
        "calibration_slope", "calibration_intercept",
    ]
    _write_metrics_csv(metric_rows, metric_fields, OUT / "paired_audit_metrics.csv")
    _write_metrics_json(metric_rows, OUT / "paired_audit_metrics.json")

    # -- (3) stratification ------------------------------------------------- #
    strat_rows: list[dict] = []

    def strat_metric(name, mask, config, signal, label_arr, probs, note=""):
        if int(mask.sum()) == 0:
            return
        m = metric_block(label_arr[mask], probs[mask], case_ids[mask], rng)
        strat_rows.append({
            "stratum": name, "config": config, "signal": signal,
            "n": m["n"], "prevalence": m["prevalence"],
            "Brier": m["Brier"], "ECE": m["ECE"],
            "adaptive_ECE": m["adaptive_ECE"],
            "note": note,
        })

    # p_wrong (binary features only) stratified by noise / certainty / verified /
    # timepoint / value / seed
    for config in CONFIGS:
        probs = np.asarray([float(r[p_wrong_col(config)]) for r in rows], dtype=float)
        base = binary_first
        # noise
        for nrate in (0.2, 0.3):
            strat_metric(f"noise={nrate}", base & (noise == nrate), config, "p_wrong",
                         wrong_report, probs)
        # certainty
        for cue in ("none", "certain", "uncertain"):
            strat_metric(f"certainty={cue}", base & (certainty == cue), config, "p_wrong",
                         wrong_report, probs)
        # verified before/after (binary, any timepoint)
        bin_all = vc == "binary"
        for vflag, vname in ((0, "before_verify"), (1, "after_verify")):
            strat_metric(f"verified={vname}", bin_all & (verified == vflag), config,
                         "p_wrong", wrong_report, probs)
        # timepoint (binary)
        for tp in ("first", "reask"):
            strat_metric(f"timepoint={tp}", bin_all & (timepoint == tp), config, "p_wrong",
                         wrong_report, probs)
        # value
        for val in ("present", "absent"):
            strat_metric(f"value={val}", base & (value == val), config, "p_wrong",
                         wrong_report, probs)

    # p_wrong on UNKNOWN values: the conflation.  The content-mismatch label
    # (true_wrong = wrong_report AND value != UNKNOWN) is 0 for an 'unknown'
    # answer, but p_wrong returns 1.0 -- so ECE over this stratum is ~1.0 for
    # every config, and NO prior alignment can reduce it.
    for config in CONFIGS:
        probs = np.asarray([float(r[p_wrong_col(config)]) for r in rows], dtype=float)
        unk_mask = first & (vc == "unknown")
        strat_metric("value=unknown", unk_mask, config, "p_wrong",
                     true_wrong_8c, probs, note="p_wrong=1.0 conflation (label=0)")
        cat_mask = first & (vc == "categorical")
        strat_metric("value=categorical", cat_mask, config, "p_wrong",
                     true_wrong_8c, probs)

    # p_mode stratified by noise
    for config in CONFIGS:
        probs = np.asarray([float(r[p_mode_col(config)]) for r in rows], dtype=float)
        for nrate in (0.2, 0.3):
            strat_metric(f"noise={nrate}", first & (noise == nrate), config, "p_mode",
                         latent, probs)

    # seed-vs-case: p_wrong legacy binary, two seeds, two diagnostic strata
    for config in ("legacy", "oracle"):
        probs = np.asarray([float(r[p_wrong_col(config)]) for r in rows], dtype=float)
        for s in ("2026", "2027"):
            strat_metric(f"seed={s}", binary_first & (seed == s), config, "p_wrong",
                         wrong_report, probs)

    strat_fields = ["stratum", "config", "signal", "n", "prevalence",
                    "Brier", "ECE", "adaptive_ECE", "note"]
    _write_strat_csv(strat_rows, strat_fields, OUT / "stratification_metrics.csv")

    # -- (4) reliability diagrams ------------------------------------------ #
    _plot_reliability(rows, first, vc, wrong_report, latent, true_wrong_8c)

    # -- (5) root-cause summary -------------------------------------------- #
    _root_cause_summary(metric_rows, repro)

    print("ANALYSIS COMPLETE")
    return 0


# --------------------------------------------------------------------------- #
# Root-cause summary
# --------------------------------------------------------------------------- #
def _root_cause_summary(metric_rows, repro):
    """Extract the headline ECE numbers per config for the diagnostic."""
    def ece_of(signal, config, subset):
        for r in metric_rows:
            if (r["signal"] == signal and r["config"] == config
                    and r["subset"] == subset):
                return r["ECE"]
        return None

    summary = {
        "reproduction_legacy_p_wrong_ECE": repro["ECE"],
        "reproduction_n": repro["n"],
        "phase8c_reference_ECE": 0.1555,
        "p_wrong_ECE_binary_first": {
            c: ece_of("p_wrong", c, "binary_first") for c in CONFIGS
        },
        "p_wrong_ECE_all_first_nonunknown": {
            c: ece_of("p_wrong", c, "all_first_nonunknown") for c in CONFIGS
        },
        "p_mode_ECE_all_first": {
            c: ece_of("p_mode", c, "all_first") for c in CONFIGS
        },
    }
    (OUT / "root_cause_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


# --------------------------------------------------------------------------- #
# Writers / plots
# --------------------------------------------------------------------------- #
def _write_metrics_csv(metric_rows, fields, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in metric_rows:
            writer.writerow({k: _fmt_cell(r.get(k)) for k in fields})
    print(f"wrote {path.name} ({len(metric_rows)} rows)")


def _write_metrics_json(metric_rows, path):
    out = {}
    for r in metric_rows:
        key = f"{r['signal']}|{r['config']}|{r['label']}|{r['subset']}"
        out[key] = {k: (r[k] if not isinstance(r[k], tuple) else list(r[k]))
                    for k in r if k not in ("signal", "config", "label", "subset")}
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.name}")


def _write_strat_csv(strat_rows, fields, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in strat_rows:
            writer.writerow({k: _fmt_cell(r.get(k)) for k in fields})
    print(f"wrote {path.name} ({len(strat_rows)} rows)")


def _fmt_cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _plot_reliability(rows, first, vc, wrong_report, latent, true_wrong_8c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    binary = np.asarray(vc == "binary", dtype=bool)
    n_rows = len(rows)
    probs_wrong = {c: np.asarray([float(r[p_wrong_col(c)]) for r in rows], dtype=float)
                   for c in CONFIGS}
    probs_mode = {c: np.asarray([float(r[p_mode_col(c)]) for r in rows], dtype=float)
                  for c in CONFIGS}

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey="row")
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    def curve(axis, probs, y, title, color):
        bins = np.linspace(0, 1, ECE_BINS + 1)
        centers, fracs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (probs >= lo) & (probs < hi)
            if sel.sum() == 0:
                continue
            centers.append((lo + hi) / 2)
            fracs.append(float(y[sel].mean()))
        axis.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        axis.plot(centers, fracs, "o-", markersize=4, color=color)
        axis.set_title(title, fontsize=9)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)

    # row 0: p_wrong vs wrong_report (binary first-answer)
    for ax, c, col in zip(axes[0], CONFIGS, colors):
        m = binary & first
        curve(ax, probs_wrong[c][m], wrong_report[m], f"p_wrong {c}\n(binary, first)", col)
    axes[0][0].set_ylabel("empirical wrong fraction")

    # row 1: p_mode vs latent_misreport (all first-answer)
    for ax, c, col in zip(axes[1], CONFIGS, colors):
        curve(ax, probs_mode[c][first], latent[first], f"p_mode {c}\n(all, first)", col)
    axes[1][0].set_ylabel("empirical misreport fraction")
    for ax in axes[1]:
        ax.set_xlabel("predicted probability")

    fig.suptitle("Reliability diagrams — offline paired audit (Phase 8D)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "reliability_diagrams.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote reliability_diagrams.png")


if __name__ == "__main__":
    raise SystemExit(main())
