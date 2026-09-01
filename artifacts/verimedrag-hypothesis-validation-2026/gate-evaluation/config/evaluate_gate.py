"""Prompt #7 independent learned-gate evaluation.

Loads the fitted learned gate, then scores every VALIDATION gate event under five
methods and reports discrimination (AUROC, AUPRC) and calibration (Brier, ECE,
calibration slope/intercept) across four labels and six conditions, with
case-clustered bootstrap 95% CIs.

Method score / probability semantics
------------------------------------
  learned_gate          learned logistic probability (both score and prob)
  heuristic_gate        frozen HeuristicMisreportGate probability (event.gate_probability)
  calibrated_surprisal  score = raw surprisal; prob = univariate Platt fit on TRAIN
  fixed_prior           train prevalence of the label (constant; score == prob)
  deterministic_random  frozen-seed hash of the event key (uniform; score == prob)

AUROC/AUPRC use the ranking score; Brier/ECE/slope use the probability. For
calibrated_surprisal these differ; for every other method they coincide.

No test split is touched. Validation events are never used for fitting: the
learned gate, the surprisal calibrator, the fixed prior, and the exploratory
decision_sensitive_wrong q75 are all frozen from TRAIN events only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path("/Users/xr-12345/Desktop/SafeMedRAG")
sys.path.insert(0, str(REPO / "src"))

from powerful_medrag.gate_analysis import load_gate_events_csv  # noqa: E402
from powerful_medrag.gate_learning import load_learned_gate  # noqa: E402
from powerful_medrag.gating import LEARNED_GATE_FEATURES  # noqa: E402

ROOT = REPO / "artifacts/verimedrag-hypothesis-validation-2026/gate-evaluation"
TRAIN_EVENTS = ROOT / "train-events" / "train_gate_events.csv"
VAL_EVENTS = ROOT / "validation-events" / "validation_gate_events.csv"
GATE_JSON = ROOT / "model" / "learned_gate.json"
FROZEN_MANIFEST = (
    REPO
    / "artifacts/verimedrag-hypothesis-validation-2026/formal/config"
    / "validation_case_manifest_with_duplicate_flag.csv"
)
EVAL_DIR = ROOT / "evaluation"

PRIMARY = "harmful_misreport"
AUX_LABELS = ("wrong_report", "latent_misreport")
EXPLORATORY = "decision_sensitive_wrong"
LABELS = (PRIMARY,) + AUX_LABELS + (EXPLORATORY,)

METHODS = (
    "learned_gate",
    "heuristic_gate",
    "calibrated_surprisal",
    "fixed_prior",
    "deterministic_random",
)
CONDITIONS = ("pooled_noisy", "all", 0.0, 0.1, 0.2, 0.3)
NOISY = (0.1, 0.2, 0.3)

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 2026
RANDOM_SEED = 2026
ECE_BINS = 10


# --------------------------------------------------------------------------- #
# Metrics (numpy, tie-aware where relevant)
# --------------------------------------------------------------------------- #
def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n_pos = float(labels.sum())
    n_neg = float(len(labels)) - n_pos
    if n_pos == 0.0 or n_neg == 0.0:
        return math.nan
    ranks = rankdata(scores, method="average")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float:
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


def _brier(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def _ece(labels: np.ndarray, probs: np.ndarray, bins: int = ECE_BINS) -> float:
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


def _calibration_slope_intercept(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
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
# Helpers
# --------------------------------------------------------------------------- #
def _features(event) -> dict[str, float]:
    return {feature: float(getattr(event, feature)) for feature in LEARNED_GATE_FEATURES}


def _event_key(event) -> str:
    return f"{event.case_id}|{event.seed}|{event.noise_rate}|{event.turn}|{event.feature}"


def _deterministic_random(event, seed: int = RANDOM_SEED) -> float:
    digest = hashlib.blake2b(
        f"gate-random|{seed}|{_event_key(event)}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / (2**64)


def _fit_univariate_surprisal(train_events, label_fn) -> tuple[float, float]:
    """Platt-style logistic fit of P(label | surprisal) on TRAIN only.

    Returns (intercept, slope) in RAW surprisal space: z = intercept + slope * surprisal.
    """
    raw = np.asarray([float(e.surprisal) for e in train_events], dtype=float)
    y = np.asarray([int(label_fn(e)) for e in train_events], dtype=float)
    mu = float(raw.mean())
    sigma = float(raw.std() + 1e-6)
    x = (raw - mu) / sigma
    intercept = math.log(max(y.mean(), 1e-6) / max(1.0 - y.mean(), 1e-6))
    slope_std = 0.0
    for iteration in range(2000):
        z = intercept + slope_std * x
        prob = 1.0 / (1.0 + np.exp(-z))
        residual = prob - y
        rate = 0.05 / math.sqrt(1.0 + iteration / 100.0)
        intercept -= rate * float(residual.mean())
        slope_std -= rate * float((residual * x).mean())
    slope_raw = slope_std / sigma
    intercept_raw = intercept - slope_raw * mu
    return intercept_raw, slope_raw


def _apply_surprisal_calibrator(intercept: float, slope: float, surprisal: float) -> float:
    z = intercept + slope * surprisal
    return 1.0 / (1.0 + math.exp(-z))


def _condition_mask(records, condition) -> np.ndarray:
    noise = np.asarray([r["noise_rate"] for r in records], dtype=float)
    if condition == "pooled_noisy":
        return np.isin(noise, list(NOISY))
    if condition == "all":
        return np.ones(len(records), dtype=bool)
    return noise == condition


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    train_events = load_gate_events_csv(TRAIN_EVENTS)
    val_events = load_gate_events_csv(VAL_EVENTS)
    gate = load_learned_gate(GATE_JSON)
    print(f"train events: {len(train_events)}  validation events: {len(val_events)}")

    # Frozen train-derived quantities ------------------------------------- #
    # exploratory q75: 75th percentile of diagnostic_impact over train wrong_report=1
    wrong_impacts = sorted(
        e.diagnostic_impact for e in train_events if e.wrong_report == 1
    )
    q75 = float(np.percentile(wrong_impacts, 75)) if wrong_impacts else math.nan

    def label_value(event, label: str) -> int:
        if label == EXPLORATORY:
            return int(event.wrong_report == 1 and event.diagnostic_impact >= q75)
        return int(getattr(event, label))

    train_prevalence = {
        label: float(np.mean([label_value(e, label) for e in train_events]))
        for label in LABELS
    }
    print(f"train prevalence (harmful_misreport): {train_prevalence[PRIMARY]:.6f}")
    print(f"exploratory q75 (diagnostic_impact, train wrong_report): {q75:.6f}")

    # surprisal calibrator per label (train only)
    surprisal_calibrators = {
        label: _fit_univariate_surprisal(
            train_events, lambda e, l=label: label_value(e, l)
        )
        for label in LABELS
    }

    # Build per-event records --------------------------------------------- #
    records: list[dict] = []
    for event in val_events:
        features = _features(event)
        surprisal = float(event.surprisal)
        learned = gate.probability_from_features(features)
        heuristic = float(event.gate_probability)
        fixed = train_prevalence[PRIMARY]
        rng = _deterministic_random(event)
        rec = {
            "case_id": event.case_id,
            "diagnosis": event.diagnosis,
            "seed": event.seed,
            "noise_rate": event.noise_rate,
            "turn": event.turn,
            "feature": event.feature,
            PRIMARY: label_value(event, PRIMARY),
            "wrong_report": label_value(event, "wrong_report"),
            "latent_misreport": label_value(event, "latent_misreport"),
            EXPLORATORY: label_value(event, EXPLORATORY),
            "learned_gate": learned,
            "heuristic_gate": heuristic,
            "calibrated_surprisal_score": surprisal,
            "calibrated_surprisal": _apply_surprisal_calibrator(
                *surprisal_calibrators[PRIMARY], surprisal
            ),
            "fixed_prior": fixed,
            "deterministic_random": rng,
        }
        records.append(rec)

    # predictions CSV ------------------------------------------------------ #
    pred_fields = [
        "case_id", "diagnosis", "seed", "noise_rate", "turn", "feature",
        PRIMARY, "wrong_report", "latent_misreport", EXPLORATORY,
        "learned_gate", "heuristic_gate", "calibrated_surprisal_score",
        "calibrated_surprisal", "fixed_prior", "deterministic_random",
    ]
    with (EVAL_DIR / "validation_gate_predictions.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote validation_gate_predictions.csv ({len(records)} rows)")

    # Metrics ----------------------------------------------------------------- #
    labels = np.asarray([rec[PRIMARY] for rec in records], dtype=float)
    wrong = np.asarray([rec["wrong_report"] for rec in records], dtype=float)
    latent = np.asarray([rec["latent_misreport"] for rec in records], dtype=float)
    exploratory = np.asarray([rec[EXPLORATORY] for rec in records], dtype=float)
    label_arrays = {PRIMARY: labels, "wrong_report": wrong,
                    "latent_misreport": latent, EXPLORATORY: exploratory}
    case_ids = np.asarray([rec["case_id"] for rec in records], dtype=object)
    noise = np.asarray([rec["noise_rate"] for rec in records], dtype=float)

    def score_array(method: str, label: str) -> np.ndarray:
        if method == "learned_gate":
            return np.asarray([rec["learned_gate"] for rec in records], dtype=float)
        if method == "heuristic_gate":
            return np.asarray([rec["heuristic_gate"] for rec in records], dtype=float)
        if method == "calibrated_surprisal":
            return np.asarray([rec["calibrated_surprisal_score"] for rec in records], dtype=float)
        if method == "fixed_prior":
            return np.full(len(records), train_prevalence[label], dtype=float)
        if method == "deterministic_random":
            return np.asarray([rec["deterministic_random"] for rec in records], dtype=float)
        raise ValueError(method)

    def prob_array(method: str, label: str) -> np.ndarray:
        if method == "calibrated_surprisal":
            return np.asarray([_apply_surprisal_calibrator(
                *surprisal_calibrators[label], rec["calibrated_surprisal_score"]
            ) for rec in records], dtype=float)
        if method == "fixed_prior":
            return np.full(len(records), train_prevalence[label], dtype=float)
        return score_array(method, label)

    metric_rows: list[dict] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for label in LABELS:
        y = label_arrays[label]
        for condition in CONDITIONS:
            mask = _condition_mask(records, condition)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            y_sub = y[idx]
            prevalence = float(y_sub.mean())
            # unique case ids in this condition (for case-clustered bootstrap)
            cond_cases = np.unique(case_ids[idx])
            for method in METHODS:
                scores = score_array(method, label)[idx]
                probs = prob_array(method, label)[idx]
                auroc = _auroc(y_sub, scores)
                auprc = _auprc(y_sub, scores)
                brier = _brier(y_sub, probs)
                ece = _ece(y_sub, probs)
                slope, intercept = _calibration_slope_intercept(y_sub, probs)

                # case-clustered bootstrap
                auroc_b = []; auprc_b = []; brier_b = []; ece_b = []
                case_event_idx: dict[str, list[int]] = defaultdict(list)
                for rel_j, abs_j in enumerate(idx):
                    case_event_idx[case_ids[abs_j]].append(rel_j)
                case_list = list(case_event_idx.keys())
                for _ in range(BOOTSTRAP_SAMPLES):
                    sampled = rng.choice(case_list, size=len(case_list), replace=True)
                    boot_idx = np.concatenate([case_event_idx[c] for c in sampled])
                    auroc_b.append(_auroc(y_sub[boot_idx], scores[boot_idx]))
                    auprc_b.append(_auprc(y_sub[boot_idx], scores[boot_idx]))
                    brier_b.append(_brier(y_sub[boot_idx], probs[boot_idx]))
                    ece_b.append(_ece(y_sub[boot_idx], probs[boot_idx]))
                metric_rows.append({
                    "label": label,
                    "condition": str(condition),
                    "method": method,
                    "events": len(idx),
                    "cases": len(cond_cases),
                    "prevalence": f"{prevalence:.6f}",
                    "AUROC": f"{auroc:.6f}",
                    "AUROC_ci_lo": f"{np.nanpercentile(auroc_b, 2.5):.6f}",
                    "AUROC_ci_hi": f"{np.nanpercentile(auroc_b, 97.5):.6f}",
                    "AUPRC": f"{auprc:.6f}",
                    "AUPRC_ci_lo": f"{np.nanpercentile(auprc_b, 2.5):.6f}",
                    "AUPRC_ci_hi": f"{np.nanpercentile(auprc_b, 97.5):.6f}",
                    "Brier": f"{brier:.6f}",
                    "Brier_ci_lo": f"{np.nanpercentile(brier_b, 2.5):.6f}",
                    "Brier_ci_hi": f"{np.nanpercentile(brier_b, 97.5):.6f}",
                    "ECE": f"{ece:.6f}",
                    "ECE_ci_lo": f"{np.nanpercentile(ece_b, 2.5):.6f}",
                    "ECE_ci_hi": f"{np.nanpercentile(ece_b, 97.5):.6f}",
                    "calibration_slope": f"{slope:.6f}",
                    "calibration_intercept": f"{intercept:.6f}",
                })

    metric_fields = [
        "label", "condition", "method", "events", "cases", "prevalence",
        "AUROC", "AUROC_ci_lo", "AUROC_ci_hi", "AUPRC", "AUPRC_ci_lo", "AUPRC_ci_hi",
        "Brier", "Brier_ci_lo", "Brier_ci_hi", "ECE", "ECE_ci_lo", "ECE_ci_hi",
        "calibration_slope", "calibration_intercept",
    ]
    with (EVAL_DIR / "validation_gate_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"wrote validation_gate_metrics.csv ({len(metric_rows)} rows)")

    # feature coefficients --------------------------------------------------- #
    with (EVAL_DIR / "feature_coefficients.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["feature", "coefficient", "mean", "scale"])
        for feature in LEARNED_GATE_FEATURES:
            writer.writerow([
                feature,
                gate.coefficients.get(feature, 0.0),
                gate.feature_means.get(feature, 0.0),
                gate.feature_scales.get(feature, 1.0),
            ])
        writer.writerow(["intercept", gate.intercept, "", ""])
        writer.writerow(["calibration_intercept", gate.calibration_intercept, "", ""])
        writer.writerow(["calibration_slope", gate.calibration_slope, "", ""])
    print("wrote feature_coefficients.csv")

    # duplicate sensitivity -------------------------------------------------- #
    dup = {r["case_id"] for r in csv.DictReader(FROZEN_MANIFEST.open())
           if r["is_exact_train_duplicate"] == "1"}
    keep = np.asarray([c not in dup for c in case_ids], dtype=bool)
    sensitivity_rows: list[dict] = []
    for label in (PRIMARY, "wrong_report", "latent_misreport"):
        y = label_arrays[label]
        cond = "pooled_noisy"
        mask = _condition_mask(records, cond)
        for scope_name, scope_mask in (("full", mask), ("exclude_duplicates", mask & keep)):
            idx = np.where(scope_mask)[0]
            for method in ("learned_gate", "heuristic_gate", "calibrated_surprisal"):
                scores = score_array(method, label)[idx]
                probs = prob_array(method, label)[idx]
                sensitivity_rows.append({
                    "label": label,
                    "condition": cond,
                    "scope": scope_name,
                    "method": method,
                    "events": len(idx),
                    "AUROC": f"{_auroc(y[idx], scores):.6f}",
                    "AUPRC": f"{_auprc(y[idx], scores):.6f}",
                    "Brier": f"{_brier(y[idx], probs):.6f}",
                    "ECE": f"{_ece(y[idx], probs):.6f}",
                })
    sens_fields = ["label", "condition", "scope", "method", "events",
                   "AUROC", "AUPRC", "Brier", "ECE"]
    with (EVAL_DIR / "duplicate_sensitivity.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sens_fields)
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    print(f"wrote duplicate_sensitivity.csv (excluded {len(dup)} duplicate cases)")

    # calibration curves ----------------------------------------------------- #
    _plot_calibration_curves(records, label_arrays)

    print("EVALUATION COMPLETE")


def _plot_calibration_curves(records: list[dict], label_arrays: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noise = np.asarray([rec["noise_rate"] for rec in records], dtype=float)
    mask = np.isin(noise, list(NOISY))
    y = label_arrays[PRIMARY][mask]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    method_probs = {
        "learned_gate": np.asarray([rec["learned_gate"] for rec in records], dtype=float)[mask],
        "heuristic_gate": np.asarray([rec["heuristic_gate"] for rec in records], dtype=float)[mask],
        "calibrated_surprisal": np.asarray([rec["calibrated_surprisal"] for rec in records], dtype=float)[mask],
    }
    for axis, (name, probs) in zip(axes, method_probs.items()):
        bins = np.linspace(0, 1, ECE_BINS + 1)
        centers, fractions, counts = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (probs >= lo) & (probs < hi)
            if sel.sum() == 0:
                continue
            centers.append((lo + hi) / 2)
            fractions.append(float(y[sel].mean()))
            counts.append(int(sel.sum()))
        axis.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        axis.plot(centers, fractions, "o-", markersize=5, color="#1f77b4")
        axis.set_title(f"{name}\n(n={int(mask.sum())}, pooled noisy)")
        axis.set_xlabel("predicted probability")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("empirical positive fraction")
    fig.suptitle("Reliability curves — primary label (harmful_misreport)")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "calibration_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote calibration_curves.png")


if __name__ == "__main__":
    main()
