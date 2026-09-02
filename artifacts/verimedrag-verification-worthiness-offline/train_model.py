"""Phase 4 -- train the two-head learned worthiness model (train split only).

For each family (Ridge+Logistic, HGB+HGB) and rag_mode (real / none / shuffled),
trains gain + harm heads with grouped CV hyperparameter selection, selects
``tau_harm`` on train OOF, and reports train OOF Spearman / harm AUC.

Family is chosen on **real-RAG** train OOF Spearman (primary) then harm AUC.
The frozen model and the no-RAG / shuffled-RAG ablations are then trained with
that family, each with its own CV-selected hyperparams + tau_harm, and saved to
``models/``.  No validation data is touched here (red line: no validation-driven
retuning).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np

from powerful_medrag import verification_worthiness as vw

OUT = Path(__file__).resolve().parent
MODELS = OUT / "models"
MODELS.mkdir(exist_ok=True)


def load_rows():
    import csv as _csv
    return list(_csv.DictReader(open(OUT / "train_features.csv", newline="", encoding="utf-8")))


def build_matrices(rows, rag_mode):
    X = vw.build_feature_matrix(rows, rag_mode=rag_mode)
    labels = [vw.derive_labels(r) for r in rows]
    y_gain = np.asarray([l["gain"] for l in labels], dtype=float)
    y_harm = np.asarray([l["harm"] for l in labels], dtype=float)
    value = np.asarray([l["value"] for l in labels], dtype=float)
    groups = np.asarray([r["case_id"] for r in rows])
    return X, y_gain, y_harm, value, groups


def train_one(rows, rag_mode, gain_kind, harm_kind):
    X, y_gain, y_harm, value, groups = build_matrices(rows, rag_mode)
    gain_model, gain_params, gain_oof = vw.train_gain_head(X, y_gain, groups, gain_kind)
    harm_model, harm_params, harm_oof = vw.train_harm_head(X, y_harm, groups, harm_kind)
    tau = vw.select_tau_harm(gain_oof, harm_oof, value)
    oof_spearman = vw._spearman(gain_oof, y_gain)
    oof_auc = vw._roc_auc(y_harm, harm_oof)
    model = vw.LearnedWorthinessModel(
        gain_model=gain_model, harm_model=harm_model, tau_harm=tau,
        gain_kind=gain_kind, harm_kind=harm_kind, n_features=X.shape[1],
        rag_mode=rag_mode,
    )
    return model, {
        "rag_mode": rag_mode, "gain_kind": gain_kind, "harm_kind": harm_kind,
        "n_features": X.shape[1], "n_samples": len(rows),
        "n_cases": len(set(groups)), "gain_oof_spearman": round(oof_spearman, 4),
        "harm_oof_auc": round(oof_auc, 4), "tau_harm": round(tau, 4),
        "gain_params": gain_params, "harm_params": harm_params,
    }


def main() -> int:
    rows = load_rows()
    print(f"train rows: {len(rows)}")

    # 1. pick family on real-RAG.
    family_rows = []
    for gain_kind in vw.GAIN_KINDS:
        harm_kind = "logistic" if gain_kind == "ridge" else "hgb"
        model, info = train_one(rows, "real", gain_kind, harm_kind)
        family_rows.append(info)
        print(f"family {gain_kind}+{harm_kind}: spearman={info['gain_oof_spearman']} "
              f"auc={info['harm_oof_auc']} tau={info['tau_harm']}")
    best_family = max(family_rows, key=lambda d: (d["gain_oof_spearman"], d["harm_oof_auc"]))
    gain_kind, harm_kind = best_family["gain_kind"], best_family["harm_kind"]
    print(f"chosen family: {gain_kind}+{harm_kind}")

    # 2. train frozen real-RAG + no-RAG + shuffled-RAG with the chosen family.
    results = []
    for rag_mode in ("real", "none", "shuffled"):
        model, info = train_one(rows, rag_mode, gain_kind, harm_kind)
        joblib.dump(model, MODELS / f"worthiness_{rag_mode}.pkl")
        results.append(info)
        print(f"{rag_mode:8s}: spearman={info['gain_oof_spearman']} "
              f"auc={info['harm_oof_auc']} tau={info['tau_harm']}")

    # write train_cv_results.csv (family comparison + per-rag_mode)
    with (OUT / "train_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(family_rows[0].keys()))
        w.writeheader()
        w.writerows(family_rows)
        w.writerows(results)

    meta = {
        "chosen_family": {"gain_kind": gain_kind, "harm_kind": harm_kind},
        "frozen_rag_mode": "real",
        "family_comparison": family_rows,
        "rag_ablation_train": results,
    }
    (OUT / "train_model_selection.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("wrote train_cv_results.csv, train_model_selection.json, models/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
