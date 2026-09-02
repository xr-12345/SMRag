"""Phase 4 -- prepare the train feature/label set from the existing audit CSV.

Reads ``action_value_samples.csv`` (VerifyOld rows only), selects the deployable
label-free columns + realized labels, and writes ``train_features.csv`` together
with ``feature_schema.json`` and ``CONFIG.json``.  No model is trained here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from powerful_medrag import verification_worthiness as vw

OUT = Path(__file__).resolve().parent
SAMPLE = Path("artifacts/verimedrag-action-value-audit/action_value_samples.csv").resolve()

# Identical schema for train_features.csv and validation_features.csv.
FEATURE_CSV_COLUMNS = [
    "case_id", "diagnosis", "noise_rate", "state_kind", "state_index",
    "turn_index", "n_reports",
    "retrospective_error_prob", "retrieval_impact", "heuristic_verify_utility",
    "error_prob_times_impact", "current_risk",
    "v_bayes", "v_real", "gross_brier_reduction",
    "top1_before_correct", "correct_to_wrong",
]

CONFIG = {
    "phase": "verification-worthiness-offline",
    "scope": "learn VerifyOld clinical value only; AskNew EIG unchanged",
    "train_source": str(SAMPLE),
    "train_cases": 245,
    "train_cases_per_disease": 5,
    "train_seed": 3031,
    "train_noise_rates": [0.2, 0.3],
    "validation_cases": 735,
    "validation_cases_per_disease": 20,
    "validation_case_sample_seed": 2026,
    "validation_seeds": [2027, 2028],
    "validation_noise_rates": [0.2, 0.3],
    "n_states": 3,
    "n_rollouts": 8,
    "top_k_asknew": 5,
    "max_verify": 5,
    "c_new": 0.03,
    "c_verify": 0.03,
    "policy": {
        "posterior_threshold": 0.85,
        "posterior_margin_threshold": 0.70,
        "minimum_action_utility": 0.08,
        "suspicious_report_threshold": 0.05,
        "verification_cost": 0.03,
        "max_total_turns": 15,
        "retrieval_mode": "dynamic_rag",
        "retrieval_gate_mode": "joint_gate",
        "retrieval_impact_weight": 1.0,
    },
    "features": {"base": list(vw.BASE_FEATURES), "rag": list(vw.RAG_FEATURES)},
    "forbidden_fields": sorted(vw.FORBIDDEN_FIELDS),
    "labels": {
        "gain": "gross_brier_reduction",
        "value": "v_real",
        "harm": "1[correct_to_wrong > 0]",
        "benefit": "1[v_real > 0]",
    },
    "baselines": list(vw.BASELINES),
    "models": {
        "gain_kinds": list(vw.GAIN_KINDS),
        "harm_kinds": list(vw.HARM_KINDS),
        "gain_grid": vw.GAIN_GRID,
        "harm_grid": vw.HARM_GRID,
    },
    "red_lines": [
        "no test split", "no online policy integration", "no AskNew redesign",
        "no EIG change", "no latent/true-disease/noise as features",
        "no oracle correction", "no validation-driven retuning",
        "no LLM/dense-retriever/SFT/RL", "no preregistered-criteria change",
        "no git commit", "validation is not a clinical result",
    ],
}


def main() -> int:
    rows = [
        r for r in csv.DictReader(open(SAMPLE, newline="", encoding="utf-8"))
        if r["action_kind"] == "verify"
    ]
    selected = [{k: r[k] for k in FEATURE_CSV_COLUMNS} for r in rows]

    # sanity: every selected row must yield a valid feature vector + labels
    for r in selected:
        vw.extract_feature_vector(r)
        vw.derive_labels(r)

    out_path = OUT / "train_features.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_CSV_COLUMNS)
        w.writeheader()
        w.writerows(selected)

    (OUT / "feature_schema.json").write_text(
        json.dumps(vw.feature_schema(), indent=2), encoding="utf-8"
    )
    (OUT / "CONFIG.json").write_text(
        json.dumps(CONFIG, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_path} with {len(selected)} rows")
    print("wrote feature_schema.json, CONFIG.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
