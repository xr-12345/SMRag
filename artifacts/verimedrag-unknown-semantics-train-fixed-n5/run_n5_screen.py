"""Phase 8E -- UNKNOWN semantic fix + protocol_fixed (train_fixed) N=5 screen.

Runs two joint configs over the frozen DDXPlus **validation** manifest (first 5
cases per disease of the frozen N=20 list, 245 cases) at noise {0.2, 0.3} x
seeds {2026, 2027, 2028}, matched 0/0 (rho_env=0, rho_model=0).

Configs (spec section 6):
  * joint_legacy_prior     -- default ChannelParameters().cue_priors (Phase 8C
                              baseline; byte-identical regression target)
  * joint_train_fixed_prior-- protocol_fixed_prior (cross-noise average, no cue
                              conditioning), the Phase 8D "train_fixed" prior.

Each config runs its OWN full dialogue on an identically-seeded patient, so both
diagnostic outcomes (Top-1/Brier/question counts) and reliability calibration
(p_wrong / p_mode / UNKNOWN rate) are compared.  ``p_wrong`` for a ``UNKNOWN``
first answer is written as an empty cell (``None``), and ``is_nonresponse`` is
recorded explicitly -- UNKNOWN rows never enter wrong-report ECE/Brier.

Red lines: validate split only; no test split; no N=20; no retraining; no RAG
change; no threshold sweep; no true state in the prediction path; no git commit;
writes only into this artifact directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

from powerful_medrag.action_value import nll_loss
from powerful_medrag.channel import (
    AnswerChannel,
    ReportMode,
    protocol_fixed_channel_parameters,
)
from powerful_medrag.decision import (
    ActionKind,
    PolicyTurn,
    ReliabilityAwareDialogueResult,
    ReliabilityAwarePolicyConfig,
)
from powerful_medrag.ddxplus import sample_balanced_ddxplus_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.joint_channel_policy import JointChannelBrierAuditPolicy
from powerful_medrag.joint_patient_simulator import JointStructuredPatientSimulator
from powerful_medrag.joint_reliability_belief import JointReliabilityBeliefTracker
from powerful_medrag.joint_report_channel import JointReportChannel, VerificationType
from powerful_medrag.reliability_experiment import _pilot_seed
from powerful_medrag.schema import UNKNOWN
from powerful_medrag.simulator import PatientProfile

OUT = Path(__file__).resolve().parent
MODEL_PATH = "data/ddxplus/model-full.json"
PATIENTS_PATH = "data/ddxplus/release_validate_patients.zip"
EVIDENCES_PATH = "data/ddxplus/release_evidences.json"
# Frozen N=20 manifest (first 5 per disease -> 245 cases), spec section 6.
MANIFEST_SOURCE = (
    "artifacts/verimedrag-rag-joint-gate-confirmation-n20/validation_case_manifest.csv"
)

NOISE_RATES = (0.2, 0.3)
SEEDS = (2026, 2027, 2028)
RHO_ENV = 0.0
RHO_MODEL = 0.0

LEGACY = "joint_legacy_prior"
TRAIN_FIXED = "joint_train_fixed_prior"
CONFIGS = (LEGACY, TRAIN_FIXED)


# --------------------------------------------------------------------------- #
# Prior config construction
# --------------------------------------------------------------------------- #
def build_config_channels() -> dict[str, JointReportChannel]:
    """The two inference channels.  Only the mode prior differs; the confusion
    ``rates`` and the re-ask persistence (rho_model=0) are shared."""
    return {
        LEGACY: JointReportChannel(AnswerChannel(), repeat_mode_persistence=RHO_MODEL),
        TRAIN_FIXED: JointReportChannel(
            AnswerChannel(protocol_fixed_channel_parameters()),
            repeat_mode_persistence=RHO_MODEL,
        ),
    }


def policy_config() -> ReliabilityAwarePolicyConfig:
    # Frozen Phase 8C joint config (matched 0/0): default thresholds, 15 turns.
    return ReliabilityAwarePolicyConfig(max_total_turns=15)


# --------------------------------------------------------------------------- #
# Model / cases
# --------------------------------------------------------------------------- #
def load_model() -> DiseaseStateModel:
    return DiseaseStateModel.load(MODEL_PATH)


def load_selected_cases(model: DiseaseStateModel, per_disease: int) -> list:
    askable = {k for k, s in model.specs.items() if s.askable}
    cases = sample_balanced_ddxplus_cases(
        PATIENTS_PATH, EVIDENCES_PATH,
        cases_per_disease=20, seed=2026, available_features=askable,
    )
    by_id = {c.case_id: c for c in cases}
    manifest: dict[str, list[str]] = {}
    with open(MANIFEST_SOURCE, newline="") as f:
        for row in csv.DictReader(f):
            manifest.setdefault(row["diagnosis"], []).append(row["case_id"])
    selected = [cid for d in sorted(manifest) for cid in manifest[d][:per_disease]]
    missing = [cid for cid in selected if cid not in by_id]
    if missing:
        raise ValueError(f"manifest case_ids not reproduced: {missing[:5]}")
    return [by_id[cid] for cid in selected]


# --------------------------------------------------------------------------- #
# One trajectory (true state enters ONLY here, never the policy)
# --------------------------------------------------------------------------- #
def run_joint_one(case, noise, seed, config_name, channel, model):
    patient = JointStructuredPatientSimulator(
        diagnosis=case.diagnosis,
        latent_states=case.states,
        model=model,
        profile=PatientProfile.from_noise_rate(noise),
        seed=_pilot_seed(seed, case.case_id, noise),
        rho_env=RHO_ENV,
        case_id=case.case_id,
    )
    tracker = JointReliabilityBeliefTracker(model, channel)
    policy = JointChannelBrierAuditPolicy(model, channel=channel, tracker=tracker)
    for obs in case.initial_observations:
        tracker.observe_single(obs)

    turns: list[PolicyTurn] = []
    reliability_rows: list[dict] = []
    verification_count = 0
    stop_reason = "total_turn_budget"
    uncertain_output = False

    t0 = time.perf_counter()
    while len(turns) < policy.config.max_total_turns:
        action = policy.choose_action()
        if action.kind is ActionKind.STOP:
            stop_reason = action.explanation
            uncertain_output = "uncertainty" in action.explanation
            break
        if action.kind is ActionKind.NEW:
            key = action.key
            observation, true_mode = patient.answer(key)
            tracker.observe_single(observation)
            true_state = patient.latent_states.get(key, UNKNOWN)
            true_wrong = int(
                true_state != UNKNOWN
                and observation.value != UNKNOWN
                and observation.value != true_state
            )
            p_wrong = tracker.p_wrong(key)
            reliability_rows.append(
                {
                    "case_id": case.case_id,
                    "diagnosis": case.diagnosis,
                    "config": config_name,
                    "noise_rate": noise,
                    "seed": seed,
                    "key": key.name,
                    "value": observation.value,
                    "certainty": observation.certainty.value,
                    "true_mode": true_mode.value,
                    "true_state": true_state,
                    "true_wrong": true_wrong,
                    "true_misreported": int(true_mode is ReportMode.MISREPORTED),
                    "is_nonresponse": int(tracker.is_nonresponse(key)),
                    "p_mode": round(tracker.p_mode_misreported(key), 8),
                    "p_wrong": "" if p_wrong is None else round(p_wrong, 8),
                }
            )
            turns.append(
                PolicyTurn(
                    index=len(turns) + 1,
                    action=action,
                    observation=observation,
                    true_report_mode=true_mode,
                    belief=dict(tracker.belief),
                )
            )
            continue

        key = action.key
        if key is None or key not in tracker.memory:
            raise RuntimeError("verification action has an invalid feature key")
        original = tracker.memory[key].original
        clarification, true_mode = patient.answer(key)
        true_state = patient.latent_states.get(key, UNKNOWN)
        original_wrong = (
            true_state != UNKNOWN
            and original.value != UNKNOWN
            and original.value != true_state
        )
        resolved_wrong = (
            true_state != UNKNOWN
            and clarification.value != UNKNOWN
            and clarification.value != true_state
        )
        tracker.observe_verification(key, clarification, VerificationType.REPEAT)
        verification_count += 1
        turns.append(
            PolicyTurn(
                index=len(turns) + 1,
                action=action,
                observation=clarification,
                true_report_mode=true_mode,
                belief=dict(tracker.belief),
                verification_changed_report=(
                    clarification.value != original.value
                    or clarification.certainty != original.certainty
                ),
                verification_was_unnecessary=not original_wrong,
                verification_resolved_wrong_report=original_wrong and not resolved_wrong,
            )
        )
    wall = time.perf_counter() - t0

    predicted = max(tracker.belief, key=tracker.belief.__getitem__)
    result = ReliabilityAwareDialogueResult(
        true_diagnosis=patient.diagnosis,
        predicted_diagnosis=predicted,
        belief=dict(tracker.belief),
        turns=tuple(turns),
        stop_reason=stop_reason,
        new_questions=sum(t.action.kind is ActionKind.NEW for t in turns),
        verification_questions=sum(t.action.kind is ActionKind.VERIFY for t in turns),
        uncertain_output=uncertain_output,
    )
    row = compute_outcomes(case, model, result, config_name, noise, seed)
    row["wall_clock"] = round(wall, 6)
    return row, reliability_rows


def compute_outcomes(case, model, result, config_name, noise, seed):
    ranking = sorted(result.belief, key=result.belief.__getitem__, reverse=True)
    predicted = ranking[0]
    brier = sum(
        (result.belief[d] - (1.0 if d == case.diagnosis else 0.0)) ** 2
        for d in model.diseases
    )
    nll = nll_loss(result.belief, case.diagnosis)
    confident_stop = ("posterior" in result.stop_reason) or (
        "confidence" in result.stop_reason
    )
    unnecessary = sum(t.verification_was_unnecessary for t in result.turns)
    resolved = sum(t.verification_resolved_wrong_report for t in result.turns)
    total_turns = result.new_questions + result.verification_questions
    action_seq = "".join(
        "N" if t.action.kind is ActionKind.NEW else
        "V" if t.action.kind is ActionKind.VERIFY else "S"
        for t in result.turns
    )

    return {
        "case_id": case.case_id,
        "diagnosis": case.diagnosis,
        "config": config_name,
        "noise_rate": noise,
        "seed": seed,
        "predicted_diagnosis": predicted,
        "correct_top1": int(predicted == case.diagnosis),
        "correct_top3": int(case.diagnosis in ranking[:3]),
        "confidence": round(result.belief[predicted], 8),
        "brier_score": round(brier, 8),
        "nll_score": round(nll, 8),
        "new_questions": result.new_questions,
        "verification_questions": result.verification_questions,
        "total_atomic_questions": total_turns,
        "unnecessary_verifications": unnecessary,
        "resolved_wrong_reports": resolved,
        "early_stop": int(confident_stop),
        "premature_stop": int(predicted != case.diagnosis and confident_stop),
        "uncertain_output": int(result.uncertain_output),
        "stop_reason": result.stop_reason,
        "action_seq": action_seq,
    }


# --------------------------------------------------------------------------- #
# Worker pool
# --------------------------------------------------------------------------- #
_WORKER = {}


def _init_worker(model, cases_by_id, channels):
    _WORKER["model"] = model
    _WORKER["cases_by_id"] = cases_by_id
    _WORKER["channels"] = channels


def _worker_task(task):
    case_id, noise, seed, config_name = task
    case = _WORKER["cases_by_id"][case_id]
    channel = _WORKER["channels"][config_name]
    return run_joint_one(case, noise, seed, config_name, channel, _WORKER["model"])


def write_csv(rows, path, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fields = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


OUTCOME_FIELDS = [
    "case_id", "diagnosis", "config", "noise_rate", "seed", "predicted_diagnosis",
    "correct_top1", "correct_top3", "confidence", "brier_score", "nll_score",
    "new_questions", "verification_questions", "total_atomic_questions",
    "unnecessary_verifications", "resolved_wrong_reports", "early_stop",
    "premature_stop", "uncertain_output", "stop_reason", "action_seq", "wall_clock",
]

RELIABILITY_FIELDS = [
    "case_id", "diagnosis", "config", "noise_rate", "seed", "key", "value",
    "certainty", "true_mode", "true_state", "true_wrong", "true_misreported",
    "is_nonresponse", "p_mode", "p_wrong",
]


def _build_tasks(cases, noises, seeds):
    return [
        (c.case_id, noise, seed, config_name)
        for c in cases
        for noise in noises
        for seed in seeds
        for config_name in CONFIGS
    ]


def _run_grid(cases, noises, seeds, workers):
    cases_by_id = {c.case_id: c for c in cases}
    channels = build_config_channels()
    tasks = _build_tasks(cases, noises, seeds)
    print(f"run: {len(tasks)} trajectories over {workers} workers", flush=True)
    ctx = mp.get_context("fork")
    outcome_rows, reliability_rows = [], []
    walls = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(_WORKER["model"], cases_by_id, channels)) as pool:
        for i, (row, reliability) in enumerate(
            pool.imap_unordered(_worker_task, tasks, chunksize=4)
        ):
            outcome_rows.append(row)
            reliability_rows.extend(reliability)
            walls.append(row["wall_clock"])
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(tasks)} done "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    write_csv(outcome_rows, OUT / "case_outcomes.csv", OUTCOME_FIELDS)
    write_csv(reliability_rows, OUT / "reliability_predictions.csv",
              RELIABILITY_FIELDS)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    mean_wall = sum(walls) / len(walls) if walls else 0.0
    print(f"done in {elapsed:.0f}s; {len(outcome_rows)} case rows, "
          f"{len(reliability_rows)} reliability rows; mean wall {mean_wall:.2f}s; "
          f"peak {peak_mb:.0f}MB", flush=True)
    write_csv(
        [
            {"metric": "trajectories", "value": len(tasks)},
            {"metric": "workers", "value": workers},
            {"metric": "elapsed_seconds", "value": round(elapsed, 1)},
            {"metric": "mean_wall_seconds_per_dialogue",
             "value": round(mean_wall, 3)},
            {"metric": "peak_rss_mb", "value": round(peak_mb, 1)},
        ],
        OUT / "runtime_profile.csv",
    )


def write_config(cases, per_disease):
    config = {
        "phase": "unknown-semantics-train-fixed-n5",
        "scope": "UNKNOWN semantic fix + protocol_fixed (train_fixed) prior "
                 "N=5 validation screen (matched 0/0)",
        "configs": {
            LEGACY: "default ChannelParameters().cue_priors (Phase 8C baseline)",
            TRAIN_FIXED: "protocol_fixed_prior (cross-noise average, no cue "
                         "conditioning) = Phase 8D train_fixed",
        },
        "scenario": {"rho_env": RHO_ENV, "rho_model": RHO_MODEL},
        "cases": len(cases),
        "cases_per_disease": per_disease,
        "noise_rates": list(NOISE_RATES),
        "seeds": list(SEEDS),
        "max_total_turns": 15,
        "manifest_source": MANIFEST_SOURCE,
        "protocol_fixed_prior": {
            "CERTAIN": 0.75, "UNCERTAIN": 0.10,
            "UNKNOWN": 0.075, "MISREPORTED": 0.075,
        },
        "frozen_params": {
            "posterior_threshold": 0.85,
            "posterior_margin_threshold": 0.70,
            "minimum_action_utility": 0.03,
            "suspicious_report_threshold": 0.05,
            "new_question_cost_weight": 0.01,
            "verification_cost": 0.03,
            "maximum_verifications": 1,
            "verification_audit_threshold": 0.03,
            "verification_advantage_margin": 0.03,
            "max_total_turns": 15,
        },
        "unknown_semantics": {
            "p_wrong_for_UNKNOWN": "None (undefined, not 1.0, not 0.0)",
            "nonresponse_signal": "is_nonresponse = 1[Y=UNKNOWN]",
            "wrong_report_metric_scope": "explicit answers only (UNKNOWN excluded)",
        },
        "red_lines": [
            "no test split", "no N=20", "no retraining", "no RAG change",
            "no threshold sweep", "no true state in prediction",
            "no git commit",
        ],
    }
    (OUT / "CONFIG.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = load_model()
    print(f"model loaded ({time.perf_counter()-t0:.2f}s): {len(model.diseases)} diseases",
          flush=True)
    _WORKER["model"] = model

    if args.mode == "smoke":
        cases = load_selected_cases(model, per_disease=1)
        print(f"smoke: {len(cases)} cases (1/disease)", flush=True)
        write_config(cases, per_disease=1)
        _run_grid(cases, (0.3,), (2026,), args.workers)
        return 0

    cases = load_selected_cases(model, per_disease=5)
    print(f"full: {len(cases)} cases (5/disease)", flush=True)
    write_config(cases, per_disease=5)
    _run_grid(cases, NOISE_RATES, SEEDS, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
