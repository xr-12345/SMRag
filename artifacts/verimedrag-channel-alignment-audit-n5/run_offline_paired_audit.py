"""Phase 8D -- offline paired audit of p_wrong / p_mode under four mode priors.

Runs the matched 0/0 joint-channel dialogue (rho_env=0, rho_model=0) over the
frozen DDXPlus validation manifest (first 5 cases/disease, 245 cases) at noise
{0.2, 0.3} x seeds {2026, 2027, 2028} = 1470 trajectories.  The dialogue
trajectory (which features are asked / verified / when to stop) is decided by the
**legacy** config only; the other three configs recompute p_wrong / p_mode
**offline on the identical observation history** -- a strict paired comparison.

Configs:
  legacy           -- default ChannelParameters().cue_priors (Phase 8C baseline)
  oracle           -- env true P(E|cue) from from_noise_rate(noise) (privileged)
  train_fixed      -- single cross-noise-average prior (no cue conditioning)
  cue_conditioned  -- env cue mechanism applied to the cross-noise-average prior

Labels (spec section 3, separated):
  wrong_report     = 1[Z_i != Y_i]        (content mismatch)
  latent_misreport = 1[E_i = MISREPORTED] (mode misreport)
  harmful_misreport= wrong_report AND latent_misreport

Red lines: validate split only; no test split; no N=20; no policy modification;
no online comparison; priors are parameterized (cross-noise average), never fit
to validation/test labels; prediction path never reads true state / noise / mode
chain; no git commit; writes only into this artifact directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

from powerful_medrag.channel import AnswerChannel, ChannelParameters, ReportMode
from powerful_medrag.decision import ActionKind
from powerful_medrag.ddxplus import sample_balanced_ddxplus_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.joint_channel_policy import JointChannelBrierAuditPolicy
from powerful_medrag.joint_patient_simulator import JointStructuredPatientSimulator
from powerful_medrag.joint_reliability_belief import JointReliabilityBeliefTracker
from powerful_medrag.joint_report_channel import JointReportChannel, VerificationType
from powerful_medrag.reliability_experiment import _pilot_seed
from powerful_medrag.schema import UNKNOWN, CertaintyCue
from powerful_medrag.simulator import PatientProfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import channel_alignment_audit as caa  # noqa: E402

OUT = Path(__file__).resolve().parent
MODEL_PATH = "data/ddxplus/model-full.json"
PATIENTS_PATH = "data/ddxplus/release_validate_patients.zip"
EVIDENCES_PATH = "data/ddxplus/release_evidences.json"
MANIFEST_SOURCE = (
    "artifacts/verimedrag-rag-joint-gate-confirmation-n20/validation_case_manifest.csv"
)

NOISE_RATES = (0.2, 0.3)
SEEDS = (2026, 2027, 2028)
CONFIG_NAMES = ("legacy", "oracle", "train_fixed", "cue_conditioned")


# --------------------------------------------------------------------------- #
# Prior config construction
# --------------------------------------------------------------------------- #


def fixed_cross_noise_prior() -> dict[ReportMode, float]:
    """Average from_noise_rate(n) over the experiment noise set (cross-noise avg)."""
    priors = [
        dict(PatientProfile.from_noise_rate(n).mode_prior) for n in NOISE_RATES
    ]
    return {
        mode: sum(p[mode] for p in priors) / len(priors)
        for mode in caa.MODES
    }


def build_config_channels(noise: float) -> dict[str, JointReportChannel]:
    rates = ChannelParameters().rates
    env_prior = dict(PatientProfile.from_noise_rate(noise).mode_prior)
    fixed_prior = fixed_cross_noise_prior()

    def make(cue_priors=None) -> JointReportChannel:
        channel = (
            AnswerChannel()
            if cue_priors is None
            else AnswerChannel(ChannelParameters(cue_priors=cue_priors))
        )
        return JointReportChannel(channel, repeat_mode_persistence=0.0)

    return {
        "legacy": make(),
        "oracle": make(caa.env_cue_conditioned_prior(env_prior, rates)),
        "train_fixed": make({cue: dict(fixed_prior) for cue in caa.CUES}),
        "cue_conditioned": make(
            caa.env_cue_conditioned_prior(fixed_prior, rates)
        ),
    }


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
# One trajectory: legacy dialogue + 4-config replay
# --------------------------------------------------------------------------- #


def run_paired_one(case, noise, seed, model) -> list[dict]:
    config_channels = build_config_channels(noise)
    legacy_channel = config_channels["legacy"]

    patient = JointStructuredPatientSimulator(
        diagnosis=case.diagnosis,
        latent_states=case.states,
        model=model,
        profile=PatientProfile.from_noise_rate(noise),
        seed=_pilot_seed(seed, case.case_id, noise),
        rho_env=0.0,
        case_id=case.case_id,
    )
    tracker = JointReliabilityBeliefTracker(model, legacy_channel)
    policy = JointChannelBrierAuditPolicy(
        model, channel=legacy_channel, tracker=tracker
    )
    for obs in case.initial_observations:
        tracker.observe_single(obs)

    events: list[tuple] = []  # ("single", Observation) | ("verify", key, Observation)
    features: list[dict] = []  # asked-feature records, in first-answer order

    while len(events) < policy.config.max_total_turns:
        action = policy.choose_action()
        if action.kind is ActionKind.STOP:
            break
        if action.kind is ActionKind.NEW:
            key = action.key
            observation, true_mode = patient.answer(key)
            tracker.observe_single(observation)
            events.append(("single", observation))
            features.append(
                {
                    "key": key,
                    "original": observation,
                    "true_state": patient.latent_states.get(key, UNKNOWN),
                    "true_mode": true_mode,
                    "verified": False,
                    "verification": None,
                }
            )
            continue
        # VerifyOld
        key = action.key
        clarification, true_mode = patient.answer(key)
        tracker.observe_verification(key, clarification, VerificationType.REPEAT)
        events.append(("verify", key, clarification))
        for f in features:
            if f["key"] == key:
                f["verified"] = True
                f["verification"] = clarification

    # Replay each config on the identical observation history; snapshot p_wrong /
    # p_mode at the "first" time point (right after the first answer) and the
    # "reask" time point (right after verification, for verified features only).
    snapshots: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for config_name, channel in config_channels.items():
        t = JointReliabilityBeliefTracker(model, channel)
        for obs in case.initial_observations:
            t.observe_single(obs)
        for event in events:
            if event[0] == "single":
                obs = event[1]
                t.observe_single(obs)
                snapshots.setdefault((obs.key.token, "first"), {})[config_name] = (
                    round(t.p_wrong(obs.key), 8),
                    round(t.p_mode_misreported(obs.key), 8),
                )
            else:
                key, clar = event[1], event[2]
                t.observe_verification(key, clar, VerificationType.REPEAT)
                snapshots.setdefault((key.token, "reask"), {})[config_name] = (
                    round(t.p_wrong(key), 8),
                    round(t.p_mode_misreported(key), 8),
                )

    # Merge snapshots into wide rows (one per feature x timepoint).
    rows: list[dict] = []
    for f in features:
        key = f["key"]
        original = f["original"]
        true_state = f["true_state"]
        true_mode = f["true_mode"]
        wrong_report = int(
            true_state != UNKNOWN and original.value != true_state
        )
        latent_misreport = int(true_mode is ReportMode.MISREPORTED)
        harmful_misreport = int(wrong_report and latent_misreport)
        verified = int(f["verified"])
        verification_value = (
            f["verification"].value if f["verification"] is not None else ""
        )

        base = {
            "case_id": case.case_id,
            "diagnosis": case.diagnosis,
            "noise_rate": noise,
            "seed": seed,
            "key": key.token,
            "value": original.value,
            "certainty": original.certainty.value,
            "true_state": true_state,
            "true_mode": true_mode.value,
            "wrong_report": wrong_report,
            "latent_misreport": latent_misreport,
            "harmful_misreport": harmful_misreport,
            "verified": verified,
        }

        for timepoint in ("first", "reask"):
            snaps = snapshots.get((key.token, timepoint))
            if snaps is None:
                continue
            row = dict(base)
            row["timepoint"] = timepoint
            row["verification_value"] = verification_value if timepoint == "reask" else ""
            for config_name in CONFIG_NAMES:
                pw, pm = snaps[config_name]
                row[f"p_wrong_{config_name}"] = pw
                row[f"p_mode_{config_name}"] = pm
            rows.append(row)

    return rows


FIELDS = [
    "case_id", "diagnosis", "noise_rate", "seed", "key", "value", "certainty",
    "verification_value", "true_state", "true_mode", "wrong_report",
    "latent_misreport", "harmful_misreport", "verified", "timepoint",
] + [
    f"{prefix}_{name}"
    for name in CONFIG_NAMES
    for prefix in ("p_wrong", "p_mode")
]


# --------------------------------------------------------------------------- #
# Worker pool
# --------------------------------------------------------------------------- #

_WORKER = {}


def _init_worker(model, cases_by_id):
    _WORKER["model"] = model
    _WORKER["cases_by_id"] = cases_by_id


def _worker_task(task):
    case_id, noise, seed = task
    case = _WORKER["cases_by_id"][case_id]
    return run_paired_one(case, noise, seed, _WORKER["model"])


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


def _run_grid(cases, noises, seeds, workers):
    cases_by_id = {c.case_id: c for c in cases}
    tasks = [
        (c.case_id, noise, seed)
        for c in cases for noise in noises for seed in seeds
    ]
    print(f"run: {len(tasks)} trajectories over {workers} workers", flush=True)
    ctx = mp.get_context("fork")
    rows: list[dict] = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(_WORKER["model"], cases_by_id)) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_task, tasks, chunksize=4)):
            rows.extend(result)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} done "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    write_csv(rows, OUT / "paired_audit_predictions.csv", FIELDS)
    print(f"done in {elapsed:.0f}s; {len(rows)} paired prediction rows", flush=True)


def write_config(cases, per_disease):
    config = {
        "phase": "channel-alignment-audit-n5",
        "scope": "offline paired audit of p_wrong/p_mode under 4 mode priors "
                 "(matched 0/0 only)",
        "scenario": {"rho_env": 0.0, "rho_model": 0.0},
        "configs": {
            "legacy": "default ChannelParameters().cue_priors",
            "oracle": "env true P(E|cue) from from_noise_rate(noise) (privileged)",
            "train_fixed": "cross-noise-average single prior (no cue conditioning)",
            "cue_conditioned": "env cue mechanism on cross-noise-average prior",
        },
        "labels": ["wrong_report", "latent_misreport", "harmful_misreport"],
        "cases": len(cases),
        "cases_per_disease": per_disease,
        "noise_rates": list(NOISE_RATES),
        "seeds": list(SEEDS),
        "manifest_source": MANIFEST_SOURCE,
        "cross_noise_prior": {
            mode.value: round(v, 6)
            for mode, v in fixed_cross_noise_prior().items()
        },
        "red_lines": [
            "no test split", "no N=20", "no policy modification",
            "no online comparison", "priors parameterized (not fit to val/test)",
            "no true state in prediction", "no git commit",
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
    print(f"model loaded ({time.perf_counter()-t0:.2f}s)", flush=True)
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
