"""Phase 8C -- mechanism smoke (toy model) + empirical/theoretical consistency.

Runs the spec section-8 mechanism cases (A-E) on the toy model and produces:

* ``empirical_joint_probabilities.csv`` -- P(Y, Y' | z) sampled from
  ``JointStructuredPatientSimulator`` for each true state and rho_env.
* ``theoretical_joint_probabilities.csv`` -- ``JointReportChannel.joint_probability``.
* ``probability_errors.csv`` -- empirical vs theoretical absolute error.
* ``smoke_metrics.json`` -- the section-8 required per-case quantities
  (disease posterior, p_mode, p_wrong, re-ask predictive, true mode chain,
  Brier risk, posterior entropy) plus the max probability error.

Case labels and environments (rho_env / rho_model):
  A: independent re-ask       0.0 / 0.0   -- consistent with old independent channel
  B: high-correlation consistent 0.9 / 0.9 -- not double-counted as two evidence
  C: high-correlation, model thinks independent 0.9 / 0.0 -- overconfidence?
  D: independent env, model thinks correlated  0.0 / 0.5 -- underweighted 2nd answer?
  E: conflicting answers       (either)      -- both preserved, joint update

Red lines: toy model only (never the test split), no rho fitting, the hidden
mode chain is recorded here for evaluation only and never enters any decision.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from powerful_medrag.channel import AnswerChannel, ReportMode
from powerful_medrag.demo_data import generate_toy_cases
from powerful_medrag.estimation import DiseaseStateModel
from powerful_medrag.joint_patient_simulator import JointStructuredPatientSimulator
from powerful_medrag.joint_reliability_belief import JointReliabilityBeliefTracker
from powerful_medrag.joint_report_channel import (
    JointReportChannel,
    VerificationType,
    joint_answer_space,
)
from powerful_medrag.schema import UNKNOWN, CertaintyCue, FeatureKey, Observation
from powerful_medrag.simulator import PatientProfile

OUT = Path(__file__).resolve().parent
N_SEEDS = 20000
TOY_SEED = 41
DISEASE = "influenza"


def _obs(key: FeatureKey, value: str, certainty: CertaintyCue = CertaintyCue.CERTAIN):
    return Observation(FeatureKey(key.name), value, certainty=certainty)


def _entropy(dist) -> float:
    return -sum(p * math.log(p) for p in dist.values() if p > 0)


def _brier_risk(belief) -> float:
    return 1.0 - sum(p * p for p in belief.values())


# --------------------------------------------------------------------------- #
# Empirical vs theoretical joint probability
# --------------------------------------------------------------------------- #


def sample_joint(sim_factory, key, true_state, rho, n_seeds):
    """Empirical P(y, y' | z) over n_seeds for one true state at a given rho."""
    counts: dict[tuple[str, str], int] = {}
    for seed in range(n_seeds):
        sim = sim_factory(true_state, rho, seed)
        first, _ = sim.answer(key)
        second, _ = sim.answer(key)
        pair = (first.value, second.value)
        counts[pair] = counts.get(pair, 0) + 1
    return {pair: count / n_seeds for pair, count in counts.items()}


def build_probability_tables(model, specs):
    key = specs[0].key
    states = specs[0].values
    answers = joint_answer_space(states)

    def factory(true_state, rho, seed):
        return JointStructuredPatientSimulator(
            diagnosis=DISEASE,
            latent_states={key: true_state},
            model=model,
            channel=AnswerChannel(),
            profile=PatientProfile(),  # default profile == cue_priors[NONE]
            seed=seed,
            rho_env=rho,
            case_id="smoke",
        )

    empirical_rows, theoretical_rows, error_rows = [], [], []
    max_error = -1.0
    max_error_cell = None
    for true_state in states:
        for rho in (0.0, 0.5, 0.9):
            empirical = sample_joint(factory, key, true_state, rho, N_SEEDS)
            channel = JointReportChannel(AnswerChannel(), repeat_mode_persistence=rho)
            for y in answers:
                for y_prime in answers:
                    theory = channel.joint_probability(
                        y, y_prime, true_state, states, VerificationType.REPEAT
                    )
                    observed = empirical.get((y, y_prime), 0.0)
                    error = abs(observed - theory)
                    empirical_rows.append(
                        {
                            "true_state": true_state,
                            "rho_env": rho,
                            "first_answer": y,
                            "second_answer": y_prime,
                            "empirical_probability": round(observed, 8),
                        }
                    )
                    theoretical_rows.append(
                        {
                            "true_state": true_state,
                            "rho_model": rho,
                            "first_answer": y,
                            "second_answer": y_prime,
                            "theoretical_probability": round(theory, 8),
                        }
                    )
                    error_rows.append(
                        {
                            "true_state": true_state,
                            "rho": rho,
                            "first_answer": y,
                            "second_answer": y_prime,
                            "empirical_probability": round(observed, 8),
                            "theoretical_probability": round(theory, 8),
                            "absolute_error": round(error, 8),
                        }
                    )
                    if error > max_error:
                        max_error = error
                        max_error_cell = (true_state, rho, y, y_prime)
    return empirical_rows, theoretical_rows, error_rows, max_error, max_error_cell


def write_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Mechanism cases A-E
# --------------------------------------------------------------------------- #


def run_case(label, model, key, first_value, second_value, rho_env, rho_model,
             true_state):
    """Run one smoke case and return the section-8 metric dict.

    The tracker is driven with the explicit ``(first_value, second_value)``
    answers so the mechanism demonstration is deterministic and readable.  In
    parallel, the correlated simulator is sampled at ``rho_env`` (noisy profile)
    purely to record the *true* hidden mode chain -- this is evaluation-only
    metadata and never enters the tracker.
    """
    sim = JointStructuredPatientSimulator(
        diagnosis=DISEASE,
        latent_states={key: true_state},
        model=model,
        channel=AnswerChannel(),
        profile=PatientProfile.from_noise_rate(0.3),
        seed=0,
        rho_env=rho_env,
        case_id=label,
    )
    sim_first, _ = sim.answer(key)
    sim_second, _ = sim.answer(key)

    channel = JointReportChannel(AnswerChannel(), repeat_mode_persistence=rho_model)
    tracker = JointReliabilityBeliefTracker(model, channel)

    first = _obs(key, first_value)
    tracker.observe_single(first)
    pre_belief = dict(tracker.disease_belief())
    pre_p_mode = tracker.p_mode_misreported(key)
    pre_p_wrong = tracker.p_wrong(key)
    reask_predictive = tracker.reask_predictive(key)

    second = _obs(key, second_value)
    tracker.observe_verification(key, second, VerificationType.REPEAT)

    belief = dict(tracker.disease_belief())
    joint_lh = tracker.joint_likelihood(
        DISEASE, key, first, second, VerificationType.REPEAT
    )
    single_prod = (
        tracker.single_likelihood(DISEASE, key, first)
        * tracker.single_likelihood(DISEASE, key, second)
    )
    return {
        "case": label,
        "rho_env": rho_env,
        "rho_model": rho_model,
        "true_state": true_state,
        "first_answer": first_value,
        "second_answer": second_value,
        "simulator_sampled_answers": [sim_first.value, sim_second.value],
        "true_mode_chain": [m.value for m in sim.mode_chain_for(key)],
        "disease_belief_before_verification": {
            d: round(p, 6) for d, p in pre_belief.items()
        },
        "disease_belief_after_verification": {
            d: round(p, 6) for d, p in belief.items()
        },
        "p_mode_before_verification": round(pre_p_mode, 6),
        "p_mode_after_verification": round(tracker.p_mode_misreported(key), 6),
        "p_wrong_before_verification": round(pre_p_wrong, 6),
        "p_wrong_after_verification": round(tracker.p_wrong(key), 6),
        "reask_predictive": {
            v: round(p, 6) for v, p in reask_predictive.items()
        },
        "brier_risk_before": round(_brier_risk(pre_belief), 6),
        "brier_risk_after": round(_brier_risk(belief), 6),
        "posterior_entropy_before": round(_entropy(pre_belief), 6),
        "posterior_entropy_after": round(_entropy(belief), 6),
        "joint_likelihood": round(joint_lh, 8),
        "independent_product_likelihood": round(single_prod, 8),
        "joint_over_independent_ratio": (
            round(joint_lh / single_prod, 6) if single_prod > 0 else None
        ),
    }


def main() -> int:
    cases, specs = generate_toy_cases(cases_per_disease=40, seed=TOY_SEED)
    model = DiseaseStateModel.fit(cases, specs)
    key = specs[0].key  # fever
    true_state = "present"

    # -- empirical/theoretical tables -------------------------------------- #
    emp, theo, err, max_error, max_cell = build_probability_tables(model, specs)
    write_csv(emp, OUT / "empirical_joint_probabilities.csv",
              ["true_state", "rho_env", "first_answer", "second_answer",
               "empirical_probability"])
    write_csv(theo, OUT / "theoretical_joint_probabilities.csv",
              ["true_state", "rho_model", "first_answer", "second_answer",
               "theoretical_probability"])
    write_csv(err, OUT / "probability_errors.csv",
              ["true_state", "rho", "first_answer", "second_answer",
               "empirical_probability", "theoretical_probability",
               "absolute_error"])

    # -- mechanism cases A-E ------------------------------------------------ #
    cases_metrics = [
        run_case("A", model, key, "present", "present", 0.0, 0.0, true_state),
        run_case("B", model, key, "present", "present", 0.9, 0.9, true_state),
        run_case("C", model, key, "present", "present", 0.9, 0.0, true_state),
        run_case("D", model, key, "present", "present", 0.0, 0.5, true_state),
        run_case("E", model, key, "absent", "present", 0.9, 0.9, true_state),
    ]

    # -- simulator mode-chain persistence demo (evaluation-only) ------------ #
    def mode_chain_demo(rho):
        sim = JointStructuredPatientSimulator(
            diagnosis=DISEASE, latent_states={key: true_state}, model=model,
            channel=AnswerChannel(), profile=PatientProfile.from_noise_rate(0.3),
            seed=1234, rho_env=rho, case_id="mode-chain",
        )
        modes = []
        for _ in range(6):
            _, mode = sim.answer(key)
            modes.append(mode.value)
        return modes

    mode_chain = {
        "rho_env_0_chain": mode_chain_demo(0.0),
        "rho_env_1_chain": mode_chain_demo(1.0),
        "note": "rho_env=1 -> every subsequent mode equals the first; "
                "rho_env=0 -> independent draws from the mode prior.",
    }

    smoke = {
        "phase": "8C",
        "description": "mechanism smoke: correlated re-ask simulator vs joint channel",
        "n_seeds_for_probability_tables": N_SEEDS,
        "max_absolute_probability_error": round(max_error, 6),
        "max_error_cell": {
            "true_state": max_cell[0],
            "rho": max_cell[1],
            "first_answer": max_cell[2],
            "second_answer": max_cell[3],
        } if max_cell else None,
        "simulator_mode_chain_demo": mode_chain,
        "cases": cases_metrics,
    }
    (OUT / "smoke_metrics.json").write_text(
        json.dumps(smoke, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # -- console summary ---------------------------------------------------- #
    print(f"max absolute probability error: {max_error:.6f} at {max_cell}")
    for m in cases_metrics:
        before = m["disease_belief_before_verification"]
        after = m["disease_belief_after_verification"]
        top_before = max(before, key=before.__getitem__)
        top_after = max(after, key=after.__getitem__)
        print(
            f"[{m['case']}] rho_env={m['rho_env']} rho_model={m['rho_model']} "
            f"{m['first_answer']}->{m['second_answer']}: top {top_before}"
            f"({before[top_before]:.3f}) -> {top_after}({after[top_after]:.3f}), "
            f"joint/indep ratio={m['joint_over_independent_ratio']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
