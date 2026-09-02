"""Deployable one-step action value and realized-evaluation returns.

This module implements the unified action value audited in Phase 3A:

    V(a|H_t) = R(b_t) - E_{o~P(o|b_t,a)}[ R(T(b_t,a,o)) ] - C(a)

It keeps two quantities strictly separated:

* **deployable value** -- computed from the belief state ``b_t``, the fitted
  model, and the frozen answer channel only.  It never reads the true disease,
  the latent clinical state, or true wrongness.
* **realized evaluation return** -- computed on the eval side with the true
  disease ``D*`` (and the simulator's sampled answers), used only to *label*
  whether an action actually reduced realized Brier loss.

Risk convention (same unit for AskNew and VerifyOld):

* deployable risk  ``R_Bayes(b)   = 1 - sum_d b(d)^2``
* realized loss    ``L_Brier(b,D*) = sum_d (b(d) - 1[d==D*])^2``

Both are non-negative and decrease as the belief concentrates; ``V`` is the
expected *reduction* minus the action cost, so a positive value means the
action is expected to reduce risk by more than its cost.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .belief import BeliefTracker
from .channel import AnswerChannel, ReportMode
from .clarification import SurprisalClarificationProtocol
from .questioning import NumpyQuestionSelector
from .schema import UNKNOWN, CertaintyCue, ClinicalCase, FeatureKey, Observation
from .simulator import PatientProfile


# --------------------------------------------------------------------------- #
# Risk / loss / cost (shared unit for AskNew and VerifyOld)
# --------------------------------------------------------------------------- #


def brier_risk(belief: Mapping[str, float]) -> float:
    """Deployable Bayes risk: 1 - sum_d b(d)^2 (Brier diversity)."""
    return 1.0 - sum(probability * probability for probability in belief.values())


def realized_brier_loss(belief: Mapping[str, float], true_disease: str) -> float:
    """Evaluation-only realized Brier loss against the true disease."""
    return sum(
        (belief[disease] - (1.0 if disease == true_disease else 0.0)) ** 2
        for disease in belief
    )


def nll_loss(belief: Mapping[str, float], true_disease: str) -> float:
    """Evaluation-only negative log-likelihood of the true disease."""
    return -math.log(max(belief[true_disease], 1e-12))


def top1_error(belief: Mapping[str, float], true_disease: str) -> int:
    """Evaluation-only Top-1 error indicator."""
    return 0 if max(belief, key=belief.__getitem__) == true_disease else 1


def posterior_margin(belief: Mapping[str, float]) -> float:
    """Top-1 minus top-2 posterior probability (0 if only one outcome)."""
    ranking = sorted(belief.values(), reverse=True)
    top = ranking[0] if ranking else 0.0
    second = ranking[1] if len(ranking) > 1 else 0.0
    return top - second


def action_cost(kind: str, *, C_new: float, C_verify: float) -> float:
    if kind == "new":
        return C_new
    if kind == "verify":
        return C_verify
    raise ValueError(f"unknown action kind: {kind!r}")


# --------------------------------------------------------------------------- #
# Tracker replay
# --------------------------------------------------------------------------- #


def replay_tracker(
    reference: BeliefTracker,
    initial_observations: Sequence[Observation],
    reports: Sequence[Observation],
) -> BeliefTracker:
    """Reconstruct a belief tracker from initial observations + reports.

    Uses ``reference.model`` and ``reference.channel`` so the reconstructed
    belief matches the state the policy actually sees.
    """
    tracker = BeliefTracker(reference.model, reference.channel)
    for observation in initial_observations:
        tracker.update(observation)
    for observation in reports:
        tracker.update(observation)
    return tracker


# --------------------------------------------------------------------------- #
# AskNew: deployable answer distribution and one-step value
# --------------------------------------------------------------------------- #


def asknew_answer_distribution(
    tracker: BeliefTracker,
    key: FeatureKey,
    selector: NumpyQuestionSelector | None = None,
) -> dict[str, float]:
    """P(y | b_t, AskNew(j)) over the answer space ``(*spec.values, UNKNOWN)``.

    Reuses the exact answer channel of the EIG selector (full channel with the
    NONE-cue mode prior), so this is the current model's deployable prediction.
    """
    sel = selector or NumpyQuestionSelector()
    return dict(sel.score(tracker, key).predicted_answers)


def asknew_value(
    tracker: BeliefTracker,
    key: FeatureKey,
    *,
    C_new: float,
    selector: NumpyQuestionSelector | None = None,
) -> float:
    """Deployable V_Bayes(AskNew(j)) = R(b_t) - E_y[R(b_{t+1}^{j,y})] - C_new."""
    current_risk = brier_risk(tracker.belief)
    distribution = asknew_answer_distribution(tracker, key, selector=selector)
    expected_risk = 0.0
    for answer, probability in distribution.items():
        posterior = tracker.posterior_for(
            Observation(key=key, value=answer, certainty=CertaintyCue.NONE)
        )
        expected_risk += probability * brier_risk(posterior)
    return current_risk - expected_risk - C_new


# --------------------------------------------------------------------------- #
# VerifyOld: deployable re-ask distribution and one-step value
# --------------------------------------------------------------------------- #


def reask_answer_distribution(
    tracker: BeliefTracker,
    key: FeatureKey,
) -> dict[tuple[str, CertaintyCue], float]:
    """P((value, certainty) | b_t, VerifyOld(key)) for a re-ask of ``key``.

    The re-ask is an independent sample from the *retained* latent state, whose
    distribution ``predictive_state_distribution(key)`` is deployable (it is the
    system's state belief after the original report).  The answer is generated
    by the full channel under the NONE-cue mode prior, so a re-ask may return
    the same (possibly wrong) value, a different value, or UNKNOWN.
    """
    spec = tracker.model.specs[key]
    predictive_states = tracker.predictive_state_distribution(key)
    channel = tracker.channel
    mode_prior = channel.parameters.cue_priors[CertaintyCue.NONE]
    rates = channel.parameters.rates
    values = spec.values

    distribution: dict[tuple[str, CertaintyCue], float] = {}

    unknown_probability = 0.0
    for state, state_probability in predictive_states.items():
        unknown_probability += state_probability * sum(
            mode_prior[mode] * rates[mode].unknown for mode in ReportMode
        )
    distribution[(UNKNOWN, CertaintyCue.NONE)] = unknown_probability

    for value in values:
        certain = 0.0
        uncertain = 0.0
        for state, state_probability in predictive_states.items():
            certain += state_probability * (
                mode_prior[ReportMode.CERTAIN]
                * channel.probability(value, state, values, ReportMode.CERTAIN)
                + mode_prior[ReportMode.MISREPORTED]
                * channel.probability(value, state, values, ReportMode.MISREPORTED)
            )
            uncertain += state_probability * mode_prior[ReportMode.UNCERTAIN] * (
                channel.probability(value, state, values, ReportMode.UNCERTAIN)
            )
        distribution[(value, CertaintyCue.CERTAIN)] = certain
        distribution[(value, CertaintyCue.UNCERTAIN)] = uncertain
    return distribution


def verify_value(
    tracker: BeliefTracker,
    reports: Sequence[Observation],
    report_index: int,
    initial_observations: Sequence[Observation],
    *,
    C_verify: float,
) -> float:
    """Deployable V_Bayes(VerifyOld(i)) = R(b_t) - E[R(b_{t+1})] - C_verify.

    The re-ask result is resolved with ``SurprisalClarificationProtocol.resolve``
    (same answer -> keep with upgraded certainty; different/UNKNOWN -> UNKNOWN),
    so it does NOT assume perfect correction.
    """
    if not 0 <= report_index < len(reports):
        raise ValueError("report_index out of range")
    original = reports[report_index]
    current_risk = brier_risk(tracker.belief)
    if original.value == UNKNOWN:
        # Verifying an UNKNOWN report cannot change the belief.
        return -C_verify

    distribution = reask_answer_distribution(tracker, original.key)
    resolved_probabilities: dict[tuple[str, CertaintyCue], float] = {}
    for (value, certainty), probability in distribution.items():
        clarification = Observation(key=original.key, value=value, certainty=certainty)
        resolved = SurprisalClarificationProtocol.resolve(original, clarification)
        outcome = (resolved.value, resolved.certainty)
        resolved_probabilities[outcome] = (
            resolved_probabilities.get(outcome, 0.0) + probability
        )

    expected_risk = 0.0
    for (value, certainty), probability in resolved_probabilities.items():
        new_reports = list(reports)
        new_reports[report_index] = Observation(
            key=original.key, value=value, certainty=certainty
        )
        after = replay_tracker(tracker, initial_observations, tuple(new_reports))
        expected_risk += probability * brier_risk(after.belief)
    return current_risk - expected_risk - C_verify


# --------------------------------------------------------------------------- #
# Realized evaluation return via Monte Carlo rollout
# --------------------------------------------------------------------------- #


def rollout_seed(
    base_seed: int,
    case_id: str,
    noise: float,
    state_index: int,
    rollout_index: int,
) -> int:
    """Deterministic per-rollout seed (common random numbers across actions)."""
    digest = hashlib.blake2b(
        f"{base_seed}|{case_id}|{noise:.8f}|{state_index}|{rollout_index}".encode(
            "utf-8"
        ),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


def _sample_observation(
    key: FeatureKey,
    true_state: str,
    values: Sequence[str],
    channel: AnswerChannel,
    profile: PatientProfile,
    rng: random.Random,
) -> Observation:
    """Sample one answer from the simulator's channel, mirroring ``answer()``."""
    observed_value, mode = channel.sample(
        true_state, values, rng=rng, mode_prior=profile.mode_prior
    )
    if observed_value == UNKNOWN:
        certainty = CertaintyCue.NONE
    elif mode == ReportMode.UNCERTAIN:
        certainty = CertaintyCue.UNCERTAIN
    else:
        certainty = CertaintyCue.CERTAIN
    return Observation(key=key, value=observed_value, certainty=certainty)


@dataclass(frozen=True)
class RealizedRollout:
    """Evaluation-only realized metrics for one candidate action.

    The true disease and the simulator's sampled answers enter ONLY these
    quantities, never the deployable ``brier_risk`` / ``asknew_value`` /
    ``verify_value`` used to rank actions.
    """

    value: float  # net realized return: realized Brier reduction - cost
    standard_error: float  # MC standard error of ``value``
    gross_brier_reduction: float  # L_Brier(b_t) - E[L_Brier(b_{t+1})]
    gross_nll_reduction: float  # NLL(b_t) - E[NLL(b_{t+1})]
    top1_before_correct: int  # 1 if argmax b_t == D*
    wrong_to_correct: float  # rollout fraction: wrong before -> correct after
    correct_to_wrong: float  # rollout fraction: correct before -> wrong after


def realized_value_mc(
    case: ClinicalCase,
    model,
    tracker: BeliefTracker,
    reports: Sequence[Observation],
    initial_observations: Sequence[Observation],
    *,
    action_kind: str,
    profile: PatientProfile,
    channel: AnswerChannel,
    key: FeatureKey | None = None,
    report_index: int | None = None,
    n_rollouts: int = 8,
    base_seed: int = 3031,
    noise: float = 0.0,
    state_index: int = 0,
    C_new: float = 0.03,
    C_verify: float = 0.03,
) -> RealizedRollout:
    """Monte Carlo estimate of the realized value V_real(a) and its standard error.

    The true disease ``case.diagnosis`` and the sampled answers are used ONLY on
    the evaluation side; the belief update itself is deployable (posterior /
    replay through the reference channel).
    """
    if action_kind == "new":
        if key is None:
            raise ValueError("AskNew rollout requires a feature key")
        cost = C_new
    elif action_kind == "verify":
        if report_index is None or not 0 <= report_index < len(reports):
            raise ValueError("VerifyOld rollout requires a valid report index")
        cost = C_verify
    else:
        raise ValueError(f"unknown action kind: {action_kind!r}")

    true_disease = case.diagnosis
    current_loss = realized_brier_loss(tracker.belief, true_disease)
    current_nll = nll_loss(tracker.belief, true_disease)
    top1_before_correct = 1 - top1_error(tracker.belief, true_disease)
    losses: list[float] = []
    nlls: list[float] = []
    after_correct: list[int] = []
    for rollout_index in range(n_rollouts):
        seed = rollout_seed(
            base_seed, case.case_id, noise, state_index, rollout_index
        )
        rng = random.Random(seed)
        if action_kind == "new":
            spec = model.specs[key]
            observation = _sample_observation(
                key, case.states.get(key, UNKNOWN), spec.values, channel, profile, rng
            )
            after_belief = tracker.posterior_for(observation)
        else:
            original = reports[report_index]
            spec = model.specs[original.key]
            clarification = _sample_observation(
                original.key,
                case.states.get(original.key, UNKNOWN),
                spec.values,
                channel,
                profile,
                rng,
            )
            resolved = SurprisalClarificationProtocol.resolve(
                original, clarification
            )
            new_reports = list(reports)
            new_reports[report_index] = resolved
            after = replay_tracker(tracker, initial_observations, tuple(new_reports))
            after_belief = after.belief
        losses.append(realized_brier_loss(after_belief, true_disease))
        nlls.append(nll_loss(after_belief, true_disease))
        after_correct.append(1 - top1_error(after_belief, true_disease))

    mean_loss = sum(losses) / len(losses)
    mean_nll = sum(nlls) / len(nlls)
    gross_brier_reduction = current_loss - mean_loss
    gross_nll_reduction = current_nll - mean_nll
    value = gross_brier_reduction - cost
    variance = (
        sum((loss - mean_loss) ** 2 for loss in losses) / (len(losses) - 1)
        if len(losses) > 1
        else 0.0
    )
    standard_error = math.sqrt(variance / len(losses))
    wrong_to_correct = (
        sum(1 for c in after_correct if not top1_before_correct and c) / n_rollouts
    )
    correct_to_wrong = (
        sum(1 for c in after_correct if top1_before_correct and not c) / n_rollouts
    )
    return RealizedRollout(
        value=value,
        standard_error=standard_error,
        gross_brier_reduction=gross_brier_reduction,
        gross_nll_reduction=gross_nll_reduction,
        top1_before_correct=top1_before_correct,
        wrong_to_correct=wrong_to_correct,
        correct_to_wrong=correct_to_wrong,
    )
