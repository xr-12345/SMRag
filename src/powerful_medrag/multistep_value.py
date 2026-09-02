"""Multi-step verification-value audit (Phase 7 / Prompt #16).

This module generates and audits *multi-step* rollout labels without training a
model or wiring anything into the online policy.  For a frozen belief state
``H_t`` and a candidate first action ``a``, it executes ``a`` and then continues
with the frozen heuristic policy ``pi_heuristic`` until the interview ends, and
evaluates the terminal cost

    J(a | H_t) = E[ Brier(b_T, D*) + c_q * N_future | H_t, a, pi_heuristic ]

with ``Q_multi(a) = -J(a)`` and, for verifying report ``i``,

    V_multi(i) = J(AskNew_best | H_t) - J(VerifyOld(i) | H_t).

The true disease ``D*`` and the simulator's sampled answers enter ONLY the
evaluation-side terminal metrics.  The frozen state, the continuation policy and
every recorded label-free feature read no true state, no latent clinical state,
no true wrongness and no oracle correction.

Red lines honoured by construction: no retraining, no online integration, no
oracle correction, no true state in the prediction path, AskNew / Stop thresholds
unchanged (the continuation is the exact frozen ``ReliabilityAwareActionPolicy``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .action_value import (
    nll_loss,
    realized_brier_loss,
    rollout_seed,
    top1_error,
)
from .belief import BeliefTracker
from .channel import AnswerChannel
from .clarification import SurprisalClarificationProtocol
from .decision import (
    ActionKind,
    ActionScore,
    ReliabilityAwareActionPolicy,
)
from .schema import FeatureKey, Observation
from .simulator import PatientProfile, StructuredPatientSimulator

# Per-atomic-question cost in the terminal objective (c_q).  Both AskNew and
# VerifyOld count as exactly one atomic question each.
C_QUESTION = 0.03


@dataclass(frozen=True)
class FrozenState:
    """A fully-described mid-dialogue state (no tracker object, so it is cheap to
    copy and fully reconstructible from ``initial_observations`` + ``reports``).

    ``reports`` reflects any already-resolved verifications (a verified report
    holds its resolved value), ``asked`` the set of features already asked (which
    includes any ``initial_observations`` keys), ``verified`` the set of report
    indices already verified, and ``verification_count`` the number of VerifyOld
    actions already performed.
    """

    initial_observations: tuple[Observation, ...]
    reports: tuple[Observation, ...]
    asked: frozenset[FeatureKey]
    verified: frozenset[int]
    verification_count: int


def reconstruct_tracker(
    model,
    channel: AnswerChannel,
    initial_observations: Sequence[Observation],
    reports: Sequence[Observation],
) -> BeliefTracker:
    """Rebuild a belief tracker from a frozen state, matching ``_replay_with_runtime``
    for the no-misreport-gate heuristic (belief + history + state beliefs are all
    deterministic functions of the replay, so the result is bit-identical to the
    tracker the frozen policy actually saw)."""
    tracker = BeliefTracker(model, channel)
    for observation in initial_observations:
        tracker.update(observation)
    for report in reports:
        tracker.update(report)
    return tracker


def terminal_objective(
    terminal_brier: float, future_questions: int, *, c_q: float = C_QUESTION
) -> float:
    """``J(a) = Brier(b_T, D*) + c_q * N_future`` (smaller is better)."""
    return terminal_brier + c_q * future_questions


@dataclass(frozen=True)
class StateSnapshot:
    """One mid-dialogue state with >=1 AskNew and >=2 VerifyOld candidates."""

    turn_index: int
    frozen: FrozenState
    belief: Mapping[str, float]
    new_actions: tuple[ActionScore, ...]
    verify_actions: tuple[ActionScore, ...]


def run_heuristic_probe(
    patient: StructuredPatientSimulator,
    policy: ReliabilityAwareActionPolicy,
    *,
    initial_observations: tuple[Observation, ...],
    channel: AnswerChannel | None = None,
    max_total_turns: int = 15,
) -> tuple[list[StateSnapshot], str]:
    """Run the frozen heuristic policy once, snapshotting every state where at
    least one AskNew and at least two VerifyOld candidates coexist.

    Returns ``(snapshots, stop_reason)``.  The loop mirrors
    ``run_reliability_aware_dialogue`` exactly (including the VerifyOld resolve /
    replay), except it does not append the informational ``retrieval_log`` (the
    decision-path dynamic RAG runs inside ``choose_action`` regardless).
    """
    model = patient.model
    channel = channel or AnswerChannel()
    tracker = BeliefTracker(model, channel)
    for observation in initial_observations:
        tracker.update(observation)
    asked = {observation.key for observation in initial_observations}
    reports: list[Observation] = []
    verified: set[int] = set()
    verification_count = 0
    turns = 0
    stop_reason = "total_turn_budget"
    snapshots: list[StateSnapshot] = []

    while turns < max_total_turns:
        ranked = policy.rank_actions(
            tracker,
            initial_observations=initial_observations,
            reports=tuple(reports),
            asked=asked,
            verified_report_indices=verified,
            verification_count=verification_count,
            oracle_states=None,
            report_risks=(),
        )
        new_actions = tuple(a for a in ranked if a.kind is ActionKind.NEW)
        verify_actions = tuple(a for a in ranked if a.kind is ActionKind.VERIFY)
        if new_actions and len(verify_actions) >= 2:
            snapshots.append(
                StateSnapshot(
                    turn_index=len(reports) + verification_count,
                    frozen=FrozenState(
                        initial_observations=initial_observations,
                        reports=tuple(reports),
                        asked=frozenset(asked),
                        verified=frozenset(verified),
                        verification_count=verification_count,
                    ),
                    belief=dict(tracker.belief),
                    new_actions=new_actions,
                    verify_actions=verify_actions,
                )
            )

        action = policy.choose_action(
            tracker,
            initial_observations=initial_observations,
            reports=tuple(reports),
            asked=asked,
            verified_report_indices=verified,
            verification_count=verification_count,
            oracle_states=None,
            report_risks=(),
        )
        if action.kind is ActionKind.STOP:
            stop_reason = action.explanation
            break
        if action.kind is ActionKind.NEW:
            if action.key is None:
                raise RuntimeError("new-question action lacks a feature key")
            observation, _ = patient.answer(action.key)
            reports.append(observation)
            tracker.update(observation)
            asked.add(action.key)
        elif action.kind is ActionKind.VERIFY:
            report_index = action.report_index
            if report_index is None or report_index >= len(reports):
                raise RuntimeError("verification action has an invalid report index")
            clarification, _ = patient.answer(reports[report_index].key)
            resolved = SurprisalClarificationProtocol.resolve(
                reports[report_index], clarification
            )
            reports[report_index] = resolved
            verified.add(report_index)
            verification_count += 1
            tracker = reconstruct_tracker(
                model, channel, initial_observations, tuple(reports)
            )
        else:  # pragma: no cover - ActionKind has no other members
            raise RuntimeError(f"unknown action kind {action.kind!r}")
        turns += 1

    return snapshots, stop_reason


def select_states(
    snapshots: Sequence[StateSnapshot],
) -> list[tuple[str, StateSnapshot]]:
    """Pick up to two states per case: the first qualifying state (early/middle)
    and the last qualifying state before Stop (late).  A single qualifying state
    yields only ``early``; none yields an empty list (recorded honestly)."""
    if not snapshots:
        return []
    early = snapshots[0]
    late = snapshots[-1]
    if late is early:
        return [("early", early)]
    return [("early", early), ("late", late)]


@dataclass(frozen=True)
class ContinuationOutcome:
    """Terminal state of one continuation run (no true disease enters here)."""

    terminal_belief: Mapping[str, float]
    future_questions: int
    stop_reason: str


def run_continuation(
    patient: StructuredPatientSimulator,
    policy: ReliabilityAwareActionPolicy,
    *,
    initial_observations: tuple[Observation, ...],
    reports: tuple[Observation, ...],
    asked: frozenset[FeatureKey],
    verified: frozenset[int],
    verification_count: int,
    first_kind: str,
    first_key: FeatureKey | None,
    first_report_index: int | None,
    channel: AnswerChannel | None = None,
    max_total_turns: int = 15,
) -> ContinuationOutcome:
    """Execute the candidate first action, then continue with ``pi_heuristic`` to
    Stop / max turns.  Returns the terminal belief and the number of atomic
    questions from the first action onward (inclusive).

    The first action is forced (``first_kind`` in {"new", "verify"}); every later
    action comes from ``policy.choose_action`` with no oracle and no learned gate.
    """
    model = patient.model
    channel = channel or AnswerChannel()
    tracker = reconstruct_tracker(model, channel, initial_observations, reports)
    reports = list(reports)
    asked = set(asked)
    verified = set(verified)
    verification_count = int(verification_count)

    # Remaining atomic-question budget relative to the frozen state.
    turns_consumed = len(reports) + verification_count
    remaining_budget = max_total_turns - turns_consumed

    stop_reason = "total_turn_budget"

    if first_kind == "new":
        if first_key is None:
            raise ValueError("AskNew first action requires first_key")
        observation, _ = patient.answer(first_key)
        reports.append(observation)
        tracker.update(observation)
        asked.add(first_key)
    elif first_kind == "verify":
        if first_report_index is None or not 0 <= first_report_index < len(reports):
            raise ValueError("VerifyOld first action requires a valid report index")
        clarification, _ = patient.answer(reports[first_report_index].key)
        resolved = SurprisalClarificationProtocol.resolve(
            reports[first_report_index], clarification
        )
        reports[first_report_index] = resolved
        verified.add(first_report_index)
        verification_count += 1
        tracker = reconstruct_tracker(
            model, channel, initial_observations, tuple(reports)
        )
    else:
        raise ValueError(f"unknown first_kind {first_kind!r}")

    future_questions = 1
    while future_questions < remaining_budget:
        action = policy.choose_action(
            tracker,
            initial_observations=initial_observations,
            reports=tuple(reports),
            asked=asked,
            verified_report_indices=verified,
            verification_count=verification_count,
            oracle_states=None,
            report_risks=(),
        )
        if action.kind is ActionKind.STOP:
            stop_reason = action.explanation
            break
        if action.kind is ActionKind.NEW:
            if action.key is None:
                raise RuntimeError("new-question action lacks a feature key")
            observation, _ = patient.answer(action.key)
            reports.append(observation)
            tracker.update(observation)
            asked.add(action.key)
        elif action.kind is ActionKind.VERIFY:
            report_index = action.report_index
            if report_index is None or report_index >= len(reports):
                raise RuntimeError("verification action has an invalid report index")
            clarification, _ = patient.answer(reports[report_index].key)
            resolved = SurprisalClarificationProtocol.resolve(
                reports[report_index], clarification
            )
            reports[report_index] = resolved
            verified.add(report_index)
            verification_count += 1
            tracker = reconstruct_tracker(
                model, channel, initial_observations, tuple(reports)
            )
        else:  # pragma: no cover
            raise RuntimeError(f"unknown action kind {action.kind!r}")
        future_questions += 1

    return ContinuationOutcome(
        terminal_belief=dict(tracker.belief),
        future_questions=future_questions,
        stop_reason=stop_reason,
    )


@dataclass(frozen=True)
class MultiStepRollout:
    """Monte-Carlo estimate of the terminal cost of one candidate first action.

    Every realized field uses the true disease ``D*`` and the sampled answers and
    is evaluation-only; nothing here feeds the continuation policy.
    """

    terminal_brier: float
    terminal_brier_se: float
    terminal_nll: float
    terminal_nll_se: float
    terminal_top1_error: float
    future_questions: float
    future_questions_se: float
    j_value: float  # mean terminal Brier + c_q * N_future
    j_se: float
    q_multi: float  # -j_value
    premature_stop: float  # rollout fraction of confident wrong stops
    wrong_to_correct: float  # rollout fraction wrong -> correct at terminal
    correct_to_wrong: float  # rollout fraction correct -> wrong at terminal
    # raw per-rollout diagnostics (for label-stability / counterexample audit)
    terminal_brier_list: tuple[float, ...]
    terminal_nll_list: tuple[float, ...]
    future_questions_list: tuple[int, ...]
    premature_stop_list: tuple[int, ...]
    wrong_to_correct_list: tuple[int, ...]
    correct_to_wrong_list: tuple[int, ...]


def multistep_rollout(
    case,
    model,
    *,
    policy: ReliabilityAwareActionPolicy,
    profile: PatientProfile,
    channel: AnswerChannel,
    frozen: FrozenState,
    first_kind: str,
    first_key: FeatureKey | None,
    first_report_index: int | None,
    base_seed: int,
    noise: float,
    state_index: int,
    n_rollouts: int = 8,
    c_q: float = C_QUESTION,
    max_total_turns: int = 15,
) -> MultiStepRollout:
    """Monte-Carlo estimate of ``J(a | H_t)`` via ``n_rollouts`` continuations.

    Common random numbers: every candidate action for the same ``(case, noise,
    state_index)`` draws the same per-``rollout_index`` patient seed (the seed does
    not depend on the action), so AskNew and all VerifyOld candidates share the
    same random stream and the comparison is variance-reduced.
    """
    true_diagnosis = case.diagnosis
    frozen_belief = reconstruct_tracker(
        model, channel, frozen.initial_observations, frozen.reports
    ).belief
    correct_before = max(frozen_belief, key=frozen_belief.__getitem__) == true_diagnosis

    briers: list[float] = []
    nlls: list[float] = []
    top1_errors: list[int] = []
    future: list[int] = []
    premature: list[int] = []
    wrong_to_correct: list[int] = []
    correct_to_wrong: list[int] = []

    for rollout_index in range(n_rollouts):
        seed = rollout_seed(base_seed, case.case_id, noise, state_index, rollout_index)
        patient = StructuredPatientSimulator(
            diagnosis=case.diagnosis,
            latent_states=case.states,
            model=model,
            profile=profile,
            seed=seed,
        )
        outcome = run_continuation(
            patient,
            policy,
            initial_observations=frozen.initial_observations,
            reports=frozen.reports,
            asked=frozen.asked,
            verified=frozen.verified,
            verification_count=frozen.verification_count,
            first_kind=first_kind,
            first_key=first_key,
            first_report_index=first_report_index,
            channel=channel,
            max_total_turns=max_total_turns,
        )
        belief = outcome.terminal_belief
        briers.append(realized_brier_loss(belief, true_diagnosis))
        nlls.append(nll_loss(belief, true_diagnosis))
        top1_errors.append(top1_error(belief, true_diagnosis))
        future.append(outcome.future_questions)
        predicted = max(belief, key=belief.__getitem__)
        confident_stop = "confidence" in outcome.stop_reason or (
            "posterior" in outcome.stop_reason
        )
        premature.append(int(predicted != true_diagnosis and confident_stop))
        correct_after = predicted == true_diagnosis
        wrong_to_correct.append(int((not correct_before) and correct_after))
        correct_to_wrong.append(int(correct_before and (not correct_after)))

    def _mean(values) -> float:
        return sum(values) / len(values)

    def _se(values) -> float:
        if len(values) <= 1:
            return 0.0
        m = _mean(values)
        variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance / len(values))

    # J is the sum of two correlated-but-separable terms; we estimate its SE
    # directly from the per-rollout J samples (the honest MC error of the label).
    j_samples = [
        terminal_objective(b, f, c_q=c_q) for b, f in zip(briers, future)
    ]
    j_value = _mean(j_samples)
    j_se = _se(j_samples)

    return MultiStepRollout(
        terminal_brier=_mean(briers),
        terminal_brier_se=_se(briers),
        terminal_nll=_mean(nlls),
        terminal_nll_se=_se(nlls),
        terminal_top1_error=_mean(top1_errors),
        future_questions=_mean(future),
        future_questions_se=_se(future),
        j_value=j_value,
        j_se=j_se,
        q_multi=-j_value,
        premature_stop=_mean(premature),
        wrong_to_correct=_mean(wrong_to_correct),
        correct_to_wrong=_mean(correct_to_wrong),
        terminal_brier_list=tuple(briers),
        terminal_nll_list=tuple(nlls),
        future_questions_list=tuple(future),
        premature_stop_list=tuple(premature),
        wrong_to_correct_list=tuple(wrong_to_correct),
        correct_to_wrong_list=tuple(correct_to_wrong),
    )
