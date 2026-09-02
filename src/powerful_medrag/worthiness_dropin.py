"""Phase 6 -- VerifyOld drop-in isolation (learned worthiness as a VerifyOld
re-ranker / filter, without touching AskNew or Stop).

Prompt #14 wired the frozen learned worthiness into the *whole* action policy and
used a unified Brier unit for AskNew, which made the system stop early (``V_Bayes_new
<= 0``) and cut the mean question count from ~9.6 to ~3.3.  That No-Go is retained
and NOT revisited here.

This module tests a different, strictly isolated question:

    Given the *same* VerifyOld opportunity the old heuristic would take, can the
    frozen learned worthiness model pick a better report to re-ask (re-rank), or
    turn a worthless re-ask into a new question (filter)?

Isolation contract (the whole point of this experiment)
-------------------------------------------------------

* The **controller** is the existing heuristic ``ReliabilityAwareActionPolicy``
  exactly as-is.  It decides the action *type*: AskNew / VerifyOld / Stop.

* The learned model may **only** act when the controller chose VerifyOld, and then
  only in one of two ways:

  * ``rerank`` -- among the *same* VerifyOld candidates, prefer ``harm_hat <=
    tau_harm`` and pick the max ``net_value = gain_hat - C_verify``; if no
    candidate passes the harm gate, fall back to the heuristic's report.  Always
    still perform exactly one VerifyOld.

  * ``filter`` -- if some candidate passes ``net_value > 0 AND harm_hat <=
    tau_harm``, verify the max-``net_value`` candidate; otherwise replace the
    verification with the controller's best AskNew (never Stop because of this).

* The learned model never opens a VerifyOld opportunity on its own, never changes
  an AskNew, never changes a Stop, and never compares AskNew value against learned
  net value.  AskNew keeps the EIG ranking; Stop keeps every threshold unchanged.

The repository default stays ``heuristic_baseline`` (a plain
``ReliabilityAwareActionPolicy``).
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Mapping

from .action_value import brier_risk
from .belief import BeliefTracker
from .decision import (
    ActionKind,
    ActionScore,
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
)
from .questioning import QuestionScore
from .schema import FeatureKey, Observation
from .verification_worthiness import (
    BASE_FEATURES,
    C_VERIFY,
    LearnedWorthinessModel,
    binary_entropy,
)
from .worthiness_policy import (
    FROZEN_MODEL_DIR,
    _validate_n_features,
    build_online_features,
    load_frozen_worthiness_model,
)

__all__ = [
    "DropinStrategy",
    "DropinPolicy",
    "build_dropin_policy",
]


class DropinStrategy(str, Enum):
    HEURISTIC_BASELINE = "heuristic_baseline"
    LEARNED_FULL_RERANK = "learned_full_rerank"
    LEARNED_FULL_FILTER = "learned_full_filter"
    LEARNED_NORAG_FILTER = "learned_norag_filter"


class DropinPolicy(ReliabilityAwareActionPolicy):
    """Heuristic controller + learned VerifyOld re-rank / filter.

    ``worthiness_model`` may be ``None`` only for the baseline (which is not
    wrapped by this class at all; it is a plain ``ReliabilityAwareActionPolicy``).
    """

    def __init__(
        self,
        worthiness_model: LearnedWorthinessModel,
        *,
        mode: str = "rerank",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if mode not in ("rerank", "filter"):
            raise ValueError(f"unknown drop-in mode {mode!r}")
        self.worthiness_model = worthiness_model
        self.mode = mode
        self._use_rag = worthiness_model.n_features > len(BASE_FEATURES)
        _validate_n_features(worthiness_model, self._use_rag)
        self.last_dropin_log: list[dict] = []
        self._learned_score_seconds = 0.0

    # -- AskNew (unchanged controller's best) -------------------------------- #

    def _best_asknew_action(
        self, tracker: BeliefTracker, asked: set[FeatureKey]
    ) -> ActionScore | None:
        """The controller's best AskNew action (EIG ordering, unchanged)."""
        best: ActionScore | None = None
        for question in self.selector.rank(tracker, excluded=asked):
            action = self._new_action(
                question, tracker.model.specs[question.key]
            )
            if best is None or action.utility > best.utility:
                best = action
        return best

    def _best_asknew_question(
        self, tracker: BeliefTracker, asked: set[FeatureKey]
    ) -> QuestionScore | None:
        ranked = self.selector.rank(tracker, excluded=asked)
        return ranked[0] if ranked else None

    # -- learned scoring ------------------------------------------------------ #

    def _score_candidates(
        self,
        tracker: BeliefTracker,
        *,
        initial_observations: tuple[Observation, ...],
        reports: tuple[Observation, ...],
        verified_report_indices: set[int],
        verification_count: int,
        report_risks: tuple[float, ...],
    ) -> list[dict]:
        """Compute ``gain_hat``/``harm_hat``/``net_value`` for each VerifyOld
        candidate from the frozen model.  Label-free: no true disease / latent
        state / true wrongness / noise label enters here."""
        scores = self._verification_scores(
            tracker,
            initial_observations=initial_observations,
            reports=reports,
            verified_report_indices=verified_report_indices,
            report_risks=report_risks,
        )
        current_risk = brier_risk(tracker.belief)
        n_reports = len(reports)
        turn_index = n_reports + verification_count
        t0 = time.perf_counter()
        out: list[dict] = []
        for score in scores:
            X = build_online_features(
                score,
                current_risk=current_risk,
                n_reports=n_reports,
                turn_index=turn_index,
                use_rag=self._use_rag,
            )
            g_hat, h_hat = self.worthiness_model.predict(X.reshape(1, -1))
            g_hat, h_hat = float(g_hat[0]), float(h_hat[0])
            net_value = g_hat - C_VERIFY
            passed_gain = net_value > 0.0
            passed_harm = h_hat <= self.worthiness_model.tau_harm
            out.append(
                {
                    "report_index": score.report_index,
                    "evidence_code": reports[score.report_index].key.name,
                    "report_value": reports[score.report_index].value,
                    "gain_hat": round(g_hat, 6),
                    "harm_hat": round(h_hat, 6),
                    "net_value": round(net_value, 6),
                    "passed_gain_gate": passed_gain,
                    "passed_harm_gate": passed_harm,
                    "error_probability": round(score.error_probability, 6),
                    "diagnostic_influence": round(score.diagnostic_influence, 6),
                    "retrieval_impact": round(
                        getattr(score, "retrieval_impact", 0.0), 6
                    ),
                }
            )
        self._learned_score_seconds += time.perf_counter() - t0
        return out

    def _make_verify_action(
        self,
        report_index: int,
        candidate: dict,
        *,
        retrieval_triggered: bool,
    ) -> ActionScore:
        return ActionScore(
            kind=ActionKind.VERIFY,
            utility=candidate["net_value"],
            disease_information_gain=candidate["error_probability"],
            reliability_information_gain=binary_entropy(
                candidate["error_probability"]
            ),
            decision_impact=candidate["diagnostic_influence"],
            burden=C_VERIFY,
            report_index=report_index,
            explanation="learned worthiness drop-in VerifyOld",
            retrieval_triggered=retrieval_triggered,
        )

    # -- choose_action override ---------------------------------------------- #

    def choose_action(
        self,
        tracker: BeliefTracker,
        *,
        initial_observations: tuple[Observation, ...],
        reports: tuple[Observation, ...],
        asked: set[FeatureKey],
        verified_report_indices: set[int],
        verification_count: int,
        oracle_states: Mapping[FeatureKey, str] | None = None,
        report_risks: tuple[float, ...] = (),
    ) -> ActionScore:
        # The controller (old heuristic) decides the action type.
        controller = super().choose_action(
            tracker,
            initial_observations=initial_observations,
            reports=reports,
            asked=asked,
            verified_report_indices=verified_report_indices,
            verification_count=verification_count,
            oracle_states=oracle_states,
            report_risks=report_risks,
        )

        log: dict = {
            "controller_action_type": controller.kind.value,
            "final_action_type": controller.kind.value,
            "heuristic_selected_report": (
                controller.report_index
                if controller.kind is ActionKind.VERIFY
                else None
            ),
            "learned_selected_report": (
                controller.report_index
                if controller.kind is ActionKind.VERIFY
                else None
            ),
            "fallback_to_heuristic": False,
            "replaced_verify_with_asknew": False,
            "best_asknew_evidence": None,
            "stop_reason": (
                controller.explanation
                if controller.kind is ActionKind.STOP
                else None
            ),
            "learned_score_seconds": 0.0,
            "candidates": [],
        }

        if controller.kind is not ActionKind.VERIFY:
            # Learned must never touch an AskNew or a Stop decision.
            self.last_dropin_log.append(log)
            return controller

        return self._learned_verify(
            controller,
            log,
            tracker,
            initial_observations=initial_observations,
            reports=reports,
            asked=asked,
            verified_report_indices=verified_report_indices,
            verification_count=verification_count,
            report_risks=report_risks,
        )

    def _learned_verify(
        self,
        controller: ActionScore,
        log: dict,
        tracker: BeliefTracker,
        *,
        initial_observations: tuple[Observation, ...],
        reports: tuple[Observation, ...],
        asked: set[FeatureKey],
        verified_report_indices: set[int],
        verification_count: int,
        report_risks: tuple[float, ...],
    ) -> ActionScore:
        heuristic_report = controller.report_index
        before = self._learned_score_seconds
        candidates = self._score_candidates(
            tracker,
            initial_observations=initial_observations,
            reports=reports,
            verified_report_indices=verified_report_indices,
            verification_count=verification_count,
            report_risks=report_risks,
        )
        log["learned_score_seconds"] = round(
            self._learned_score_seconds - before, 6
        )
        best_asknew_q = self._best_asknew_question(tracker, asked)
        log["candidates"] = candidates
        log["best_asknew_evidence"] = (
            best_asknew_q.key.name if best_asknew_q is not None else None
        )

        by_index = {c["report_index"]: c for c in candidates}

        if self.mode == "rerank":
            safe = [c for c in candidates if c["passed_harm_gate"]]
            if safe:
                safe.sort(key=lambda c: (-c["net_value"], c["report_index"]))
                final_report = safe[0]["report_index"]
                log["fallback_to_heuristic"] = False
            else:
                final_report = heuristic_report
                log["fallback_to_heuristic"] = True
            if final_report is None:
                final_report = candidates[0]["report_index"] if candidates else None
            candidate = by_index.get(final_report)
            if candidate is None and candidates:
                candidate = candidates[0]
                final_report = candidate["report_index"]
            action = self._make_verify_action(
                final_report,
                candidate,
                retrieval_triggered=controller.retrieval_triggered,
            )
            log["final_action_type"] = "verify"
            log["learned_selected_report"] = final_report
            log["replaced_verify_with_asknew"] = False
            self.last_dropin_log.append(log)
            return action

        # filter mode
        safe = [
            c for c in candidates if c["passed_gain_gate"] and c["passed_harm_gate"]
        ]
        if safe:
            safe.sort(key=lambda c: (-c["net_value"], c["report_index"]))
            final_report = safe[0]["report_index"]
            action = self._make_verify_action(
                final_report,
                by_index[final_report],
                retrieval_triggered=controller.retrieval_triggered,
            )
            log["final_action_type"] = "verify"
            log["learned_selected_report"] = final_report
            log["fallback_to_heuristic"] = False
            log["replaced_verify_with_asknew"] = False
            self.last_dropin_log.append(log)
            return action

        # No candidate passes both gates: replace the verification with AskNew.
        best_asknew = self._best_asknew_action(tracker, asked)
        if best_asknew is not None:
            log["final_action_type"] = "new"
            log["learned_selected_report"] = None
            log["fallback_to_heuristic"] = False
            log["replaced_verify_with_asknew"] = True
            log["best_asknew_evidence"] = best_asknew.key.name
            self.last_dropin_log.append(log)
            return best_asknew

        # No AskNew left to fall back on: keep the verification (never Stop here).
        final_report = heuristic_report if heuristic_report is not None else (
            candidates[0]["report_index"] if candidates else None
        )
        candidate = by_index.get(final_report) or (candidates[0] if candidates else None)
        action = self._make_verify_action(
            final_report,
            candidate,
            retrieval_triggered=controller.retrieval_triggered,
        )
        log["final_action_type"] = "verify"
        log["learned_selected_report"] = final_report
        log["fallback_to_heuristic"] = True
        log["replaced_verify_with_asknew"] = False
        self.last_dropin_log.append(log)
        return action


def build_dropin_policy(
    strategy: DropinStrategy | str,
    *,
    config: ReliabilityAwarePolicyConfig | None = None,
    retriever=None,
    worthiness_model: LearnedWorthinessModel | None = None,
) -> ReliabilityAwareActionPolicy:
    """Build the policy for a drop-in strategy (default stays heuristic)."""
    strategy = DropinStrategy(strategy)
    if strategy is DropinStrategy.HEURISTIC_BASELINE:
        return ReliabilityAwareActionPolicy(config=config, retriever=retriever)
    if strategy is DropinStrategy.LEARNED_FULL_RERANK:
        model = worthiness_model or load_frozen_worthiness_model("real")
        return DropinPolicy(
            model, mode="rerank", config=config, retriever=retriever
        )
    if strategy is DropinStrategy.LEARNED_FULL_FILTER:
        model = worthiness_model or load_frozen_worthiness_model("real")
        return DropinPolicy(
            model, mode="filter", config=config, retriever=retriever
        )
    if strategy is DropinStrategy.LEARNED_NORAG_FILTER:
        model = worthiness_model or load_frozen_worthiness_model("none")
        return DropinPolicy(
            model, mode="filter", config=config, retriever=retriever
        )
    raise ValueError(f"unknown strategy {strategy!r}")
