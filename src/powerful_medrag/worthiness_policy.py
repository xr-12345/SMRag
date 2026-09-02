"""Phase 5 -- online integration of the frozen learned verification-worthiness.

This module wires the frozen two-head ``LearnedWorthinessModel`` (Phase 4) into
the complete AskNew / VerifyOld / Stop policy under four strategies, and provides
the unified cross-type comparison the spec requires: EIG (nats) and Brier value
are **different units** and must never be compared directly.

Strategy semantics
------------------

* ``heuristic_verify`` -- the existing ``ReliabilityAwareActionPolicy`` exactly
  as-is (baseline).  AskNew uses EIG; VerifyOld uses the label-free heuristic
  utility; the history / learned / retrieval joint gate still applies.  This is
  the repository default and is left byte-identical.

* ``model_based_vbayes_verify`` -- AskNew selected by EIG, VerifyOld scored by
  the deployable one-step ``verify_value`` (V_Bayes), and both sides compared in
  the **same** Brier risk-reduction unit.  V_Bayes is deployable (it reads only
  the belief, the fitted model and the frozen answer channel, never the true
  disease / latent state / noise label) but does a full model rollout per
  candidate, so it is the expensive reference point.

* ``learned_worthiness_full_rag`` -- the frozen full-RAG model (9 features)
  fed real retrieval features, gated by ``net_value = g_hat - C_verify > 0`` AND
  ``h_hat <= tau_harm``, sorted by ``net_value``.

* ``learned_worthiness_no_rag`` -- the frozen no-RAG model (7 features, no
  retrieval columns), same learned gate.

Unified cross-type comparison (strategies 2/3/4)
------------------------------------------------

1. Use EIG to select the single best AskNew question.
2. Compute its deployable ``V_Bayes_new = R(b_t) - E_y[R(b_{t+1})] - C_new``.
3. Score VerifyOld in Brier units (V_Bayes_verify or learned ``net_value``).
4. Compare V_Bayes_new vs the VerifyOld value in the same Brier unit, then hand
   the ranked list to the existing Stop condition in ``choose_action``.

Red lines honoured: no retraining, no feature/model change, no tau_harm change,
no AskNew EIG reordering, no true state in the prediction path, no oracle
correction, and the learned model is never the repository default.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Mapping

import numpy as np

from .action_value import asknew_value, brier_risk, verify_value
from .belief import BeliefTracker
from .decision import (
    ActionKind,
    ActionScore,
    ReliabilityAwareActionPolicy,
    ReliabilityAwarePolicyConfig,
)
from .questioning import QuestionScore
from .schema import FeatureKey, Observation, VariableSpec
from .verification_worthiness import (
    BASE_FEATURES,
    C_VERIFY,
    LearnedWorthinessModel,
    binary_entropy,
)

C_NEW = 0.03

FROZEN_MODEL_DIR = Path(
    "artifacts/verimedrag-verification-worthiness-offline/models"
)


class WorthinessStrategy(str, Enum):
    HEURISTIC_VERIFY = "heuristic_verify"
    MODEL_BASED_VBAYES_VERIFY = "model_based_vbayes_verify"
    LEARNED_WORTHINESS_FULL_RAG = "learned_worthiness_full_rag"
    LEARNED_WORTHINESS_NO_RAG = "learned_worthiness_no_rag"


# --------------------------------------------------------------------------- #
# Frozen model loading
# --------------------------------------------------------------------------- #


def load_frozen_worthiness_model(rag_mode: str) -> LearnedWorthinessModel:
    """Load a frozen Phase-4 worthiness model (``real`` / ``none`` / ``shuffled``).

    Fails loudly (``FileNotFoundError``) if the frozen artifact is missing, so a
    broken deploy can never silently fall back to the heuristic.
    """
    if rag_mode not in ("real", "none", "shuffled"):
        raise ValueError(f"unknown worthiness rag_mode {rag_mode!r}")
    path = FROZEN_MODEL_DIR / f"worthiness_{rag_mode}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"frozen worthiness model not found: {path}")
    import joblib

    model = joblib.load(path)
    if not isinstance(model, LearnedWorthinessModel):
        raise TypeError(
            f"frozen model at {path} is {type(model).__name__}, "
            "expected LearnedWorthinessModel"
        )
    return model


# --------------------------------------------------------------------------- #
# Online feature derivation (label-free, causal)
# --------------------------------------------------------------------------- #


def state_index_feature(turn_index: int) -> float:
    """Causal early/middle/late bin (0/1/2) matching the offline ``state_index``.

    Offline ``state_index`` was 0 for the first report-bearing state, 1 for the
    middle state, 2 for the last.  Online we cannot see the future, so we bin the
    current turn index against the observed training mapping: turn 1 -> early,
    turns 2-7 -> middle, turns 8+ -> late.  (``state_index`` has negligible model
    importance -- ~0.005 -- so this causal approximation is immaterial.)
    """
    if turn_index <= 1:
        return 0.0
    if turn_index >= 8:
        return 2.0
    return 1.0


def build_online_features(
    score,
    *,
    current_risk: float,
    n_reports: int,
    turn_index: int,
    use_rag: bool,
) -> np.ndarray:
    """Build the label-free feature vector for one VerifyOld candidate online.

    Mirrors ``extract_feature_vector`` (Phase 4) but reads the live
    ``RetrospectiveScore`` directly instead of a CSV row: ``diagnostic_influence``
    is ``score.diagnostic_influence`` (exactly what ``back_out_influence``
    recovered offline), and the RAG columns are included only when ``use_rag``.

    No true disease, latent state, true wrongness, or noise label enters here.
    """
    error_prob = float(score.error_probability)
    influence = float(score.diagnostic_influence)
    retrieval_impact = (
        float(getattr(score, "retrieval_impact", 0.0)) if use_rag else 0.0
    )
    base = [
        error_prob,
        influence,
        error_prob * influence,
        float(current_risk),
        float(n_reports),
        float(turn_index),
        state_index_feature(turn_index),
    ]
    if use_rag:
        base.extend([retrieval_impact, error_prob * retrieval_impact])
    return np.asarray(base, dtype=float)


def _validate_n_features(model: LearnedWorthinessModel, use_rag: bool) -> None:
    expected = len(BASE_FEATURES) + (2 if use_rag else 0)
    if model.n_features != expected:
        raise ValueError(
            f"worthiness model expects {model.n_features} features but the "
            f"online path builds {expected} (use_rag={use_rag}); wrong model / "
            "feature-schema mismatch"
        )


# --------------------------------------------------------------------------- #
# Unified-Brier policies (strategies 2/3/4)
# --------------------------------------------------------------------------- #


class _UnifiedBrierPolicy(ReliabilityAwareActionPolicy):
    """Base for strategies that compare actions in a single Brier-risk unit.

    Overrides ``rank_actions`` to (a) pick the single best AskNew by EIG and
    value it by deployable ``V_Bayes_new``, and (b) add VerifyOld candidates
    valued by a subclass-supplied Brier quantity, so ``choose_action``'s unified
    ``sorted(actions, key=utility)`` comparison is never EIG-vs-Brier.
    """

    def _best_asknew(self, tracker: BeliefTracker, asked: set[FeatureKey]):
        ranked = self.selector.rank(tracker, excluded=asked)
        return ranked[0] if ranked else None

    def _unified_new_action(
        self, question: QuestionScore, spec: VariableSpec, v_bayes_new: float
    ) -> ActionScore:
        return ActionScore(
            kind=ActionKind.NEW,
            utility=v_bayes_new,
            disease_information_gain=question.expected_information_gain,
            decision_impact=question.expected_information_gain,
            burden=self.config.new_question_cost_weight * spec.cost,
            key=question.key,
            explanation=(
                "best AskNew by EIG, valued by deployable V_Bayes_new "
                "(Brier risk-reduction unit)"
            ),
        )

    def _verifyold_brier_candidates(
        self,
        tracker,
        *,
        initial_observations,
        reports,
        verified_report_indices,
        verification_count,
        report_risks,
        best_asknew_eig,
        best_asknew_vbayes,
    ):
        """Return ``(action_scores, log_entries)`` for VerifyOld candidates."""
        raise NotImplementedError

    def rank_actions(
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
    ) -> list[ActionScore]:
        del oracle_states  # prediction path never reads oracle/true state
        actions: list[ActionScore] = []

        best_asknew = self._best_asknew(tracker, asked)
        best_asknew_eig = None
        best_asknew_vbayes = None
        if best_asknew is not None:
            best_asknew_eig = float(best_asknew.expected_information_gain)
            best_asknew_vbayes = float(
                asknew_value(
                    tracker, best_asknew.key, C_new=C_NEW, selector=self.selector
                )
            )
            spec = tracker.model.specs[best_asknew.key]
            actions.append(
                self._unified_new_action(best_asknew, spec, best_asknew_vbayes)
            )

        logs: list[dict] = []
        verify_actions: list[ActionScore] = []
        if verification_count < self.config.maximum_verifications and reports:
            verify_actions, logs = self._verifyold_brier_candidates(
                tracker,
                initial_observations=initial_observations,
                reports=reports,
                verified_report_indices=verified_report_indices,
                verification_count=verification_count,
                report_risks=report_risks,
                best_asknew_eig=best_asknew_eig,
                best_asknew_vbayes=best_asknew_vbayes,
            )
        actions.extend(verify_actions)

        ranked = sorted(actions, key=lambda action: action.utility, reverse=True)
        chosen_kind = ranked[0].kind.value if ranked else None
        chosen_report = (
            ranked[0].report_index
            if ranked and ranked[0].kind is ActionKind.VERIFY
            else None
        )
        for entry in logs:
            entry["chosen_action"] = chosen_kind
            entry["selected"] = (
                entry["report_index"] == chosen_report
                if chosen_report is not None
                else False
            )
        self.last_verification_log = logs
        return ranked


class ModelBasedVBayesPolicy(_UnifiedBrierPolicy):
    """Strategy 2: AskNew by EIG, VerifyOld by deployable V_Bayes."""

    def _verifyold_brier_candidates(
        self,
        tracker,
        *,
        initial_observations,
        reports,
        verified_report_indices,
        verification_count,
        report_risks,
        best_asknew_eig,
        best_asknew_vbayes,
    ):
        del verification_count
        scores = self._verification_scores(
            tracker,
            initial_observations=initial_observations,
            reports=reports,
            verified_report_indices=verified_report_indices,
            report_risks=report_risks,
        )
        actions: list[ActionScore] = []
        logs: list[dict] = []
        for score in scores:
            v_bayes = float(
                verify_value(
                    tracker, reports, score.report_index,
                    initial_observations, C_verify=C_VERIFY,
                )
            )
            heuristic_utility, _ = self._verification_utility(score)
            logs.append(
                {
                    "report_index": score.report_index,
                    "evidence_code": reports[score.report_index].key.name,
                    "error_probability": round(score.error_probability, 6),
                    "diagnostic_influence": round(score.diagnostic_influence, 6),
                    "v_bayes_verify": round(v_bayes, 6),
                    "heuristic_verify_utility": round(heuristic_utility, 6),
                    "selected": False,
                    "best_asknew_eig": (
                        round(best_asknew_eig, 6)
                        if best_asknew_eig is not None
                        else None
                    ),
                    "best_asknew_v_bayes": (
                        round(best_asknew_vbayes, 6)
                        if best_asknew_vbayes is not None
                        else None
                    ),
                    "chosen_action": None,
                }
            )
            if v_bayes > 0.0:
                actions.append(
                    ActionScore(
                        kind=ActionKind.VERIFY,
                        utility=v_bayes,
                        disease_information_gain=score.error_probability,
                        reliability_information_gain=binary_entropy(
                            score.error_probability
                        ),
                        decision_impact=score.diagnostic_influence,
                        burden=C_VERIFY,
                        report_index=score.report_index,
                        explanation="deployable V_Bayes(VerifyOld) in Brier units",
                    )
                )
        return actions, logs


class LearnedWorthinessPolicy(_UnifiedBrierPolicy):
    """Strategies 3/4: frozen learned worthiness (full-RAG or no-RAG)."""

    def __init__(self, worthiness_model: LearnedWorthinessModel, **kwargs) -> None:
        super().__init__(**kwargs)
        self.worthiness_model = worthiness_model
        self._use_rag = worthiness_model.n_features > len(BASE_FEATURES)
        _validate_n_features(worthiness_model, self._use_rag)

    def _verifyold_brier_candidates(
        self,
        tracker,
        *,
        initial_observations,
        reports,
        verified_report_indices,
        verification_count,
        report_risks,
        best_asknew_eig,
        best_asknew_vbayes,
    ):
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
        actions: list[ActionScore] = []
        logs: list[dict] = []
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
            allowed = passed_gain and passed_harm
            v_bayes = float(
                verify_value(
                    tracker, reports, score.report_index,
                    initial_observations, C_verify=C_VERIFY,
                )
            )
            heuristic_utility, _ = self._verification_utility(score)
            logs.append(
                {
                    "report_index": score.report_index,
                    "evidence_code": reports[score.report_index].key.name,
                    "gain_hat": round(g_hat, 6),
                    "harm_hat": round(h_hat, 6),
                    "verification_cost": C_VERIFY,
                    "net_value": round(net_value, 6),
                    "passed_gain_gate": passed_gain,
                    "passed_harm_gate": passed_harm,
                    "v_bayes_verify": round(v_bayes, 6),
                    "heuristic_verify_utility": round(heuristic_utility, 6),
                    "retrieval_impact": round(
                        getattr(score, "retrieval_impact", 0.0), 6
                    ),
                    "error_prob_times_impact": round(
                        score.error_probability
                        * getattr(score, "retrieval_impact", 0.0),
                        6,
                    ),
                    "error_probability": round(score.error_probability, 6),
                    "diagnostic_influence": round(score.diagnostic_influence, 6),
                    "selected": False,
                    "best_asknew_eig": (
                        round(best_asknew_eig, 6)
                        if best_asknew_eig is not None
                        else None
                    ),
                    "best_asknew_v_bayes": (
                        round(best_asknew_vbayes, 6)
                        if best_asknew_vbayes is not None
                        else None
                    ),
                    "chosen_action": None,
                }
            )
            if allowed:
                actions.append(
                    ActionScore(
                        kind=ActionKind.VERIFY,
                        utility=net_value,
                        disease_information_gain=score.error_probability,
                        reliability_information_gain=binary_entropy(
                            score.error_probability
                        ),
                        decision_impact=score.diagnostic_influence,
                        burden=C_VERIFY,
                        report_index=score.report_index,
                        explanation=(
                            "learned worthiness net_value = g_hat - C_verify "
                            "(harm gate passed)"
                        ),
                    )
                )
        return actions, logs


# --------------------------------------------------------------------------- #
# Strategy factory
# --------------------------------------------------------------------------- #


def build_policy(
    strategy: WorthinessStrategy | str,
    *,
    config: ReliabilityAwarePolicyConfig | None = None,
    retriever=None,
    worthiness_model: LearnedWorthinessModel | None = None,
) -> ReliabilityAwareActionPolicy:
    """Build the policy for a strategy (default stays ``heuristic_verify``)."""
    strategy = WorthinessStrategy(strategy)
    if strategy is WorthinessStrategy.HEURISTIC_VERIFY:
        return ReliabilityAwareActionPolicy(config=config, retriever=retriever)
    if strategy is WorthinessStrategy.MODEL_BASED_VBAYES_VERIFY:
        return ModelBasedVBayesPolicy(config=config, retriever=retriever)
    if strategy is WorthinessStrategy.LEARNED_WORTHINESS_FULL_RAG:
        if worthiness_model is None:
            worthiness_model = load_frozen_worthiness_model("real")
        return LearnedWorthinessPolicy(
            worthiness_model, config=config, retriever=retriever
        )
    if strategy is WorthinessStrategy.LEARNED_WORTHINESS_NO_RAG:
        if worthiness_model is None:
            worthiness_model = load_frozen_worthiness_model("none")
        return LearnedWorthinessPolicy(
            worthiness_model, config=config, retriever=retriever
        )
    raise ValueError(f"unknown strategy {strategy!r}")
