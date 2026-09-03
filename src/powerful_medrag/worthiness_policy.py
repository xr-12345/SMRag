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
from .schema import UNKNOWN, FeatureKey, Observation, VariableSpec
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
    UNIFIED_BRIER_RELIABILITY_AUDIT = "unified_brier_reliability_audit"
    # Prompt #18: the Prompt #17 behaviour (pre-stop audit forces VerifyOld) is
    # kept as an independent ablation; the corrected policy only uses gross
    # verification gain to *block* Stop, never to force VerifyOld.
    UNIFIED_BRIER_AUDIT_FORCED_VERIFY = "unified_brier_audit_forced_verify"
    UNIFIED_BRIER_AUDIT_CORRECTED = "unified_brier_audit_corrected"
    # Phase 8B: all quantities derive from ONE JointReportChannel (disease
    # posterior, p_mode, p_wrong, re-ask prediction).  Standalone -- not built
    # through this factory, which targets the BeliefTracker-based family.
    JOINT_CHANNEL_BRIER_AUDIT = "joint_channel_brier_audit"


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
# Unified-Brier reliability-audit policy (Phase 8A)
# --------------------------------------------------------------------------- #


class UnifiedBrierReliabilityAuditPolicy(ReliabilityAwareActionPolicy):
    """Independent unified-Brier action value with an explicit pre-stop audit.

    Sits alongside ``_UnifiedBrierPolicy`` but fixes its two theoretical gaps:

    1. **AskNew on one Brier scale.**  Every still-unasked question is valued
       by ``V_new(j) = R_B(b_t) - E_y[R_B(b_{t+1}^{j,y})] - C_new,j`` with the
       per-question cost ``C_new,j = new_question_cost_weight * spec.cost_j``.
       The chosen question is ``argmax_j V_new(j)`` (Brier-best), *not* the
       EIG-best question valued in Brier units.  The EIG-best and Brier-best
       keys are both recorded so their agreement is observable.

    2. **Explicit VerifyOld audit before Stop.**  Every unverified, non-UNKNOWN
       report is valued by ``verify_value`` (net ``V_verify``) and its gross
       risk reduction ``G_verify = V_verify + C_verify``.  If
       ``G_t^verify = max_i G_verify(i) >= verification_audit_threshold`` the
       policy verifies ``argmax_i G_verify`` instead of stopping.

    .. note::
       This class is the **Prompt #17 forced-verify ablation**.  Prompt #18 found
       that forcing VerifyOld on gross gain breaks the unified action-value
       comparison; the corrected behaviour (gross gain only *blocks* Stop, with a
       global ``verification_advantage_margin`` on the net comparison) lives in
       :class:`CorrectedUnifiedBrierAuditPolicy`.

    The action value reads only the belief, the fitted model, and the frozen
    answer channel.  Retrieval still runs (``retriever`` / ``retrieval_mode``)
    but its Jaccard reordering never enters the Brier value, so it is a pure
    logging side effect here, not a clinical benefit.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_asknew_log: list[dict] = []
        self.last_eig_best_key: FeatureKey | None = None
        self.last_brier_best_key: FeatureKey | None = None
        self._last_audit_g_verify: float | None = None
        self._last_audit_argmax: int | None = None
        self._last_best_new: ActionScore | None = None
        self._last_best_verify: ActionScore | None = None
        self.last_decision_log: dict = {}

    # -- AskNew: enumerate every unasked question ---------------------------- #

    def _asknew_brier_values(
        self, tracker: BeliefTracker, asked: set[FeatureKey]
    ) -> list[tuple[QuestionScore, VariableSpec, float]]:
        values: list[tuple[QuestionScore, VariableSpec, float]] = []
        for question in self.selector.rank(tracker, excluded=asked):
            spec = tracker.model.specs[question.key]
            c_new = self.config.new_question_cost_weight * spec.cost
            v_new = asknew_value(
                tracker, question.key, C_new=c_new, selector=self.selector
            )
            values.append((question, spec, v_new))
        return values

    # -- VerifyOld: every unverified, non-UNKNOWN report --------------------- #

    def _verifyold_brier_values(
        self,
        tracker: BeliefTracker,
        initial_observations: tuple[Observation, ...],
        reports: tuple[Observation, ...],
        verified_report_indices: set[int],
    ) -> list[tuple[int, float, float]]:
        values: list[tuple[int, float, float]] = []
        for index, report in enumerate(reports):
            if index in verified_report_indices or report.value == UNKNOWN:
                continue
            v_verify = verify_value(
                tracker, reports, index, initial_observations,
                C_verify=self.config.verification_cost,
            )
            g_verify = v_verify + self.config.verification_cost
            values.append((index, v_verify, g_verify))
        return values

    # -- unified ranking ---------------------------------------------------- #

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
        del oracle_states, report_risks  # prediction path never reads oracle/true state
        actions: list[ActionScore] = []
        new_actions: list[ActionScore] = []
        verify_actions: list[ActionScore] = []

        asknew_values = self._asknew_brier_values(tracker, asked)
        eig_best_key = asknew_values[0][0].key if asknew_values else None
        brier_best_key = (
            max(asknew_values, key=lambda entry: entry[2])[0].key
            if asknew_values
            else None
        )
        self.last_eig_best_key = eig_best_key
        self.last_brier_best_key = brier_best_key

        asknew_log: list[dict] = []
        for question, spec, v_new in asknew_values:
            new_action = ActionScore(
                kind=ActionKind.NEW,
                utility=v_new,
                disease_information_gain=question.expected_information_gain,
                decision_impact=question.expected_information_gain,
                burden=self.config.new_question_cost_weight * spec.cost,
                key=question.key,
                explanation=(
                    "AskNew V_new = R_B(b_t) - E R_B(b_{t+1}) - C_new,j "
                    "(Brier risk-reduction unit)"
                ),
            )
            actions.append(new_action)
            new_actions.append(new_action)
            asknew_log.append(
                {
                    "key": question.key.name,
                    "expected_information_gain": round(
                        float(question.expected_information_gain), 6
                    ),
                    "question_cost": round(float(spec.cost), 6),
                    "v_new": round(v_new, 6),
                    "is_eig_best": question.key == eig_best_key,
                    "is_brier_best": question.key == brier_best_key,
                }
            )
        self.last_asknew_log = asknew_log

        verify_values: list[tuple[int, float, float]] = []
        if verification_count < self.config.maximum_verifications:
            verify_values = self._verifyold_brier_values(
                tracker, initial_observations, reports, verified_report_indices
            )
        verification_log: list[dict] = []
        for index, v_verify, g_verify in verify_values:
            verify_action = ActionScore(
                kind=ActionKind.VERIFY,
                utility=v_verify,
                burden=self.config.verification_cost,
                report_index=index,
                explanation=(
                    "VerifyOld V_verify = R_B(b_t) - E R_B(b_{t+1}) - C_verify "
                    "(Brier risk-reduction unit)"
                ),
            )
            actions.append(verify_action)
            verify_actions.append(verify_action)
            verification_log.append(
                {
                    "report_index": index,
                    "evidence_code": reports[index].key.name,
                    "v_verify": round(v_verify, 6),
                    "g_verify": round(g_verify, 6),
                }
            )
        self.last_verification_log = verification_log

        if verify_values:
            self._last_audit_argmax = max(verify_values, key=lambda entry: entry[2])[0]
            self._last_audit_g_verify = max(entry[2] for entry in verify_values)
        else:
            self._last_audit_argmax = None
            self._last_audit_g_verify = None

        self._last_best_new = max(new_actions, key=lambda a: a.utility) if new_actions else None
        self._last_best_verify = max(verify_actions, key=lambda a: a.utility) if verify_actions else None

        return sorted(actions, key=lambda action: action.utility, reverse=True)

    # -- pre-stop audit + stop decision ------------------------------------- #

    def _audit_verify_old(self, verification_count: int) -> ActionScore | None:
        if verification_count >= self.config.maximum_verifications:
            return None
        if self._last_audit_g_verify is None or self._last_audit_argmax is None:
            return None
        if self._last_audit_g_verify < self.config.verification_audit_threshold:
            return None
        return ActionScore(
            kind=ActionKind.VERIFY,
            utility=self._last_audit_g_verify - self.config.verification_cost,
            burden=self.config.verification_cost,
            report_index=self._last_audit_argmax,
            explanation=(
                "pre-stop VerifyOld audit: gross Brier risk reduction "
                "G_t^verify exceeds verification_audit_threshold"
            ),
        )

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
        del oracle_states  # deployable policy never reads oracle/true state
        actions = self.rank_actions(
            tracker,
            initial_observations=initial_observations,
            reports=reports,
            asked=asked,
            verified_report_indices=verified_report_indices,
            verification_count=verification_count,
            oracle_states=None,
            report_risks=report_risks,
        )
        best = actions[0] if actions else None

        all_report_scores = (
            self._verification_scores(
                tracker,
                initial_observations=initial_observations,
                reports=reports,
                verified_report_indices=verified_report_indices,
                report_risks=report_risks,
            )
            if reports
            else []
        )
        suspicious = max((score.score for score in all_report_scores), default=0.0)

        ranked = tracker.ranked_diseases()
        top_probability = ranked[0][1]
        margin = top_probability - (ranked[1][1] if len(ranked) > 1 else 0.0)
        confidence_ready = (
            top_probability >= self.config.posterior_threshold
            or margin >= self.config.posterior_margin_threshold
        )
        utility_low = best is None or best.utility <= self.config.minimum_action_utility
        reliability_ready = suspicious <= self.config.suspicious_report_threshold
        safety_ready = (
            self.safety_constraint is None
            or self.safety_constraint.is_clear(tracker, asked)
        )

        if confidence_ready and utility_low and reliability_ready and safety_ready:
            audited = self._audit_verify_old(verification_count)
            if audited is not None:
                return audited
            return ActionScore(
                kind=ActionKind.STOP,
                utility=0.0,
                explanation="confidence, marginal utility, and report-risk checks passed",
            )
        if best is not None and best.utility > 0:
            return best
        audited = self._audit_verify_old(verification_count)
        if audited is not None:
            return audited
        return ActionScore(
            kind=ActionKind.STOP,
            utility=0.0,
            explanation=(
                "no positive-utility acquisition action or a reliability/safety "
                "check remains; return uncertainty"
            ),
        )


class CorrectedUnifiedBrierAuditPolicy(UnifiedBrierReliabilityAuditPolicy):
    """Prompt #18 correction: gross verification gain *blocks* Stop only.

    The Prompt #17 ``_audit_verify_old`` returned ``VerifyOld`` directly whenever
    ``G_t^verify >= tau_V``, which **forced** verification and broke the unified
    action-value comparison.  The corrected policy replaces that with a pure
    boolean ``audit_blocks_stop``:

    * ``G_t^verify > tau_V`` forbids Stop (strict ``>`` boundary); it never forces
      a VerifyOld.
    * The actual choice between the Brier-best AskNew and the Brier-best
      VerifyOld is a net-Brier comparison with a **global** advantage margin
      ``tau_A = verification_advantage_margin``: VerifyOld wins only when
      ``V_verify > V_new + tau_A``.
    * The margin applies to *every* AskNew/VerifyOld comparison, not only after
      a Stop audit.

    The per-turn decision is recorded in ``last_decision_log`` (label-free).
    """

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
        del oracle_states  # deployable policy never reads oracle/true state
        actions = self.rank_actions(
            tracker,
            initial_observations=initial_observations,
            reports=reports,
            asked=asked,
            verified_report_indices=verified_report_indices,
            verification_count=verification_count,
            oracle_states=None,
            report_risks=report_risks,
        )
        best = actions[0] if actions else None
        best_new = self._last_best_new
        best_verify = self._last_best_verify

        all_report_scores = (
            self._verification_scores(
                tracker,
                initial_observations=initial_observations,
                reports=reports,
                verified_report_indices=verified_report_indices,
                report_risks=report_risks,
            )
            if reports
            else []
        )
        suspicious = max((score.score for score in all_report_scores), default=0.0)

        ranked = tracker.ranked_diseases()
        top_probability = ranked[0][1]
        margin = top_probability - (ranked[1][1] if len(ranked) > 1 else 0.0)
        confidence_ready = (
            top_probability >= self.config.posterior_threshold
            or margin >= self.config.posterior_margin_threshold
        )
        utility_low = best is None or best.utility <= self.config.minimum_action_utility
        reliability_ready = suspicious <= self.config.suspicious_report_threshold
        safety_ready = (
            self.safety_constraint is None
            or self.safety_constraint.is_clear(tracker, asked)
        )

        base_stop_ready = (
            confidence_ready and utility_low and reliability_ready and safety_ready
        )

        max_gross_verify_gain = (
            best_verify.utility + self.config.verification_cost
            if best_verify is not None
            and verification_count < self.config.maximum_verifications
            else 0.0
        )
        audit_blocks_stop = (
            best_verify is not None
            and verification_count < self.config.maximum_verifications
            and max_gross_verify_gain > self.config.verification_audit_threshold
        )
        stop_blocked_by_verify_audit = base_stop_ready and audit_blocks_stop

        self.last_decision_log = {
            "best_eig_question": (
                self.last_eig_best_key.name if self.last_eig_best_key else None
            ),
            "best_brier_question": (
                self.last_brier_best_key.name if self.last_brier_best_key else None
            ),
            "eig_brier_agreement": (
                self.last_eig_best_key is not None
                and self.last_eig_best_key == self.last_brier_best_key
            ),
            "best_new_gross_gain": (
                round(best_new.utility + best_new.burden, 6)
                if best_new is not None
                else None
            ),
            "best_new_net_value": (
                round(best_new.utility, 6) if best_new is not None else None
            ),
            "best_verify_report_index": (
                best_verify.report_index if best_verify is not None else None
            ),
            "best_verify_gross_gain": (
                round(max_gross_verify_gain, 6) if best_verify is not None else None
            ),
            "best_verify_net_value": (
                round(best_verify.utility, 6) if best_verify is not None else None
            ),
            "verification_audit_threshold": self.config.verification_audit_threshold,
            "verification_advantage_margin": self.config.verification_advantage_margin,
            "base_stop_ready": base_stop_ready,
            "audit_blocks_stop": audit_blocks_stop,
            "stop_blocked_by_verify_audit": stop_blocked_by_verify_audit,
            "retrieval_triggered": False,
        }

        if base_stop_ready and not audit_blocks_stop:
            self.last_decision_log["chosen_action"] = ActionKind.STOP.value
            return ActionScore(
                kind=ActionKind.STOP,
                utility=0.0,
                explanation="confidence, marginal utility, and report-risk checks passed",
            )

        if best_new is not None and best_verify is not None:
            if (
                best_verify.utility
                > best_new.utility + self.config.verification_advantage_margin
            ):
                self.last_decision_log["chosen_action"] = ActionKind.VERIFY.value
                return best_verify
            self.last_decision_log["chosen_action"] = ActionKind.NEW.value
            return best_new
        if best_new is not None:
            self.last_decision_log["chosen_action"] = ActionKind.NEW.value
            return best_new
        if best_verify is not None and best_verify.utility > 0:
            self.last_decision_log["chosen_action"] = ActionKind.VERIFY.value
            return best_verify
        self.last_decision_log["chosen_action"] = ActionKind.STOP.value
        return ActionScore(
            kind=ActionKind.STOP,
            utility=0.0,
            explanation=(
                "no positive-utility acquisition action or a reliability/safety "
                "check remains; return uncertainty"
            ),
        )


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
    if strategy is WorthinessStrategy.UNIFIED_BRIER_RELIABILITY_AUDIT:
        return UnifiedBrierReliabilityAuditPolicy(config=config, retriever=retriever)
    if strategy is WorthinessStrategy.UNIFIED_BRIER_AUDIT_FORCED_VERIFY:
        return UnifiedBrierReliabilityAuditPolicy(config=config, retriever=retriever)
    if strategy is WorthinessStrategy.UNIFIED_BRIER_AUDIT_CORRECTED:
        return CorrectedUnifiedBrierAuditPolicy(config=config, retriever=retriever)
    if strategy is WorthinessStrategy.JOINT_CHANNEL_BRIER_AUDIT:
        raise NotImplementedError(
            "joint_channel_brier_audit is a standalone strategy that runs on the "
            "joint tracker, not BeliefTracker; construct JointChannelBrierAuditPolicy "
            "(model, channel=...) and run run_joint_channel_dialogue directly"
        )
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
