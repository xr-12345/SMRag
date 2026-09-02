"""Learned verification-worthiness (Phase 4) -- offline training / evaluation.

Scope (narrow, per spec): learn *only* VerifyOld's clinical value, and learn
when NOT to verify.  AskNew keeps the existing EIG unchanged.  This module is
offline: it extracts deployable (label-free) features, derives realized labels,
trains a two-head model (gain regression + harm classifier), and evaluates it
against fixed baselines.  It never wires into the online policy.

Red lines honoured by construction:

* Features are deployable and label-free: reliability (retrospective error
  probability), posterior (Brier diversity), posterior sensitivity
  (diagnostic influence), RAG (retrieval impact), and cost/context
  (n_reports / turn / state index).
* The following are **forbidden as features** and are never exposed by
  :func:`extract_feature_vector`: true disease ``D*``, latent state, true
  wrongness, noise label, realized Brier/NLL, post-action results, oracle,
  and future-turn features.  See :data:`FORBIDDEN_FIELDS`.

The two heads share the same feature vector ``phi_i``:

* gain head ``g_hat_i ~ E[g_i | phi_i]`` (Ridge or HistGradientBoosting),
* harm head  ``h_hat_i ~ P(h_i=1 | phi_i)`` (Logistic or HGB classifier).

Final score ``S_i = g_hat_i - C_verify``; a candidate is allowed only if
``S_i > 0`` and ``h_hat_i <= tau_harm`` (``tau_harm`` selected on train CV).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

C_VERIFY = 0.03
# Frozen weights used by the existing heuristic so we can invert it to recover
# ``diagnostic_influence`` from the recorded ``heuristic_verify_utility``.
DECISION_WEIGHT = 0.25
RELIABILITY_WEIGHT = 0.01

# --- feature schema -------------------------------------------------------- #

# Deployable, label-free features.  Order is canonical: BASE first, then RAG.
BASE_FEATURES: tuple[str, ...] = (
    "retrospective_error_prob",   # reliability: leave-one-out error prob
    "diagnostic_influence",       # posterior sensitivity: L1 belief shift
    "error_prob_times_influence", # interaction
    "current_risk",               # posterior: 1 - sum_d b_d^2 (Brier diversity)
    "n_reports",                  # context / cost
    "turn_index",                 # context / cost
    "state_index",                # context: 0=early 1=middle 2=late
)
RAG_FEATURES: tuple[str, ...] = (
    "retrieval_impact",           # RAG: 1 - jaccard
    "error_prob_times_impact",    # RAG interaction
)
ALL_FEATURES: tuple[str, ...] = BASE_FEATURES + RAG_FEATURES

# Fields that must NEVER appear in the feature vector.  (Diagnosis/noise are
# legitimate *grouping* keys for the split / shuffle, but not features.)
FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    {
        "diagnosis",
        "noise_rate",
        "current_brier",          # realized Brier loss (uses D*)
        "current_true_prob",      # belief of true disease (uses D*)
        "eig",                    # AskNew EIG (AskNew-only, not VerifyOld)
        "v_bayes",                # baseline, not a learned feature
        "v_real",
        "v_real_se",
        "gross_brier_reduction",
        "gross_nll_reduction",
        "top1_before_correct",
        "wrong_to_correct",
        "correct_to_wrong",
        "reported_value",
        "true_state",
        "is_report_wrong",
    }
)


def feature_schema() -> list[dict]:
    """Machine-readable schema for ``feature_schema.json``."""
    rows = []
    for name in ALL_FEATURES:
        family = "rag" if name in RAG_FEATURES else (
            "reliability" if name == "retrospective_error_prob"
            else "posterior" if name in ("diagnostic_influence",
                                         "error_prob_times_influence",
                                         "current_risk")
            else "context"
        )
        rows.append({"feature": name, "family": family, "deployable": True,
                     "label_free": True, "rag": name in RAG_FEATURES})
    return rows


# --- label-free feature derivation ----------------------------------------- #


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def back_out_influence(utility: float, error_prob: float) -> float:
    """Recover ``diagnostic_influence`` from the recorded heuristic utility.

    ``heuristic_verify_utility = error*influence + w_r*H(error) + w_d*influence
    - C`` with ``w_r=RELIABILITY_WEIGHT``, ``w_d=DECISION_WEIGHT``, ``C=0.03``.
    Inverting exactly (up to the 10-decimal rounding of the CSV) gives the same
    influence the policy computed, so train (backed out) and validation
    (recorded) features agree.
    """
    entropy = binary_entropy(error_prob)
    return (utility - RELIABILITY_WEIGHT * entropy + C_VERIFY) / (
        error_prob + DECISION_WEIGHT
    )


def extract_feature_vector(row: Mapping[str, object]) -> np.ndarray:
    """Return the full feature vector (BASE + RAG) for one VerifyOld sample."""
    error_prob = float(row["retrospective_error_prob"])
    influence = back_out_influence(
        float(row["heuristic_verify_utility"]), error_prob
    )
    current_risk = float(row["current_risk"])
    n_reports = float(row["n_reports"])
    turn_index = float(row["turn_index"])
    state_index = float(row["state_index"])
    retrieval_impact = float(row["retrieval_impact"])
    error_prob_times_impact = float(row["error_prob_times_impact"])
    return np.asarray(
        [
            error_prob,
            influence,
            error_prob * influence,
            current_risk,
            n_reports,
            turn_index,
            state_index,
            retrieval_impact,
            error_prob_times_impact,
        ],
        dtype=float,
    )


def build_feature_matrix(
    rows: Sequence[Mapping[str, object]], rag_mode: str = "real"
) -> np.ndarray:
    """Feature matrix over VerifyOld rows.

    ``rag_mode``:
      * ``"real"`` -- base + real RAG features (default),
      * ``"none"`` -- base features only (RAG columns dropped),
      * ``"shuffled"`` -- base + shuffled RAG (see :func:`shuffle_rag_features`).
    """
    if rag_mode not in ("real", "none", "shuffled"):
        raise ValueError(f"unknown rag_mode {rag_mode!r}")
    X = np.vstack([extract_feature_vector(r) for r in rows])
    if rag_mode == "none":
        return X[:, : len(BASE_FEATURES)]
    if rag_mode == "shuffled":
        return shuffle_rag_features(rows, seed=3031)
    return X


# --- realized labels ------------------------------------------------------- #


def derive_labels(row: Mapping[str, object]) -> dict[str, float]:
    """Derive the four learning targets / eval labels for one VerifyOld sample.

    * ``gain``   = ``gross_brier_reduction``  (gross Brier gain g_i)
    * ``value``  = ``v_real``                 (net value v_i = g_i - C)
    * ``harm``   = ``1[correct_to_wrong > 0]``(binary: can flip correct->wrong)
    * ``benefit``= ``1[v_real > 0]``          (binary: net positive)
    """
    value = float(row["v_real"])
    gain = float(row["gross_brier_reduction"])
    correct_to_wrong = float(row["correct_to_wrong"])
    return {
        "gain": gain,
        "value": value,
        "harm": float(correct_to_wrong > 0.0),
        "benefit": float(value > 0.0),
    }


# --- fixed baselines ------------------------------------------------------- #


BASELINES: tuple[str, ...] = (
    "heuristic_verify_utility",
    "retrospective_error_probability",
    "retrieval_impact",
    "error_prob_times_impact",
    "v_bayes",
)


def baseline_score(name: str, row: Mapping[str, object]) -> float:
    """Score of one VerifyOld sample under a fixed baseline (bigger = verify)."""
    if name == "heuristic_verify_utility":
        return float(row["heuristic_verify_utility"])
    if name == "retrospective_error_probability":
        return float(row["retrospective_error_prob"])
    if name == "retrieval_impact":
        return float(row["retrieval_impact"])
    if name == "error_prob_times_impact":
        return float(row["error_prob_times_impact"])
    if name == "v_bayes":
        return float(row["v_bayes"])
    raise KeyError(name)


# --- matched-budget helper ------------------------------------------------- #


def top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask selecting the top-``k`` indices by score (descending).

    Used to compare methods at a matched verification budget.  Ties broken by
    index order.  ``k`` is clamped to ``len(scores)``.
    """
    scores = np.asarray(scores, dtype=float)
    k = max(0, min(int(k), scores.size))
    if k == 0:
        return np.zeros(scores.size, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(scores.size, dtype=bool)
    mask[order[:k]] = True
    return mask


# --- RAG shuffle ablation -------------------------------------------------- #


def _stratum(row: Mapping[str, object], turn_bucket: int = 4) -> tuple:
    turn = int(row["turn_index"])
    return (row["diagnosis"], str(row["noise_rate"]), turn // turn_bucket)


def shuffle_rag_features(
    rows: Sequence[Mapping[str, object]],
    seed: int = 3031,
    turn_bucket: int = 4,
) -> np.ndarray:
    """Return a feature matrix whose RAG columns are shuffled within stratum.

    Strata are ``(diagnosis, noise_rate, turn_index // turn_bucket)`` so the
    marginal distribution of each RAG feature is preserved within disease /
    noise / turn buckets, but the link between a sample's RAG value and its
    realized outcome is broken.  Used to test whether RAG carries real signal.
    """
    X = build_feature_matrix(rows, rag_mode="real")
    rng = np.random.default_rng(seed)
    out = X.copy()
    rag_start = len(BASE_FEATURES)
    strata: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        strata.setdefault(_stratum(r, turn_bucket), []).append(i)
    for idxs in strata.values():
        if len(idxs) < 2:
            continue
        perm = idxs.copy()
        rng.shuffle(perm)
        for col in range(rag_start, out.shape[1]):
            out[idxs, col] = out[perm, col]
    return out


# --- models ---------------------------------------------------------------- #

GAIN_KINDS = ("ridge", "hgb")
HARM_KINDS = ("logistic", "hgb")


@dataclass
class LearnedWorthinessModel:
    """Frozen two-head learned worthiness model."""

    gain_model: object
    harm_model: object
    tau_harm: float
    gain_kind: str
    harm_kind: str
    n_features: int
    rag_mode: str

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(g_hat, h_hat)`` for feature matrix ``X``."""
        return (
            np.asarray(self.gain_model.predict(X), dtype=float),
            np.asarray(self.harm_model.predict_proba(X)[:, 1], dtype=float),
        )

    def score(self, X: np.ndarray) -> np.ndarray:
        """Final score ``S_i = g_hat_i - C_verify`` (bigger = verify)."""
        g_hat, _ = self.predict(X)
        return g_hat - C_VERIFY

    def allowed(self, X: np.ndarray) -> np.ndarray:
        """Boolean mask: ``S_i > 0 AND h_hat_i <= tau_harm``."""
        g_hat, h_hat = self.predict(X)
        return (g_hat - C_VERIFY > 0.0) & (h_hat <= self.tau_harm)

    def to_dict(self) -> dict:
        return {
            "tau_harm": self.tau_harm,
            "gain_kind": self.gain_kind,
            "harm_kind": self.harm_kind,
            "n_features": self.n_features,
            "rag_mode": self.rag_mode,
        }


def _make_gain(kind: str, **params):
    if kind == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=params.get("alpha", 1.0))
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            learning_rate=params.get("learning_rate", 0.1),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            max_iter=params.get("max_iter", 200),
            early_stopping=False,
            random_state=0,
        )
    raise ValueError(kind)


def _make_harm(kind: str, **params):
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=params.get("C", 1.0), max_iter=1000, random_state=0
        )
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.1),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            max_iter=params.get("max_iter", 200),
            early_stopping=False,
            random_state=0,
        )
    raise ValueError(kind)


GAIN_GRID = {
    "ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)],
    "hgb": [
        {"learning_rate": lr, "max_leaf_nodes": ln}
        for lr in (0.05, 0.1)
        for ln in (15, 31)
    ],
}
HARM_GRID = {
    "logistic": [{"C": c} for c in (0.1, 1.0, 10.0)],
    "hgb": [
        {"learning_rate": lr, "max_leaf_nodes": ln}
        for lr in (0.05, 0.1)
        for ln in (15, 31)
    ],
}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if a.size < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return 0.0
    return float(spearmanr(a, b).statistic)


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if np.unique(y).size < 2:
        return 0.5
    return float(roc_auc_score(y, p))


def _group_oof_predict(model_fn, X, y, groups, *, proba=False):
    """Grouped K-fold out-of-fold predictions (or predicted proba)."""
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        # Too few groups to split: fit on all and return in-sample (degenerate).
        model = model_fn()
        model.fit(X, y)
        return (model.predict_proba(X)[:, 1] if proba else model.predict(X))
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        model = model_fn()
        model.fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1] if proba else model.predict(X[te])
    return pred


def _group_cv_select(X, y, groups, kind, grid, *, harm=False):
    """Select hyperparams by grouped CV; return (best_params, best_oof_pred)."""
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        params = grid[0]
        model = (_make_harm if harm else _make_gain)(kind, **params)
        model.fit(X, y)
        pred = (
            model.predict_proba(X)[:, 1] if harm else model.predict(X)
        )
        return params, pred

    gkf = GroupKFold(n_splits=n_splits)
    best_params, best_score = None, -np.inf
    best_pred = None
    for params in grid:
        pred = np.full(len(y), np.nan)
        for tr, te in gkf.split(X, y, groups):
            model = (_make_harm if harm else _make_gain)(kind, **params)
            model.fit(X[tr], y[tr])
            pred[te] = (
                model.predict_proba(X[te])[:, 1] if harm else model.predict(X[te])
            )
        score = _roc_auc(y, pred) if harm else _spearman(y, pred)
        if score > best_score:
            best_score, best_params, best_pred = score, params, pred
    return best_params, best_pred


def train_gain_head(X, y, groups, kind: str):
    """Train gain head; return (fitted_model, best_params, oof_pred)."""
    params, oof = _group_cv_select(X, y, groups, kind, GAIN_GRID[kind], harm=False)
    model = _make_gain(kind, **params)
    model.fit(X, y)
    return model, params, oof


def train_harm_head(X, y, groups, kind: str):
    """Train harm head; return (fitted_model, best_params, oof_pred)."""
    params, oof = _group_cv_select(X, y, groups, kind, HARM_GRID[kind], harm=True)
    model = _make_harm(kind, **params)
    model.fit(X, y)
    return model, params, oof


def select_tau_harm(g_hat_oof, h_hat_oof, value, *, min_allowed: int = 50):
    """Select ``tau_harm`` on train OOF to maximize mean realized value of
    allowed candidates (``S>0 AND h_hat<=tau``).  Falls back to 1.0 if too few
    candidates survive, and never returns a threshold that allows fewer than
    ``min_allowed`` (else the gate would be vacuous / degenerate).
    """
    S = np.asarray(g_hat_oof) - C_VERIFY
    value = np.asarray(value, dtype=float)
    candidates = np.flatnonzero(S > 0.0)
    if candidates.size < min_allowed:
        return 1.0
    h_cand = np.asarray(h_hat_oof)[candidates]
    v_cand = value[candidates]
    order = np.argsort(h_cand)
    h_sorted = h_cand[order]
    v_sorted = v_cand[order]
    best_tau, best_mean = 1.0, float(v_cand.mean())
    for k in range(min_allowed, candidates.size + 1):
        mean_v = float(v_sorted[:k].mean())
        if mean_v > best_mean:
            best_mean, best_tau = mean_v, float(h_sorted[k - 1])
    return float(best_tau)


def freeze_model(
    X, y_gain, y_harm, groups, value, *,
    gain_kind: str, harm_kind: str, rag_mode: str, n_features: int,
) -> LearnedWorthinessModel:
    gain_model, gain_params, gain_oof = train_gain_head(X, y_gain, groups, gain_kind)
    harm_model, harm_params, harm_oof = train_harm_head(X, y_harm, groups, harm_kind)
    tau_harm = select_tau_harm(gain_oof, harm_oof, value)
    return LearnedWorthinessModel(
        gain_model=gain_model,
        harm_model=harm_model,
        tau_harm=tau_harm,
        gain_kind=gain_kind,
        harm_kind=harm_kind,
        n_features=n_features,
        rag_mode=rag_mode,
    )
