"""Minimal BM25 medical retrieval over the MedRAG Textbooks corpus.

Sparse (BM25) retrieval only — no dense retriever, no LLM.  Built to be
self-contained (standard library only) so it needs no Java/Pyserini and no
third-party BM25 dependency.

The public surface used by the decision layer:
  * ``MedicalRetriever`` — loads the corpus, builds an in-memory BM25 index,
    and caches ``(query, snippet_id, score)`` results.
  * ``build_retrieval_query`` — turns patient evidence + top-k diseases +
    unresolved conflicts into a single retrieval query string.
  * ``apply_counterfactual`` — delete / weaken / flip an existing report and
    re-run the query + retrieval (for retrieval-impact measurement).
  * ``jaccard`` / ``rbo`` / ``ndcg`` — retrieval-change impact metrics.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .schema import Observation, UNKNOWN

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization (matches MedRAG's BM25 tokenizer
    closely enough for sparse retrieval; no stemming)."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class RetrievalHit:
    """One ranked snippet returned by the retriever."""

    snippet_id: str
    score: float
    title: str = ""
    content: str = ""


@dataclass(frozen=True)
class RetrievalQuery:
    """A constructed query plus the human-readable parts that produced it."""

    text: str
    disease_terms: tuple[str, ...]
    evidence_terms: tuple[str, ...]
    conflict_terms: tuple[str, ...]


@dataclass
class BM25Index:
    """Okapi BM25 over tokenized documents, with an inverted index for speed."""

    documents: list[list[str]]
    k1: float = 1.5
    b: float = 0.75
    _postings: dict[str, list[int]] = field(default_factory=dict, repr=False)
    _doc_len: list[int] = field(default_factory=list, repr=False)
    _tf: list[dict[str, int]] = field(default_factory=list, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _avgdl: float = 0.0

    def __post_init__(self) -> None:
        self._doc_len = [len(d) for d in self.documents]
        self._avgdl = sum(self._doc_len) / max(1, len(self.documents))
        df: Counter[str] = Counter()
        for i, doc in enumerate(self.documents):
            tf = Counter(doc)
            self._tf.append(tf)
            for token in set(doc):
                df[token] += 1
                self._postings.setdefault(token, []).append(i)
        n = len(self.documents)
        self._idf = {
            token: math.log(1.0 + (n - count + 0.5) / (count + 0.5))
            for token, count in df.items()
        }

    def _score(self, query_tokens: list[str], doc_index: int) -> float:
        tf = self._tf[doc_index]
        dl = self._doc_len[doc_index]
        score = 0.0
        for token in set(query_tokens):
            idf = self._idf.get(token)
            if idf is None:
                continue
            freq = tf.get(token, 0)
            if freq == 0:
                continue
            denom = freq + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
            score += idf * freq * (self.k1 + 1.0) / denom
        return score

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        # Only score documents that contain at least one query token.
        candidate = set()
        for token in tokens:
            candidate.update(self._postings.get(token, ()))
        scored = [(i, self._score(tokens, i)) for i in candidate]
        scored = [(i, s) for i, s in scored if s > 0.0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


def load_textbook_corpus(corpus_dir: str | Path) -> list[dict[str, str]]:
    """Load all ``*.jsonl`` textbook snippets into ``[{id,title,content,contents}]``."""
    corpus_dir = Path(corpus_dir)
    snippets: list[dict[str, str]] = []
    for path in sorted(corpus_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                snippets.append(
                    {
                        "id": str(obj["id"]),
                        "title": str(obj.get("title", "")),
                        "content": str(obj.get("content", "")),
                        "contents": str(obj.get("contents") or obj.get("content", "")),
                    }
                )
    return snippets


class MedicalRetriever:
    """In-memory BM25 retriever over the textbook corpus, with a result cache."""

    def __init__(self, corpus_dir: str | Path, k1: float = 1.5, b: float = 0.75):
        self.corpus_dir = Path(corpus_dir)
        self.snippets = load_textbook_corpus(self.corpus_dir)
        self.index = BM25Index(
            [tokenize(s["contents"]) for s in self.snippets], k1=k1, b=b
        )
        self.cache: dict[str, list[RetrievalHit]] = {}

    def __len__(self) -> int:
        return len(self.snippets)

    def retrieve(self, query: str, k: int = 10, *, use_cache: bool = True) -> list[RetrievalHit]:
        if use_cache and query in self.cache:
            return self.cache[query]
        hits = [
            RetrievalHit(
                snippet_id=self.snippets[i]["id"],
                score=score,
                title=self.snippets[i]["title"],
                content=self.snippets[i]["content"],
            )
            for i, score in self.index.search(query, k=k)
        ]
        if use_cache:
            self.cache[query] = hits
        return hits


# --------------------------------------------------------------------------- #
# Query construction
# --------------------------------------------------------------------------- #


def _feature_name(feature_key, feature_names: Mapping | None = None) -> str:
    # DDXPlus FeatureKey.name is an opaque evidence code (e.g. "E_146"); the
    # human-readable clinical term lives in ``VariableSpec.question``.  When a
    # code -> readable-text mapping is supplied, use it so the BM25 query carries
    # real medical words instead of codes.
    if feature_names is not None and feature_key in feature_names:
        return str(feature_names[feature_key])
    return str(getattr(feature_key, "name", feature_key))


_NEGATIVE_VALUES = {"absent", "no", "false", "negative", "none"}


def _is_positive(value: str) -> bool:
    """A value contributes a symptom name to the query only when it asserts the
    finding is present (not "absent"/"no"/"false"/"unknown").  This makes the
    query (and therefore retrieval) change under flip/delete/weaken."""
    v = str(value).lower()
    return v != UNKNOWN and v not in _NEGATIVE_VALUES


def build_retrieval_query(
    reports: Sequence[Observation],
    ranked_diseases: Sequence[tuple[str, float]],
    *,
    top_k: int = 5,
    feature_names: Mapping | None = None,
) -> RetrievalQuery:
    """Build a BM25 query from patient evidence + top-k diseases + conflicts.

    Only *positive* finding names (e.g. "fever" when present) and disease names
    enter the query — textbooks are organized by "what symptoms a disease has",
    not by "what is absent".  This also makes the query respond to a flip
    (present↔absent), delete, or weaken intervention on a report.

    ``feature_names`` maps a FeatureKey to a human-readable term (typically the
    DDXPlus ``question_en``).  Without it the query falls back to the raw code.
    """
    evidence_terms: list[str] = []
    seen_evidence: set[str] = set()
    conflicts: set[str] = set()
    values_by_feature: dict[str, set[str]] = defaultdict(set)

    for report in reports:
        name = _feature_name(report.key, feature_names)
        values_by_feature[name].add(report.value)
        if _is_positive(report.value) and name not in seen_evidence:
            seen_evidence.add(name)
            evidence_terms.append(name)

    for name, values in values_by_feature.items():
        # A feature reported with two different non-UNKNOWN values is a conflict.
        non_unknown = {v for v in values if v != UNKNOWN}
        if len(non_unknown) >= 2:
            conflicts.add(name)

    disease_terms = [name for name, _ in ranked_diseases[:top_k]]

    # Disease terms lead (strongest topical signal); evidence and conflicts follow.
    text_terms = list(dict.fromkeys(disease_terms + evidence_terms + sorted(conflicts)))
    return RetrievalQuery(
        text=" ".join(text_terms),
        disease_terms=tuple(disease_terms),
        evidence_terms=tuple(evidence_terms),
        conflict_terms=tuple(sorted(conflicts)),
    )


# --------------------------------------------------------------------------- #
# Counterfactual interventions
# --------------------------------------------------------------------------- #

CounterfactualOp = str  # "delete" | "weaken" | "flip"


def apply_counterfactual(
    reports: Sequence[Observation],
    report_index: int,
    operation: CounterfactualOp,
) -> list[Observation]:
    """Return a copy of ``reports`` with one report altered.

    * ``delete`` — remove the report.
    * ``weaken`` — replace its value with UNKNOWN (the patient "doesn't know").
    * ``flip`` — swap the value with its opposite on the same feature.
    """
    if not 0 <= report_index < len(reports):
        raise ValueError(f"report index {report_index} out of range")
    out = list(reports)
    target = out[report_index]
    if operation == "delete":
        return out[:report_index] + out[report_index + 1 :]
    if operation == "weaken":
        out[report_index] = Observation(
            target.key, UNKNOWN, certainty=target.certainty
        )
        return out
    if operation == "flip":
        flipped = _flip_value(target)
        out[report_index] = flipped
        return out
    raise ValueError(f"unknown counterfactual operation: {operation}")


def _flip_value(observation: Observation) -> Observation:
    value = observation.value
    low = str(value).lower()
    if low in ("present", "yes", "true"):
        new_value = "absent"
    elif low in ("absent", "no", "false"):
        new_value = "present"
    else:
        new_value = value  # multi-valued feature: can't flip naively, leave as-is
    return Observation(observation.key, new_value, certainty=observation.certainty)


# --------------------------------------------------------------------------- #
# Retrieval-impact metrics
# --------------------------------------------------------------------------- #


def _ids(hits: Sequence[RetrievalHit]) -> list[str]:
    return [h.snippet_id for h in hits]


def jaccard(a: Sequence[RetrievalHit], b: Sequence[RetrievalHit]) -> float:
    sa, sb = set(_ids(a)), set(_ids(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def rbo(
    a: Sequence[RetrievalHit],
    b: Sequence[RetrievalHit],
    p: float = 0.9,
) -> float:
    """Rank-biased overlap (Webber et al. 2010). 1.0 = identical ranking."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    depth = max(len(a), len(b))
    overlap = 0.0
    weight = 0.0
    for d in range(1, depth + 1):
        set_a = set(_ids(a[:d]))
        set_b = set(_ids(b[:d]))
        agree = len(set_a & set_b) / d
        overlap += p ** (d - 1) * agree
        weight += p ** (d - 1)
    return overlap / weight


def ndcg(
    hits: Sequence[RetrievalHit],
    ideal_scores: Sequence[float] | None = None,
) -> float:
    """DCG over retrieved scores, normalized by a sorted (ideal) ranking."""
    if not hits:
        return 0.0
    ideal = sorted((ideal_scores or [h.score for h in hits]), reverse=True)
    dcg = sum((2 ** h.score - 1) / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_change(
    before: Sequence[RetrievalHit],
    after: Sequence[RetrievalHit],
) -> dict[str, float]:
    """Impact of a counterfactual: how much the retrieval result changed."""
    return {
        "topk_jaccard": jaccard(before, after),
        "rbo": rbo(before, after),
        "ndcg_before": ndcg(before),
        "ndcg_after": ndcg(after),
        "ndcg_change": ndcg(after) - ndcg(before),
    }


def counterfactual_retrieval(
    retriever: MedicalRetriever,
    reports: Sequence[Observation],
    ranked_diseases: Sequence[tuple[str, float]],
    report_index: int,
    operation: CounterfactualOp,
    *,
    top_k: int = 5,
    k: int = 10,
    feature_names: Mapping | None = None,
) -> dict:
    """Re-run retrieval after a counterfactual intervention on one report.

    Returns the before/after queries, hits, and retrieval-change metrics.
    """
    before_query = build_retrieval_query(
        reports, ranked_diseases, top_k=top_k, feature_names=feature_names
    )
    before_hits = retriever.retrieve(before_query.text, k=k)
    altered = apply_counterfactual(reports, report_index, operation)
    after_query = build_retrieval_query(
        altered, ranked_diseases, top_k=top_k, feature_names=feature_names
    )
    after_hits = retriever.retrieve(after_query.text, k=k)
    return {
        "operation": operation,
        "report_index": report_index,
        "before_query": before_query.text,
        "after_query": after_query.text,
        "before_hits": before_hits,
        "after_hits": after_hits,
        "change": retrieval_change(before_hits, after_hits),
    }
