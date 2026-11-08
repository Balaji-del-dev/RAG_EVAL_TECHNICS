"""Small lexical retrievers in plain Python, over a dataset's chunk corpus.

They exist so you can run real experiments (BM25 vs TF-IDF, stemming on/off,
fusion) without any external service. For your production retriever (dense,
hybrid, reranked ...), use the HTTP adapter or upload its results instead.
"""

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers herself him himself his how
    i if in into is it its itself just me more most my myself no nor not now of off on
    once only or other our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves get got may might must shall also per""".split()
)

TOKEN_RE = re.compile(r"[a-z0-9]+")


def light_stem(token: str) -> str:
    """Strip common English endings (plurals, -ing, -ed). Deliberately simple."""
    if len(token) > 4 and token.endswith(("ies", "ied")):
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, stemming: bool = False) -> list[str]:
    tokens = [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
    return [light_stem(t) for t in tokens] if stemming else tokens


@dataclass(frozen=True)
class Document:
    chunk_id: str
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    score: float


def _ranked(scores: dict[str, float]) -> list[ScoredChunk]:
    """Positive scores only, best first; ties broken by chunk ID so runs are repeatable."""
    return [
        ScoredChunk(cid, score)
        for cid, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if score > 0
    ]


class BM25Index:
    """Okapi BM25 with the Lucene IDF, idf = ln(1 + (N - df + 0.5) / (df + 0.5))."""

    def __init__(self, documents: Iterable[Document], k1: float = 1.5, b: float = 0.75, stemming: bool = False):
        self.k1, self.b, self.stemming = k1, b, stemming
        self.doc_terms: dict[str, Counter] = {}
        self.doc_len: dict[str, int] = {}
        df: Counter = Counter()
        for doc in documents:
            terms = Counter(tokenize(doc.text, stemming))
            self.doc_terms[doc.chunk_id] = terms
            self.doc_len[doc.chunk_id] = sum(terms.values())
            df.update(terms.keys())
        n = len(self.doc_terms)
        self.avgdl = (sum(self.doc_len.values()) / n) if n else 0.0
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def search(self, query: str) -> list[ScoredChunk]:
        q_terms = [t for t in tokenize(query, self.stemming) if t in self.idf]
        scores: dict[str, float] = {}
        for cid, terms in self.doc_terms.items():
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[cid] / self.avgdl) if self.avgdl else self.k1
            score = 0.0
            for term in q_terms:
                tf = terms.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            scores[cid] = score
        return _ranked(scores)


class TfidfIndex:
    """Cosine similarity of sublinear TF-IDF vectors (tf = 1 + ln tf, smooth idf)."""

    def __init__(self, documents: Iterable[Document], stemming: bool = False):
        self.stemming = stemming
        raw = {doc.chunk_id: Counter(tokenize(doc.text, stemming)) for doc in documents}
        df: Counter = Counter()
        for terms in raw.values():
            df.update(terms.keys())
        n = len(raw)
        self.idf = {term: math.log((1 + n) / (1 + d)) + 1 for term, d in df.items()}
        self.doc_vectors = {cid: self._vector(terms) for cid, terms in raw.items()}

    def _vector(self, terms: Counter) -> dict[str, float]:
        vec = {t: (1 + math.log(tf)) * self.idf[t] for t, tf in terms.items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm else {}

    def search(self, query: str) -> list[ScoredChunk]:
        q_vec = self._vector(Counter(tokenize(query, self.stemming)))
        scores = {
            cid: sum(weight * doc_vec.get(term, 0.0) for term, weight in q_vec.items())
            for cid, doc_vec in self.doc_vectors.items()
        }
        return _ranked(scores)


def reciprocal_rank_fusion(rankings: Iterable[list[ScoredChunk]], rrf_k: int = 60) -> list[ScoredChunk]:
    """RRF: score(chunk) = sum over rankings of 1 / (rrf_k + rank)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item.chunk_id] = fused.get(item.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return _ranked(fused)
