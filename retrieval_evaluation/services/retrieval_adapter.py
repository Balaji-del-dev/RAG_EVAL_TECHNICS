"""Retriever adapters: collect the ranked chunk IDs a retriever returns per query.

Django evaluates a retriever's output; it never invents retrieval results.
Three ways to get that output:

* built-in  - BM25 / TF-IDF / fusion over the dataset's chunk corpus
* http      - POST each query to your retriever's endpoint
* precomputed - rankings you exported from your retriever and uploaded

Every adapter returns a list of RetrievedItem, best first.
"""

import csv
import io
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

from django.conf import settings

from . import builtin_retrievers as br


class RetrieverError(Exception):
    """The retriever could not return results for a query (or at all)."""


class ResultsFormatError(ValueError):
    """An uploaded results file could not be read."""


@dataclass(frozen=True)
class RetrievedItem:
    chunk_id: str
    score: float | None = None


@dataclass(frozen=True)
class QueryInput:
    query_id: str
    text: str


class RetrieverAdapter(ABC):
    def prepare(self, dataset) -> None:  # noqa: B027 - optional hook
        """Called once per run before the first query (build indexes, open sessions)."""

    @abstractmethod
    def retrieve(self, query: QueryInput, k: int) -> list[RetrievedItem]:
        """Ranked chunks for one query, best first. May return more or fewer than k."""


# ---------------------------------------------------------------------------
# Built-in lexical retrievers
# ---------------------------------------------------------------------------
class BuiltinRetriever(RetrieverAdapter):
    def __init__(self, kind: str, config: Mapping):
        self.kind = kind
        self.k1 = float(config.get("k1", 1.5))
        self.b = float(config.get("b", 0.75))
        self.rrf_k = int(config.get("rrf_k", 60))
        self.stemming = bool(config.get("stemming", False))
        self.indexes: list = []

    def prepare(self, dataset) -> None:
        docs = [
            br.Document(c.chunk_id, f"{c.title}\n{c.text}")
            for c in dataset.chunks.all().only("chunk_id", "title", "text")
        ]
        if not docs:
            raise RetrieverError(
                "This dataset has no chunk corpus, so the built-in retrievers have nothing to search. "
                "Upload a dataset with a \"chunks\" list, or use the HTTP or uploaded-results adapter."
            )
        if self.kind in ("bm25", "fusion"):
            self.indexes.append(br.BM25Index(docs, k1=self.k1, b=self.b, stemming=self.stemming))
        if self.kind in ("tfidf", "fusion"):
            self.indexes.append(br.TfidfIndex(docs, stemming=self.stemming))

    def rank_all(self, text: str) -> list[br.ScoredChunk]:
        """Every chunk with a positive score, best first (used to see how far down a missed chunk is)."""
        if self.kind == "fusion":
            return br.reciprocal_rank_fusion((index.search(text) for index in self.indexes), self.rrf_k)
        return self.indexes[0].search(text)

    def retrieve(self, query: QueryInput, k: int) -> list[RetrievedItem]:
        return [RetrievedItem(item.chunk_id, round(item.score, 6)) for item in self.rank_all(query.text)[:k]]


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------
class HttpRetriever(RetrieverAdapter):
    """POSTs {"query", "query_id", "top_k"} as JSON and reads the ranked chunks back.

    Accepted responses: ["C1", "C2"], [{"chunk_id": "C1", "score": 0.9}, ...],
    or an object holding one of those under "results", "chunks", "documents"
    or "retrieved". "id" or "doc_id" also work instead of "chunk_id".
    """

    def __init__(self, config: Mapping, secrets: Mapping | None = None):
        self.url = validate_http_url(config.get("url", ""))
        self.timeout = float(config.get("timeout") or settings.RETRIEVAL_EVAL["HTTP_TIMEOUT"])
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        auth = (secrets or {}).get("authorization")
        if auth:
            self.headers["Authorization"] = auth

    def retrieve(self, query: QueryInput, k: int) -> list[RetrievedItem]:
        body = json.dumps({"query": query.text, "query_id": query.query_id, "top_k": k}).encode()
        request = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - scheme checked
                raw = response.read(5 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            raise RetrieverError(f"HTTP {exc.code} from retriever: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RetrieverError(f"Could not reach the retriever at {self.url}: {reason}") from exc
        if len(raw) > 5 * 1024 * 1024:
            raise RetrieverError("Retriever response is larger than 5 MB.")
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RetrieverError("Retriever response is not valid JSON.") from exc
        try:
            return normalize_ranked_list(payload)
        except ResultsFormatError as exc:
            raise RetrieverError(f"Unexpected retriever response: {exc}") from exc


def validate_http_url(url: str) -> str:
    parsed = urlparse(url or "")
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise RetrieverError("The retriever URL must start with http:// or https://.")
    return url


# ---------------------------------------------------------------------------
# Uploaded (precomputed) results
# ---------------------------------------------------------------------------
class PrecomputedRetriever(RetrieverAdapter):
    def __init__(self, config: Mapping):
        self.results = config.get("results") or {}
        if not self.results:
            raise RetrieverError("No uploaded retriever results were stored for this run.")

    def retrieve(self, query: QueryInput, k: int) -> list[RetrievedItem]:
        if query.query_id not in self.results:
            raise RetrieverError(f"The uploaded results have no ranking for query {query.query_id}.")
        return [RetrievedItem(item["chunk_id"], item.get("score")) for item in self.results[query.query_id]]


def build_adapter(retriever_type: str, config: Mapping, secrets: Mapping | None = None) -> RetrieverAdapter:
    if retriever_type in ("bm25", "tfidf", "fusion"):
        return BuiltinRetriever(retriever_type, config)
    if retriever_type == "http":
        return HttpRetriever(config, secrets)
    if retriever_type == "precomputed":
        return PrecomputedRetriever(config)
    raise RetrieverError(f"Unknown retriever type {retriever_type!r}.")


# ---------------------------------------------------------------------------
# Parsing retriever output
# ---------------------------------------------------------------------------
LIST_KEYS = ("results", "chunks", "documents", "retrieved", "chunk_ids")
ID_KEYS = ("chunk_id", "id", "doc_id")


def _item_from(value) -> tuple[dict, int | None]:
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        chunk_id = str(value).strip()
        if not chunk_id:
            raise ResultsFormatError("empty chunk ID")
        return {"chunk_id": chunk_id, "score": None}, None
    if isinstance(value, dict):
        chunk_id = next((str(value[key]).strip() for key in ID_KEYS if value.get(key) not in (None, "")), "")
        if not chunk_id:
            raise ResultsFormatError(f"an item has no chunk ID (expected one of {', '.join(ID_KEYS)})")
        score = value.get("score")
        try:
            score = float(score) if score not in (None, "") else None
        except (TypeError, ValueError) as exc:
            raise ResultsFormatError(f"score of {chunk_id} is not a number") from exc
        rank = value.get("rank")
        try:
            rank = int(rank) if rank not in (None, "") else None
        except (TypeError, ValueError) as exc:
            raise ResultsFormatError(f"rank of {chunk_id} is not a whole number") from exc
        return {"chunk_id": chunk_id, "score": score}, rank
    raise ResultsFormatError(f"unsupported item {value!r}")


def normalize_ranked_list(payload) -> list[RetrievedItem]:
    """Turn one query's retriever output into RetrievedItems, best first."""
    if isinstance(payload, dict):
        for key in LIST_KEYS:
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            raise ResultsFormatError(f"expected a list, or an object with one of: {', '.join(LIST_KEYS)}")
    if not isinstance(payload, list):
        raise ResultsFormatError("expected a list of chunk IDs or chunk objects")
    parsed = [_item_from(value) for value in payload]
    if any(rank is not None for _, rank in parsed):
        if any(rank is None for _, rank in parsed):
            raise ResultsFormatError("give a rank for every item or for none")
        parsed.sort(key=lambda pair: pair[1])
    return [RetrievedItem(item["chunk_id"], item["score"]) for item, _ in parsed]


def parse_results_file(filename: str, content: bytes) -> dict[str, list[dict]]:
    """Read an uploaded results file into {query_id: [{"chunk_id", "score"}, ...]}.

    JSON: {"Q1": ["C1", "C2"]}, {"results": {"Q1": [...]}},
          or [{"query_id": "Q1", "retrieved": [...]}, ...]
    CSV:  columns query_id, chunk_id, and optionally rank and score
          (without rank, rows are read in file order).
    """
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ResultsFormatError("The file is not UTF-8 text.") from exc
    if filename.lower().endswith(".csv"):
        return _parse_results_csv(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResultsFormatError(f"Invalid JSON: {exc.msg} (line {exc.lineno}).") from exc
    return _parse_results_json(data)


def _parse_results_json(data) -> dict[str, list[dict]]:
    # {"retriever": "...", "results": {...}}: metadata next to the rankings is fine.
    if isinstance(data, dict) and isinstance(data.get("results"), (dict, list)):
        data = data["results"]
    rankings: dict[str, list[dict]] = {}
    if isinstance(data, dict):
        entries = list(data.items())
    elif isinstance(data, list):
        entries = []
        for index, row in enumerate(data, start=1):
            if not isinstance(row, dict) or not row.get("query_id"):
                raise ResultsFormatError(f"Entry {index} needs a \"query_id\".")
            ranked = next((row[key] for key in LIST_KEYS if isinstance(row.get(key), list)), None)
            if ranked is None:
                raise ResultsFormatError(f"Entry {index} needs a ranked list under one of: {', '.join(LIST_KEYS)}.")
            entries.append((row["query_id"], ranked))
    else:
        raise ResultsFormatError("Expected a JSON object or list.")
    for query_id, ranked in entries:
        query_id = str(query_id).strip()
        if query_id in rankings:
            raise ResultsFormatError(f"Query {query_id} appears twice.")
        try:
            items = normalize_ranked_list(ranked)
        except ResultsFormatError as exc:
            raise ResultsFormatError(f"Query {query_id}: {exc}.") from exc
        rankings[query_id] = [{"chunk_id": i.chunk_id, "score": i.score} for i in items]
    if not rankings:
        raise ResultsFormatError("The file has no rankings.")
    return rankings


def _parse_results_csv(text: str) -> dict[str, list[dict]]:
    reader = csv.DictReader(io.StringIO(text))
    fields = {f.strip().lower() for f in (reader.fieldnames or [])}
    if not {"query_id", "chunk_id"} <= fields:
        raise ResultsFormatError("The CSV needs the columns query_id and chunk_id (rank and score are optional).")
    rows: dict[str, list] = {}
    for line_no, row in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        if not row.get("query_id") or not row.get("chunk_id"):
            raise ResultsFormatError(f"Line {line_no}: query_id and chunk_id are required.")
        rows.setdefault(row["query_id"], []).append(
            {"chunk_id": row["chunk_id"], "score": row.get("score"), "rank": row.get("rank")}
        )
    if not rows:
        raise ResultsFormatError("The CSV has no rows.")
    rankings = {}
    for query_id, items in rows.items():
        try:
            ranked = normalize_ranked_list(items)
        except ResultsFormatError as exc:
            raise ResultsFormatError(f"Query {query_id}: {exc}.") from exc
        rankings[query_id] = [{"chunk_id": i.chunk_id, "score": i.score} for i in ranked]
    return rankings
