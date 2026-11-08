"""Dataset loader: imports evaluation datasets and loads them for a run.

Import formats
--------------
JSON (full dataset, including an optional chunk corpus)::

    {
      "name": "HR Policy Evaluation", "version": "1.0", "description": "...",
      "chunks": [{"chunk_id": "C101", "title": "...", "text": "..."}],
      "queries": [
        {"query_id": "Q1", "text": "What is the maternity leave policy?",
         "ground_truth": [{"chunk_id": "C101", "grade": 3}, {"chunk_id": "C102", "grade": 2}]}
      ]
    }

  "ground_truth" may also be {"C101": 3, "C102": 2}, and "relevant_chunks":
  ["C101", "C102"] gives every listed chunk grade 1.

CSV (queries and ground truth only), one row per label::

    query_id,query,chunk_id,grade
    Q1,What is the maternity leave policy?,C101,3
"""

import csv
import io
import json
from dataclasses import dataclass, field

from django.db import IntegrityError, transaction

from ..models import MAX_GRADE, EvaluationDataset, EvaluationQuery, GroundTruth, KnowledgeChunk


class DatasetFormatError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class ParsedDataset:
    name: str = ""
    version: str = ""
    description: str = ""
    chunks: list[dict] = field(default_factory=list)
    queries: list[dict] = field(default_factory=list)  # {"query_id", "text", "ground_truth": {cid: grade}}
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalItem:
    """One query ready to evaluate: its text and the expected chunks with grades."""

    query_pk: int
    query_id: str
    text: str
    expected: dict[str, int]

    @property
    def relevant_count(self) -> int:
        return sum(1 for grade in self.expected.values() if grade >= 1)


MAX_ERRORS_SHOWN = 20


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _grade(value, where: str, errors: list[str]) -> int | None:
    try:
        grade = int(str(value).strip())
    except (TypeError, ValueError):
        errors.append(f"{where}: grade {value!r} is not a whole number between 0 and {MAX_GRADE}.")
        return None
    if not 0 <= grade <= MAX_GRADE:
        errors.append(f"{where}: grade {grade} is outside 0-{MAX_GRADE}.")
        return None
    return grade


def _labels_from_json(raw_query: dict, where: str, errors: list[str]) -> dict[str, int]:
    labels: dict[str, int] = {}

    def add(chunk_id, grade_value):
        chunk_id = str(chunk_id or "").strip()
        if not chunk_id:
            errors.append(f"{where}: a ground-truth label has no chunk_id.")
            return
        grade = _grade(grade_value, f"{where}, chunk {chunk_id}", errors)
        if grade is None:
            return
        if chunk_id in labels:
            errors.append(f"{where}: chunk {chunk_id} is labelled twice.")
            return
        labels[chunk_id] = grade

    gt = raw_query.get("ground_truth")
    if isinstance(gt, dict):
        for chunk_id, grade in gt.items():
            add(chunk_id, grade)
    elif isinstance(gt, list):
        for label in gt:
            if isinstance(label, dict):
                add(label.get("chunk_id") or label.get("id"), label.get("grade", 1))
            else:
                add(label, 1)
    elif gt is not None:
        errors.append(f"{where}: \"ground_truth\" must be a list or an object.")
    for chunk_id in raw_query.get("relevant_chunks") or []:
        add(chunk_id, 1)
    return labels


def _check_queries(parsed: ParsedDataset, errors: list[str]) -> None:
    if not parsed.queries:
        errors.append("The dataset has no queries.")
    seen: set[str] = set()
    for q in parsed.queries:
        if q["query_id"] in seen:
            errors.append(f"Query ID {q['query_id']} is used twice.")
        seen.add(q["query_id"])
        if not any(grade >= 1 for grade in q["ground_truth"].values()):
            errors.append(
                f"Query {q['query_id']} has no relevant chunk (grade 1-3), so recall and NDCG are undefined for it."
            )
    if parsed.chunks:
        chunk_ids = {c["chunk_id"] for c in parsed.chunks}
        unknown = sorted({cid for q in parsed.queries for cid in q["ground_truth"] if cid not in chunk_ids})
        if unknown:
            shown = ", ".join(unknown[:10]) + (" ..." if len(unknown) > 10 else "")
            parsed.warnings.append(
                f"{len(unknown)} ground-truth chunk ID(s) are not in the chunk corpus, so the built-in "
                f"retrievers can never return them: {shown}"
            )


def parse_json_dataset(text: str) -> ParsedDataset:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetFormatError([f"Invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})."]) from exc
    if not isinstance(data, dict):
        raise DatasetFormatError(["The JSON file must be an object with a \"queries\" list."])
    errors: list[str] = []
    parsed = ParsedDataset(
        name=str(data.get("name") or "").strip(),
        version=str(data.get("version") or "").strip(),
        description=str(data.get("description") or "").strip(),
    )

    raw_chunks = data.get("chunks") or []
    if not isinstance(raw_chunks, list):
        errors.append("\"chunks\" must be a list.")
        raw_chunks = []
    chunk_ids: set[str] = set()
    for index, chunk in enumerate(raw_chunks, start=1):
        if not isinstance(chunk, dict):
            errors.append(f"Chunk {index} must be an object.")
            continue
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "").strip()
        if not chunk_id:
            errors.append(f"Chunk {index} has no chunk_id.")
            continue
        if chunk_id in chunk_ids:
            errors.append(f"Chunk ID {chunk_id} is used twice.")
            continue
        chunk_ids.add(chunk_id)
        parsed.chunks.append(
            {"chunk_id": chunk_id, "title": str(chunk.get("title") or "")[:300], "text": str(chunk.get("text") or "")}
        )

    raw_queries = data.get("queries")
    if not isinstance(raw_queries, list):
        errors.append("\"queries\" must be a list.")
        raw_queries = []
    for index, raw in enumerate(raw_queries, start=1):
        if not isinstance(raw, dict):
            errors.append(f"Query {index} must be an object.")
            continue
        query_id = str(raw.get("query_id") or raw.get("id") or "").strip()
        text = str(raw.get("text") or raw.get("query") or "").strip()
        where = f"Query {query_id or index}"
        if not query_id:
            errors.append(f"Query {index} has no query_id.")
        if not text:
            errors.append(f"{where} has no text.")
        labels = _labels_from_json(raw, where, errors)
        if query_id and text:
            parsed.queries.append({"query_id": query_id, "text": text, "ground_truth": labels})

    if not errors:
        _check_queries(parsed, errors)
    if errors:
        raise DatasetFormatError(errors[:MAX_ERRORS_SHOWN])
    return parsed


def parse_csv_dataset(text: str) -> ParsedDataset:
    reader = csv.DictReader(io.StringIO(text))
    fields = {(f or "").strip().lower() for f in (reader.fieldnames or [])}
    required = {"query_id", "query", "chunk_id"}
    if not required <= fields:
        raise DatasetFormatError(
            ["The CSV needs the columns query_id, query, chunk_id and grade (grade defaults to 1 if missing)."]
        )
    errors: list[str] = []
    queries: dict[str, dict] = {}
    for line_no, row in enumerate(reader, start=2):
        if len(errors) >= MAX_ERRORS_SHOWN:
            break
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items() if k}
        where = f"Line {line_no}"
        query_id, text, chunk_id = row.get("query_id"), row.get("query"), row.get("chunk_id")
        if not query_id or not text or not chunk_id:
            errors.append(f"{where}: query_id, query and chunk_id are required.")
            continue
        query = queries.setdefault(query_id, {"query_id": query_id, "text": text, "ground_truth": {}})
        if query["text"] != text:
            errors.append(f"{where}: query {query_id} has a different text than on an earlier line.")
            continue
        grade = _grade(row.get("grade") or 1, f"{where}, chunk {chunk_id}", errors)
        if grade is None:
            continue
        if chunk_id in query["ground_truth"]:
            errors.append(f"{where}: chunk {chunk_id} is labelled twice for query {query_id}.")
            continue
        query["ground_truth"][chunk_id] = grade
    parsed = ParsedDataset(queries=list(queries.values()))
    if not errors:
        _check_queries(parsed, errors)
    if errors:
        raise DatasetFormatError(errors[:MAX_ERRORS_SHOWN])
    return parsed


def parse_dataset_file(filename: str, content: bytes) -> ParsedDataset:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetFormatError(["The file is not UTF-8 text."]) from exc
    lower = filename.lower()
    if lower.endswith(".json"):
        return parse_json_dataset(text)
    if lower.endswith(".csv"):
        return parse_csv_dataset(text)
    raise DatasetFormatError(["Upload a .json or .csv file."])


# ---------------------------------------------------------------------------
# Importing
# ---------------------------------------------------------------------------
def import_dataset(
    parsed: ParsedDataset, *, name: str = "", version: str = "", description: str = "", source_filename: str = ""
) -> EvaluationDataset:
    """Create the dataset, its chunks, queries and ground truth in one transaction."""
    name = (name or parsed.name).strip()
    version = (version or parsed.version or "1.0").strip()
    if not name:
        raise DatasetFormatError(["Give the dataset a name (CSV files do not carry one)."])
    if EvaluationDataset.objects.filter(name=name, version=version).exists():
        raise DatasetFormatError([f"A dataset named \"{name}\" version {version} already exists. Change the version."])
    try:
        with transaction.atomic():
            dataset = EvaluationDataset.objects.create(
                name=name,
                version=version,
                description=(description or parsed.description).strip(),
                source_filename=source_filename[:255],
            )
            KnowledgeChunk.objects.bulk_create(
                [KnowledgeChunk(dataset=dataset, **chunk) for chunk in parsed.chunks], batch_size=500
            )
            queries = EvaluationQuery.objects.bulk_create(
                [EvaluationQuery(dataset=dataset, query_id=q["query_id"], text=q["text"]) for q in parsed.queries],
                batch_size=500,
            )
            # bulk_create sets primary keys on SQLite and PostgreSQL.
            labels = [
                GroundTruth(query=query, chunk_id=chunk_id, grade=grade)
                for query, raw in zip(queries, parsed.queries, strict=True)
                for chunk_id, grade in raw["ground_truth"].items()
            ]
            GroundTruth.objects.bulk_create(labels, batch_size=1000)
    except IntegrityError as exc:
        raise DatasetFormatError([f"Could not save the dataset: {exc}"]) from exc
    return dataset


def load_for_evaluation(dataset: EvaluationDataset) -> list[EvalItem]:
    """The dataset's queries with their ground-truth grades, in a stable order."""
    queries = dataset.queries.prefetch_related("ground_truth").order_by("id")
    return [
        EvalItem(
            query_pk=q.pk,
            query_id=q.query_id,
            text=q.text,
            expected={label.chunk_id: label.grade for label in q.ground_truth.all()},
        )
        for q in queries
    ]


def export_dataset(dataset: EvaluationDataset) -> dict:
    """The dataset in the JSON import format (round-trips through import)."""
    return {
        "name": dataset.name,
        "version": dataset.version,
        "description": dataset.description,
        "chunks": [{"chunk_id": c.chunk_id, "title": c.title, "text": c.text} for c in dataset.chunks.all()],
        "queries": [
            {
                "query_id": item.query_id,
                "text": item.text,
                "ground_truth": [
                    {"chunk_id": cid, "grade": grade}
                    for cid, grade in sorted(item.expected.items(), key=lambda kv: (-kv[1], kv[0]))
                ],
            }
            for item in load_for_evaluation(dataset)
        ],
    }
