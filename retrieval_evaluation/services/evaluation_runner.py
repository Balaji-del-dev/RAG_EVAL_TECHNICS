"""Runs an evaluation: the execution flow behind the "Run Evaluation" button.

    1. create_run()          -> EvaluationRun record (status: pending)
    2. load_for_evaluation() -> test queries + ground-truth grades
    3. adapter.retrieve()    -> ranked chunk IDs from the configured retriever
    4. match()               -> retrieved IDs compared with expected IDs/grades
    5. compute_query_metrics -> six metric scores per query
    6. aggregate()           -> run-level averages
    7. save                  -> MetricResult + RetrievedChunk rows, run completed

Retrieval and scoring happen before any database write, and all results are
written in one transaction, so a slow HTTP retriever never holds a database
lock and a crashed run leaves no half-written results.
"""

import logging
import time
from dataclasses import dataclass, field

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from ..analysis.failure_analysis import detect_issues
from ..metrics import aggregate, compute_query_metrics
from ..models import EvaluationRun, MetricResult, RetrievedChunk
from .dataset_loader import EvalItem, load_for_evaluation
from .ground_truth_matcher import MatchResult, match
from .retrieval_adapter import QueryInput, RetrieverError, build_adapter

logger = logging.getLogger(__name__)


@dataclass
class QueryOutcome:
    item: EvalItem
    match: MatchResult | None = None
    scores: object = None
    issues: list[str] = field(default_factory=list)
    latency_ms: float | None = None
    error: str = ""


def create_run(dataset, *, name, top_k, retriever_type, retriever_config=None, notes="", ndcg_gain=None) -> EvaluationRun:
    return EvaluationRun.objects.create(
        dataset=dataset,
        name=name,
        top_k=top_k,
        retriever_type=retriever_type,
        retriever_config=retriever_config or {},
        notes=notes,
        ndcg_gain=ndcg_gain or settings.RETRIEVAL_EVAL["NDCG_GAIN"],
    )


def _evaluate_query(adapter, item: EvalItem, k: int, gain_type: str) -> QueryOutcome:
    outcome = QueryOutcome(item=item)
    if not item.relevant_count:
        outcome.error = "The query has no relevant chunk (grade 1-3) in its ground truth, so it cannot be scored."
        return outcome
    started = time.perf_counter()
    try:
        retrieved = adapter.retrieve(QueryInput(item.query_id, item.text), k)
    except RetrieverError as exc:
        outcome.error = str(exc)
        return outcome
    finally:
        outcome.latency_ms = round((time.perf_counter() - started) * 1000, 2)
    outcome.match = match(retrieved, item.expected, k)
    outcome.scores = compute_query_metrics(outcome.match.ranked_ids, item.expected, k, gain_type)
    outcome.issues = detect_issues(outcome.scores.as_metric_dict(), item.relevant_count, k)
    return outcome


def _result_row(run: EvaluationRun, outcome: QueryOutcome) -> MetricResult:
    item = outcome.item
    row = MetricResult(
        run=run,
        query_id=item.query_pk,
        query_external_id=item.query_id,
        query_text=item.text,
        expected=item.expected,
        total_relevant=item.relevant_count,
        latency_ms=outcome.latency_ms,
        error=outcome.error,
    )
    if outcome.match is not None:
        m, s = outcome.match, outcome.scores
        row.precision, row.recall, row.f1 = s.precision, s.recall, s.f1
        row.hit_rate, row.reciprocal_rank, row.ndcg = s.hit_rate, s.reciprocal_rank, s.ndcg
        row.first_relevant_rank = s.first_relevant_rank
        row.relevant_found = len(m.relevant_found)
        row.irrelevant_found = len(m.irrelevant_found)
        row.relevant_missed = len(m.missed)
        row.missed_chunk_ids = m.missed
        row.retrieved_count = len(m.ranked)
        row.issues = outcome.issues
        row.warnings = m.warnings
    return row


def execute_run(run: EvaluationRun, secrets: dict | None = None) -> EvaluationRun:
    """Evaluate every query of the run's dataset and store the results.

    secrets (e.g. an HTTP Authorization header) are used for this execution
    only and never stored.
    """
    gain_type = run.ndcg_gain
    run.status, run.started_at, run.error_message = EvaluationRun.Status.RUNNING, timezone.now(), ""
    run.save(update_fields=["status", "started_at", "error_message"])
    started = time.perf_counter()
    try:
        items = load_for_evaluation(run.dataset)
        if not items:
            raise RetrieverError("The dataset has no queries.")
        adapter = build_adapter(run.retriever_type, run.retriever_config, secrets)
        adapter.prepare(run.dataset)
        outcomes = [_evaluate_query(adapter, item, run.top_k, gain_type) for item in items]
        scored = [o for o in outcomes if o.match is not None]
        if not scored:
            first_error = next((o.error for o in outcomes if o.error), "unknown error")
            raise RetrieverError(f"No query could be evaluated. First error: {first_error}")

        averages = aggregate(o.scores for o in scored)
        with transaction.atomic():
            run.results.all().delete()
            rows = MetricResult.objects.bulk_create([_result_row(run, o) for o in outcomes], batch_size=500)
            RetrievedChunk.objects.bulk_create(
                [
                    RetrievedChunk(
                        result=row,
                        rank=m.rank,
                        chunk_id=m.chunk_id,
                        score=m.score,
                        grade=m.grade,
                        is_relevant=m.is_relevant,
                        within_k=m.within_k,
                    )
                    for row, outcome in zip(rows, outcomes, strict=True)
                    if outcome.match is not None
                    for m in outcome.match.ranked
                ],
                batch_size=1000,
            )
            run.query_count = len(outcomes)
            run.evaluated_count = len(scored)
            run.error_count = len(outcomes) - len(scored)
            run.avg_precision = averages["precision"]
            run.avg_recall = averages["recall"]
            run.avg_f1 = averages["f1"]
            run.avg_hit_rate = averages["hit_rate"]
            run.mrr = averages["mrr"]
            run.avg_ndcg = averages["ndcg"]
            run.status = EvaluationRun.Status.COMPLETED
            run.finished_at = timezone.now()
            run.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            run.save()
    except RetrieverError as exc:
        _mark_failed(run, str(exc), started)
    except Exception as exc:  # keep the run record honest, then let the caller see the crash in logs
        logger.exception("Evaluation run %s crashed", run.pk)
        _mark_failed(run, f"Unexpected error: {exc.__class__.__name__}: {exc}", started)
    return run


def _mark_failed(run: EvaluationRun, message: str, started: float) -> None:
    run.status = EvaluationRun.Status.FAILED
    run.error_message = message
    run.finished_at = timezone.now()
    run.duration_ms = round((time.perf_counter() - started) * 1000, 2)
    run.save(update_fields=["status", "error_message", "finished_at", "duration_ms"])


def rerun(run: EvaluationRun, secrets: dict | None = None) -> EvaluationRun:
    """A new run with the same dataset and retriever configuration (current ground truth)."""
    new_run = create_run(
        run.dataset,
        name=f"{run.name} (re-run)"[:200],
        top_k=run.top_k,
        retriever_type=run.retriever_type,
        retriever_config=run.retriever_config,
        notes=f"Re-run of #{run.pk}. {run.notes}".strip(),
        ndcg_gain=run.ndcg_gain,
    )
    return execute_run(new_run, secrets)
