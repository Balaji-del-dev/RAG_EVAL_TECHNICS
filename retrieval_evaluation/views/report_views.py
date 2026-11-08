"""Dashboard, results, query-level failure analysis, exports and experiment comparison."""

import csv

from django.conf import settings
from django.db.models import Count
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.utils.text import slugify

from ..analysis.experiment_comparison import compare_runs
from ..analysis.run_explanation import explain_run
from ..analysis.run_lesson import build_lesson
from ..analysis.failure_analysis import (
    ISSUES,
    STATUS_LABELS,
    precision_ceiling,
    query_status,
    recall_ceiling,
    summarize_run,
)
from ..forms import ComparisonForm
from ..metrics import METRICS, explain_query_metrics
from ..models import EvaluationDataset, EvaluationQuery, EvaluationRun, MetricResult


def metric_cards(run: EvaluationRun) -> list[dict]:
    """The six run averages with their metric metadata, in display order."""
    averages = run.averages()
    return [{"metric": m, "label": m.label(run.top_k), "value": averages[m.key]} for m in METRICS]


def dashboard(request):
    latest = EvaluationRun.objects.filter(status=EvaluationRun.Status.COMPLETED).select_related("dataset").first()
    recent = EvaluationRun.objects.select_related("dataset")[:8]
    datasets = EvaluationDataset.objects.annotate(
        query_total=Count("queries", distinct=True), run_total=Count("runs", distinct=True)
    )[:6]
    context = {
        "latest": latest,
        "latest_cards": metric_cards(latest) if latest else [],
        "latest_summary": summarize_run(latest) if latest else None,
        "recent_runs": recent,
        "datasets": datasets,
        "totals": {
            "datasets": EvaluationDataset.objects.count(),
            "queries": EvaluationQuery.objects.count(),
            "runs": EvaluationRun.objects.filter(status=EvaluationRun.Status.COMPLETED).count(),
        },
    }
    return render(request, "retrieval_evaluation/dashboard.html", context)


def run_list(request):
    runs = EvaluationRun.objects.select_related("dataset")
    dataset = None
    dataset_pk = request.GET.get("dataset", "")
    if dataset_pk.isdigit():
        dataset = get_object_or_404(EvaluationDataset, pk=dataset_pk)
        runs = runs.filter(dataset=dataset)
    return render(
        request,
        "retrieval_evaluation/runs.html",
        {"runs": runs[:500], "dataset": dataset, "datasets": EvaluationDataset.objects.all(), "metrics": METRICS},
    )


def run_detail(request, pk):
    run = get_object_or_404(EvaluationRun.objects.select_related("dataset"), pk=pk)
    results = list(run.results.all())
    for result in results:
        result.status = query_status(result)

    status_filter = request.GET.get("status", "")
    issue_filter = request.GET.get("issue", "")
    shown = results
    if status_filter in STATUS_LABELS:
        shown = [r for r in shown if r.status == status_filter]
    if issue_filter in ISSUES:
        shown = [r for r in shown if issue_filter in (r.issues or [])]

    previous = (
        EvaluationRun.objects.filter(
            dataset=run.dataset, status=EvaluationRun.Status.COMPLETED, created_at__lt=run.created_at
        )
        .exclude(pk=run.pk)
        .first()
    )
    return render(
        request,
        "retrieval_evaluation/results.html",
        {
            "run": run,
            "cards": metric_cards(run) if run.is_completed else [],
            "summary": summarize_run(run) if run.is_completed else None,
            "results": shown,
            "result_total": len(results),
            "status_filter": status_filter,
            "issue_filter": issue_filter,
            "status_labels": STATUS_LABELS,
            "issues": ISSUES,
            "previous_run": previous,
            "metrics": METRICS,
        },
    )


def _int_param(request, name):
    value = request.GET.get(name, "")
    return int(value) if value.isdigit() else None


def run_explain(request, pk):
    """One page that explains the run's six scores step by step, with its own numbers.

    ?q=<result id> picks the example question, ?fail=<result id> the failed one.
    """
    run = get_object_or_404(EvaluationRun.objects.select_related("dataset"), pk=pk)
    if not run.is_completed:
        raise Http404("This run has no scores to explain.")
    return render(
        request,
        "retrieval_evaluation/explain.html",
        {
            "run": run,
            "cards": metric_cards(run),
            "x": explain_run(run),
            "lesson": build_lesson(run, _int_param(request, "q"), _int_param(request, "fail")),
        },
    )


def result_detail(request, pk, result_pk):
    """How one query was scored: ranked chunks vs ground truth, and the working behind each metric."""
    result = get_object_or_404(MetricResult.objects.select_related("run__dataset"), pk=result_pk, run_id=pk)
    run = result.run
    retrieved = list(result.retrieved.all())
    chunk_ids = {r.chunk_id for r in retrieved} | set(result.expected)
    chunks = {c.chunk_id: c for c in run.dataset.chunks.filter(chunk_id__in=chunk_ids)}

    rank_of = {r.chunk_id: r.rank for r in retrieved}
    expected_rows = [
        {
            "chunk_id": cid,
            "grade": grade,
            "chunk": chunks.get(cid),
            "rank": rank_of.get(cid),
            "found": cid in rank_of and rank_of[cid] <= run.top_k,
        }
        for cid, grade in sorted(result.expected.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    explanation = None
    if not result.error:
        explanation = explain_query_metrics([r.chunk_id for r in retrieved], result.expected, run.top_k, run.ndcg_gain)

    siblings = list(run.results.values_list("pk", flat=True))
    index = siblings.index(result.pk)
    return render(
        request,
        "retrieval_evaluation/result_detail.html",
        {
            "run": run,
            "result": result,
            "status": query_status(result),
            "status_labels": STATUS_LABELS,
            "retrieved": [{"row": r, "chunk": chunks.get(r.chunk_id)} for r in retrieved],
            "expected_rows": expected_rows,
            "explanation": explanation,
            "metrics": METRICS,
            "issues": [ISSUES[code] for code in result.issues or [] if code in ISSUES],
            "precision_ceiling": precision_ceiling(result.total_relevant, run.top_k) if result.total_relevant else None,
            "recall_ceiling": recall_ceiling(result.total_relevant, run.top_k) if result.total_relevant else None,
            "prev_pk": siblings[index - 1] if index > 0 else None,
            "next_pk": siblings[index + 1] if index + 1 < len(siblings) else None,
            "position": index + 1,
            "total": len(siblings),
        },
    )


def _safe_cell(value):
    """Stop spreadsheet apps from running a cell that starts like a formula."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def run_export(request, pk, fmt):
    run = get_object_or_404(EvaluationRun.objects.select_related("dataset"), pk=pk)
    results = run.results.prefetch_related("retrieved")
    base = f"run_{run.pk}_{slugify(run.name) or 'evaluation'}"
    if fmt == "json":
        payload = {
            "run": {
                "id": run.pk,
                "name": run.name,
                "dataset": str(run.dataset),
                "top_k": run.top_k,
                "retriever_type": run.retriever_type,
                "retriever": run.retriever_summary(),
                "ndcg_gain": run.ndcg_gain,
                "status": run.status,
                "created_at": run.created_at.isoformat(),
                "query_count": run.query_count,
                "evaluated_count": run.evaluated_count,
                "error_count": run.error_count,
                "averages": run.averages(),
            },
            "queries": [
                {
                    "query_id": r.query_external_id,
                    "query": r.query_text,
                    "scores": r.scores(),
                    "expected": r.expected,
                    "retrieved": [
                        {"rank": c.rank, "chunk_id": c.chunk_id, "score": c.score, "grade": c.grade}
                        for c in r.retrieved.all()
                    ],
                    "missed": r.missed_chunk_ids,
                    "issues": r.issues,
                    "error": r.error,
                }
                for r in results
            ],
        }
        response = JsonResponse(payload, json_dumps_params={"indent": 2, "ensure_ascii": False})
        response["Content-Disposition"] = f'attachment; filename="{base}.json"'
        return response
    if fmt != "csv":
        raise Http404("Unknown export format")

    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="{base}.csv"'
    writer = csv.writer(response)
    writer.writerow([
        "query_id", "query", "status", f"precision@{run.top_k}", f"recall@{run.top_k}", "f1",
        f"hit_rate@{run.top_k}", "reciprocal_rank", f"ndcg@{run.top_k}", "relevant_found", "relevant_missed",
        "irrelevant_found", "first_relevant_rank", "top_k_chunk_ids", "missed_chunk_ids", "issues", "error",
    ])
    for r in results:
        top = [c.chunk_id for c in r.retrieved.all() if c.within_k]
        writer.writerow([_safe_cell(v) for v in (
            r.query_external_id, r.query_text, query_status(r), r.precision, r.recall, r.f1, r.hit_rate,
            r.reciprocal_rank, r.ndcg, r.relevant_found, r.relevant_missed, r.irrelevant_found,
            r.first_relevant_rank, " ".join(top), " ".join(r.missed_chunk_ids), " ".join(r.issues), r.error,
        )])
    return response


def comparison(request):
    data = request.GET.copy()
    candidate_pk = data.get("candidate", "")
    if candidate_pk.isdigit() and not data.get("baseline"):
        candidate = EvaluationRun.objects.filter(pk=candidate_pk).first()
        if candidate:
            earlier = (
                EvaluationRun.objects.filter(dataset=candidate.dataset, status=EvaluationRun.Status.COMPLETED)
                .exclude(pk=candidate.pk)
                .filter(created_at__lte=candidate.created_at)
                .first()
            )
            if earlier:
                data["baseline"] = str(earlier.pk)
    bound = bool(data.get("baseline") and data.get("candidate"))
    if bound and not data.get("tolerance"):
        data["tolerance"] = str(settings.RETRIEVAL_EVAL["REGRESSION_TOLERANCE"])
    form = ComparisonForm(data if bound else None, initial={"candidate": candidate_pk or None})
    result = None
    if bound and form.is_valid():
        result = compare_runs(
            form.cleaned_data["baseline"], form.cleaned_data["candidate"], form.cleaned_data.get("tolerance")
        )
    chart = []
    if result:
        chart = [
            {
                "label": m.label,
                "display": m.display,
                "baseline": m.baseline,
                "candidate": m.candidate,
                "delta": m.delta,
                "verdict": m.verdict,
            }
            for m in result.metrics
        ]
    return render(
        request,
        "retrieval_evaluation/comparison.html",
        {
            "form": form,
            "comparison": result,
            "chart": chart,
            "metrics": METRICS,
            "has_runs": EvaluationRun.objects.filter(status=EvaluationRun.Status.COMPLETED).count() >= 2,
        },
    )
