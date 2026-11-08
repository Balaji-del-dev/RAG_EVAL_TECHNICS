"""Dataset management and ground-truth management."""

from pathlib import Path

from django.contrib import messages
from django.db import transaction
from django.db.models import Count, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.text import slugify
from django.views.decorators.http import require_http_methods, require_POST

from ..forms import DatasetUploadForm, GroundTruthForm, QueryEditForm, QueryForm
from ..models import EvaluationDataset, EvaluationQuery, GroundTruth
from ..services.dataset_loader import DatasetFormatError, export_dataset, import_dataset, parse_dataset_file

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
SAMPLE_FILES = ("hr_worked_example.json", "hr_leave_example.json", "hr_policy_v1.json")


def load_sample_datasets() -> tuple[list[EvaluationDataset], list[str]]:
    """Import the bundled sample datasets that are not loaded yet."""
    created, skipped = [], []
    for filename in SAMPLE_FILES:
        parsed = parse_dataset_file(filename, (SAMPLE_DIR / filename).read_bytes())
        if EvaluationDataset.objects.filter(name=parsed.name, version=parsed.version).exists():
            skipped.append(f"{parsed.name} v{parsed.version}")
            continue
        created.append(import_dataset(parsed, source_filename=filename))
    return created, skipped


@require_http_methods(["GET", "POST"])
def dataset_list(request):
    form = DatasetUploadForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        parsed = form.cleaned_data["parsed"]
        try:
            dataset = import_dataset(
                parsed,
                name=form.cleaned_data["name"],
                version=form.cleaned_data["version"],
                description=form.cleaned_data["description"],
                source_filename=form.cleaned_data["file"].name,
            )
        except DatasetFormatError as exc:
            for error in exc.errors:
                form.add_error(None, error)
        else:
            messages.success(
                request, f"Imported {dataset}: {len(parsed.queries)} queries, {len(parsed.chunks)} chunks."
            )
            for warning in parsed.warnings:
                messages.warning(request, warning)
            return redirect(dataset)

    datasets = EvaluationDataset.objects.annotate(
        query_total=Count("queries", distinct=True),
        chunk_total=Count("chunks", distinct=True),
        run_total=Count("runs", distinct=True),
    )
    return render(request, "retrieval_evaluation/dataset.html", {"form": form, "datasets": datasets})


@require_POST
def load_sample(request):
    created, skipped = load_sample_datasets()
    for dataset in created:
        messages.success(request, f"Loaded sample dataset {dataset}.")
    if skipped:
        messages.info(request, "Already loaded: " + ", ".join(skipped) + ".")
    return redirect("retrieval_evaluation:dataset_list")


@require_http_methods(["GET", "POST"])
def dataset_detail(request, pk):
    dataset = get_object_or_404(EvaluationDataset, pk=pk)
    form = QueryForm(request.POST or None, dataset=dataset)
    if request.method == "POST" and form.is_valid():
        with transaction.atomic():
            query = EvaluationQuery.objects.create(
                dataset=dataset, query_id=form.cleaned_data["query_id"], text=form.cleaned_data["text"].strip()
            )
            GroundTruth.objects.create(
                query=query, chunk_id=form.cleaned_data["chunk_id"].strip(), grade=form.cleaned_data["grade"]
            )
        messages.success(request, f"Added query {query.query_id}.")
        return redirect("retrieval_evaluation:query_detail", pk=dataset.pk, query_pk=query.pk)

    queries = dataset.queries.prefetch_related("ground_truth").annotate(
        relevant_total=Count("ground_truth", filter=Q(ground_truth__grade__gte=1))
    )
    chunk_titles = dict(dataset.chunks.values_list("chunk_id", "title"))
    runs = dataset.runs.all()[:50]
    return render(
        request,
        "retrieval_evaluation/dataset_detail.html",
        {
            "dataset": dataset,
            "queries": queries,
            "chunk_titles": chunk_titles,
            "chunk_count": len(chunk_titles),
            "label_count": GroundTruth.objects.filter(query__dataset=dataset).count(),
            "runs": runs,
            "form": form,
            "unlabelled": [q for q in queries if not q.relevant_total],
        },
    )


@require_POST
def dataset_delete(request, pk):
    dataset = get_object_or_404(EvaluationDataset, pk=pk)
    name, run_count = str(dataset), dataset.runs.count()
    dataset.delete()
    messages.success(request, f"Deleted {name} and its {run_count} evaluation run(s).")
    return redirect("retrieval_evaluation:dataset_list")


def dataset_export(request, pk):
    dataset = get_object_or_404(EvaluationDataset, pk=pk)
    response = JsonResponse(export_dataset(dataset), json_dumps_params={"indent": 2, "ensure_ascii": False})
    filename = f"{slugify(dataset.name) or 'dataset'}_v{slugify(dataset.version) or '1'}.json"
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@require_http_methods(["GET", "POST"])
def query_detail(request, pk, query_pk):
    """Ground-truth management for one query: edit the question, add, regrade or remove labels."""
    query = get_object_or_404(EvaluationQuery.objects.select_related("dataset"), pk=query_pk, dataset_id=pk)
    dataset = query.dataset
    label_form = GroundTruthForm(prefix="label")
    edit_form = QueryEditForm(prefix="edit", initial={"text": query.text})

    if request.method == "POST":
        action = request.POST.get("action")
        if action == "save_label":
            label_form = GroundTruthForm(request.POST, prefix="label")
            if label_form.is_valid():
                label, created = GroundTruth.objects.update_or_create(
                    query=query,
                    chunk_id=label_form.cleaned_data["chunk_id"],
                    defaults={"grade": label_form.cleaned_data["grade"]},
                )
                verb = "Added" if created else "Updated"
                messages.success(request, f"{verb} {label.chunk_id} with grade {label.grade}.")
                if not dataset.chunks.filter(chunk_id=label.chunk_id).exists() and dataset.chunks.exists():
                    messages.warning(
                        request, f"{label.chunk_id} is not in this dataset's chunk corpus; "
                                 "the built-in retrievers can never return it."
                    )
                return redirect(request.path)
        elif action == "delete_label":
            label = get_object_or_404(GroundTruth, pk=request.POST.get("label_id"), query=query)
            label.delete()
            messages.success(request, f"Removed {label.chunk_id}.")
            return redirect(request.path)
        elif action == "edit_query":
            edit_form = QueryEditForm(request.POST, prefix="edit")
            if edit_form.is_valid():
                query.text = edit_form.cleaned_data["text"].strip()
                query.save(update_fields=["text"])
                messages.success(request, "Saved the question.")
                return redirect(request.path)
        else:
            messages.error(request, "Unknown action.")

    labels = list(query.ground_truth.all())
    chunks = {c.chunk_id: c for c in dataset.chunks.filter(chunk_id__in=[label.chunk_id for label in labels])}
    rows = [{"label": label, "chunk": chunks.get(label.chunk_id)} for label in labels]
    recent = query.metric_results.select_related("run").order_by("-run__created_at")[:10]
    return render(
        request,
        "retrieval_evaluation/query_detail.html",
        {
            "dataset": dataset,
            "query": query,
            "rows": rows,
            "has_relevant": any(label.grade >= 1 for label in labels),
            "label_form": label_form,
            "edit_form": edit_form,
            "recent_results": recent,
            "chunk_options": dataset.chunks.values_list("chunk_id", "title")[:5000],
        },
    )


@require_POST
def query_delete(request, pk, query_pk):
    query = get_object_or_404(EvaluationQuery, pk=query_pk, dataset_id=pk)
    query_id = query.query_id
    query.delete()
    messages.success(request, f"Deleted query {query_id}. Past runs keep their scores for it.")
    return redirect("retrieval_evaluation:dataset_detail", pk=pk)
