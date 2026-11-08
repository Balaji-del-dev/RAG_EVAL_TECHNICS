"""Create and run evaluations."""

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods, require_POST

from ..forms import EvaluationForm
from ..models import EvaluationDataset, EvaluationRun
from ..services.evaluation_runner import create_run, execute_run, rerun


def _report(request, run: EvaluationRun) -> None:
    if run.status == EvaluationRun.Status.COMPLETED:
        text = f"Evaluated {run.evaluated_count} of {run.query_count} queries in {run.duration_ms:.0f} ms."
        if run.error_count:
            messages.warning(request, text + f" {run.error_count} query(ies) could not be evaluated; see the list below.")
        else:
            messages.success(request, text)
    else:
        messages.error(request, f"The run failed: {run.error_message}")


@require_http_methods(["GET", "POST"])
def evaluation_create(request):
    initial = {}
    dataset_pk = request.GET.get("dataset")
    if dataset_pk and dataset_pk.isdigit():
        initial["dataset"] = EvaluationDataset.objects.filter(pk=dataset_pk).first()
    form = EvaluationForm(request.POST or None, request.FILES or None, initial=initial)
    if request.method == "POST" and form.is_valid():
        data = form.cleaned_data
        run = create_run(
            data["dataset"],
            name=data["name"].strip(),
            top_k=data["top_k"],
            retriever_type=data["retriever_type"],
            retriever_config=data["retriever_config"],
            notes=data["notes"].strip(),
        )
        # Runs synchronously: fine for evaluation sets of a few hundred queries.
        execute_run(run, secrets=data["secrets"])
        _report(request, run)
        return redirect(run)

    has_datasets = EvaluationDataset.objects.exists()
    datasets_without_chunks = list(
        EvaluationDataset.objects.filter(chunks__isnull=True).values_list("pk", flat=True)
    )
    return render(
        request,
        "retrieval_evaluation/evaluation.html",
        {"form": form, "has_datasets": has_datasets, "datasets_without_chunks": datasets_without_chunks},
    )


@require_POST
def run_rerun(request, pk):
    run = get_object_or_404(EvaluationRun, pk=pk)
    if run.retriever_type == EvaluationRun.Retriever.HTTP:
        messages.info(request, "Re-run sent without an Authorization header (headers are never stored).")
    new_run = rerun(run)
    _report(request, new_run)
    return redirect(new_run)


@require_POST
def run_delete(request, pk):
    run = get_object_or_404(EvaluationRun, pk=pk)
    label = str(run)
    run.delete()
    messages.success(request, f"Deleted run {label}.")
    return redirect("retrieval_evaluation:run_list")
