import json
import os
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from ...metrics import METRICS
from ...models import EvaluationDataset, EvaluationRun
from ...services.evaluation_runner import create_run, execute_run
from ...services.retrieval_adapter import ResultsFormatError, RetrieverError, parse_results_file, validate_http_url


class Command(BaseCommand):
    help = "Run a retrieval evaluation and print the six average scores."

    def add_arguments(self, parser):
        parser.add_argument("--dataset", required=True, help="Dataset id, or 'name' / 'name@version'.")
        parser.add_argument("--k", type=int, default=5, help="Top K (default 5).")
        parser.add_argument("--retriever", required=True, choices=[c for c, _ in EvaluationRun.Retriever.choices])
        parser.add_argument("--name", default="", help="Experiment name.")
        parser.add_argument("--results", type=Path, help="Results file for --retriever precomputed.")
        parser.add_argument("--url", help="Endpoint for --retriever http.")
        parser.add_argument("--timeout", type=float, default=None, help="HTTP timeout in seconds.")
        parser.add_argument("--auth-env", help="Environment variable holding the HTTP Authorization header.")
        parser.add_argument("--k1", type=float, default=1.5)
        parser.add_argument("--b", type=float, default=0.75)
        parser.add_argument("--rrf-k", type=int, default=60)
        parser.add_argument("--stemming", action="store_true")
        parser.add_argument("--notes", default="")
        parser.add_argument("--json", action="store_true", help="Print the result as JSON.")

    def _dataset(self, ref: str) -> EvaluationDataset:
        if ref.isdigit():
            dataset = EvaluationDataset.objects.filter(pk=int(ref)).first()
        else:
            name, _, version = ref.partition("@")
            qs = EvaluationDataset.objects.filter(name=name)
            dataset = (qs.filter(version=version) if version else qs).first()
        if not dataset:
            raise CommandError(f"Dataset {ref!r} not found.")
        return dataset

    def handle(self, *args, **o):
        if not 1 <= o["k"] <= 100:
            raise CommandError("--k must be between 1 and 100.")
        dataset = self._dataset(o["dataset"])
        kind = o["retriever"]
        config: dict = {}
        secrets: dict = {}
        if kind in ("bm25", "tfidf", "fusion"):
            config = {"stemming": o["stemming"]}
            if kind in ("bm25", "fusion"):
                config.update(k1=o["k1"], b=o["b"])
            if kind == "fusion":
                config["rrf_k"] = o["rrf_k"]
        elif kind == "http":
            try:
                config = {"url": validate_http_url(o["url"] or ""), "timeout": o["timeout"]}
            except RetrieverError as exc:
                raise CommandError(str(exc)) from exc
            if o["auth_env"]:
                secrets["authorization"] = os.environ.get(o["auth_env"], "")
        elif kind == "precomputed":
            if not o["results"] or not o["results"].is_file():
                raise CommandError("--retriever precomputed needs --results FILE.")
            try:
                config = {
                    "results": parse_results_file(o["results"].name, o["results"].read_bytes()),
                    "source_filename": o["results"].name,
                }
            except ResultsFormatError as exc:
                raise CommandError(str(exc)) from exc

        run = create_run(
            dataset,
            name=o["name"] or f"{kind} K={o['k']}",
            top_k=o["k"],
            retriever_type=kind,
            retriever_config=config,
            notes=o["notes"],
        )
        execute_run(run, secrets)
        if o["json"]:
            self.stdout.write(json.dumps({
                "run_id": run.pk, "status": run.status, "error": run.error_message,
                "evaluated": run.evaluated_count, "errors": run.error_count, "averages": run.averages(),
            }, indent=2))
        elif run.is_completed:
            self.stdout.write(self.style.SUCCESS(
                f"Run #{run.pk} {run.name}: {run.evaluated_count}/{run.query_count} queries evaluated"
            ))
            averages = run.averages()
            for m in METRICS:
                value = averages[m.key]
                text = f"{value * 100:.1f}%" if m.display == "percent" else f"{value:.3f}"
                self.stdout.write(f"  {m.label(run.top_k):<14} {text}")
        if not run.is_completed:
            raise CommandError(f"Run #{run.pk} failed: {run.error_message}")
