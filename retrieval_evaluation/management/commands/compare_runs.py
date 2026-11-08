from django.core.management.base import BaseCommand, CommandError

from ...analysis.experiment_comparison import compare_runs
from ...models import EvaluationRun


class Command(BaseCommand):
    help = (
        "Compare a candidate run with a baseline run. With --fail-on-regression, exits with status 1 "
        "when any metric average drops by more than the tolerance (for CI regression tests)."
    )

    def add_arguments(self, parser):
        parser.add_argument("baseline", type=int, help="Baseline run id.")
        parser.add_argument("candidate", type=int, help="Candidate run id.")
        parser.add_argument("--tolerance", type=float, default=None)
        parser.add_argument("--fail-on-regression", action="store_true")

    def handle(self, *args, **o):
        runs = {r.pk: r for r in EvaluationRun.objects.filter(pk__in=[o["baseline"], o["candidate"]])}
        for key in ("baseline", "candidate"):
            run = runs.get(o[key])
            if not run:
                raise CommandError(f"Run {o[key]} not found.")
            if not run.is_completed:
                raise CommandError(f"Run {o[key]} is {run.status}, not completed.")
        result = compare_runs(runs[o["baseline"]], runs[o["candidate"]], o["tolerance"])
        for warning in result.warnings:
            self.stdout.write(self.style.WARNING(warning))
        self.stdout.write(f"{'Metric':<14}{'Baseline':>10}{'Candidate':>11}{'Change':>10}")
        for m in result.metrics:
            fmt = (lambda v: f"{v * 100:.1f}%") if m.display == "percent" else (lambda v: f"{v:.3f}")
            change = f"{m.delta * 100:+.1f}pt" if m.display == "percent" else f"{m.delta:+.3f}"
            self.stdout.write(f"{m.label:<14}{fmt(m.baseline):>10}{fmt(m.candidate):>11}{change:>10}  {m.verdict}")
        counts = result.query_counts()
        self.stdout.write(
            f"Queries: {result.common_count} compared, {counts['improved']} improved, "
            f"{counts['regressed']} regressed (NDCG, tolerance {result.tolerance})"
        )
        style = self.style.ERROR if result.is_regression else self.style.SUCCESS
        self.stdout.write(style(f"Verdict: {result.verdict}"))
        if o["fail_on_regression"] and result.is_regression:
            raise CommandError("Regression detected.", returncode=1)
