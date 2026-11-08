"""Command line: `generation-eval tree | metrics | run FILE`.

    generation-eval tree                         the production tree (metrics, formulas, frameworks)
    generation-eval metrics                      one table: metric, priority, formula, framework, threshold
    generation-eval run data/hr_leave_samples.json [--metrics bleu,rouge] [--engine faithfulness=deepeval]
                                                 [--out report.json] [--no-track]
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .catalog import METRICS, METRICS_BY_KEY, production_tree
from .config import get_settings
from .schemas import BatchReport, EvaluationSample


def _parse_engines(values: list[str]) -> dict[str, str]:
    engines = {}
    for item in values:
        metric, _, engine = item.partition("=")
        if metric not in METRICS_BY_KEY or not engine:
            raise SystemExit(f"--engine expects METRIC=ENGINE, e.g. faithfulness=deepeval (got {item!r})")
        engines[metric] = engine
    return engines


def _load_samples(path: Path) -> list[EvaluationSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["samples"] if isinstance(data, dict) else data
    try:
        return TypeAdapter(list[EvaluationSample]).validate_python(rows)
    except ValidationError as exc:
        raise SystemExit(f"{path} is not a valid sample file:\n{exc}") from exc


def _fmt(value: float | None) -> str:
    return "  —  " if value is None else f"{value:.3f}"


def print_metrics() -> None:
    settings = get_settings()
    print(f"{'No.':<4} {'Metric':<20} {'Priority':<9} {'Framework':<24} {'Pass at':<8} Formula")
    for m in METRICS:
        engine = settings.default_engine(m.key) if m.llm_judged else m.frameworks[0]
        frameworks = " / ".join(m.frameworks)
        print(f"{m.number:<4} {m.label:<20} {m.priority:<9} {frameworks + f' [{engine}]':<24} "
              f">= {settings.threshold_for(m.key):<5} {m.formula}")


def print_batch(batch: BatchReport) -> None:
    for report in batch.reports:
        mark = {"pass": "PASS", "fail": "FAIL", "incomplete": "INCOMPLETE"}[report.status]
        print(f"\n[{mark}] {report.sample_id}: {report.generated_answer}")
        print(f"  overall {_fmt(report.overall_score)}   {report.seconds:.1f}s   tracking: {report.tracking}")
        for s in report.scores:
            result = "pass" if s.passed else "fail" if s.passed is False else s.status
            note = s.explanation if s.status == "scored" else s.error
            print(f"  {s.number} {s.label:<20} {s.framework:<22} {_fmt(s.score)} (>= {s.threshold}) {result:<7} {note}")
        if report.trace_url:
            print(f"  Langfuse: {report.trace_url}")
    print(f"\n{batch.passed}/{batch.total} passed, {batch.failed} failed, {batch.incomplete} incomplete")
    print("Averages: " + ", ".join(f"{k} {_fmt(v)}" for k, v in batch.metric_averages.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generation-eval", description="Evaluate RAG answers with 8 generation metrics.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tree", help="Show the production tree.")
    sub.add_parser("metrics", help="Show the metric table.")
    run = sub.add_parser("run", help="Evaluate a JSON file of samples.")
    run.add_argument("file", type=Path)
    run.add_argument("--metrics", help="Comma-separated metric keys (default: all eight).")
    run.add_argument("--engine", action="append", default=[], help="METRIC=ENGINE, e.g. faithfulness=deepeval")
    run.add_argument("--out", type=Path, help="Write the full JSON report here.")
    run.add_argument("--no-track", action="store_true", help="Do not send scores to Langfuse.")
    args = parser.parse_args(argv)

    if args.command == "tree":
        print(production_tree())
        return 0
    if args.command == "metrics":
        print_metrics()
        return 0

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    from .evaluation import GenerationEvaluator

    metrics = [m.strip() for m in args.metrics.split(",")] if args.metrics else None
    if metrics and (unknown := [m for m in metrics if m not in METRICS_BY_KEY]):
        raise SystemExit(f"Unknown metrics: {', '.join(unknown)}. Choose from: {', '.join(METRICS_BY_KEY)}")
    samples = _load_samples(args.file)
    evaluator = GenerationEvaluator()
    try:
        batch = asyncio.run(evaluator.evaluate_batch(samples, metrics, _parse_engines(args.engine), not args.no_track))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print_batch(batch)
    if args.out:
        args.out.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
        print(f"Report written to {args.out}")
    return 0 if batch.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
