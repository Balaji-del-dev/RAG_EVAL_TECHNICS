"""GenerationEvaluator: one sample in, one EvaluationReport out.

Order of work for a sample:
  1. Text metrics (Exact Match, BLEU, ROUGE): instant, pure Python libraries.
  2. Semantic Similarity: Sentence Transformers embeddings (in a worker thread).
  3. LLM-judged metrics, one at a time: on a CPU the judge model is the bottleneck,
     and running them in parallel only adds memory pressure.
A metric that fails or lacks an input is reported as "error" or "skipped"; the
others still run.
"""

import asyncio
import logging
import time
from collections.abc import Iterable

from ..catalog import METRICS, METRICS_BY_KEY, MetricSpec
from ..config import Settings, get_settings
from ..judges import Judge, build_judges
from ..metrics import TEXT_METRICS, SentenceTransformerSimilarity
from ..schemas import BatchReport, EvaluationReport, EvaluationSample, MetricScore
from ..tracking import LangfuseTracker
from .scoring import decide_status, overall_score

logger = logging.getLogger(__name__)

# Cheap metrics first, LLM-judged last.
RUN_ORDER = ("exact_match", "bleu", "rouge", "semantic_similarity",
             "faithfulness", "answer_relevancy", "answer_correctness", "answer_completeness")


class GenerationEvaluator:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        judges: dict[str, Judge] | None = None,
        similarity: SentenceTransformerSimilarity | None = None,
        tracker: LangfuseTracker | None = None,
    ):
        self.settings = settings or get_settings()
        self.judges = judges if judges is not None else build_judges(self.settings)
        self.similarity = similarity or SentenceTransformerSimilarity(self.settings.sentence_model, self.settings.torch_threads)
        self.tracker = tracker or LangfuseTracker(self.settings)
        self._slots = asyncio.Semaphore(self.settings.max_concurrent_evaluations)

    # ------------------------------------------------------------------ public
    def engine_for(self, metric: str, overrides: dict[str, str] | None = None) -> str:
        spec = METRICS_BY_KEY[metric]
        if not spec.llm_judged:
            return spec.frameworks[0]
        engine = (overrides or {}).get(metric) or self.settings.default_engine(metric) or spec.frameworks[0]
        if engine not in spec.frameworks:
            raise ValueError(f"{metric} can use {' or '.join(spec.frameworks)}, not {engine}.")
        return engine

    async def evaluate(
        self,
        sample: EvaluationSample,
        metrics: Iterable[str] | None = None,
        engines: dict[str, str] | None = None,
        track: bool = True,
    ) -> EvaluationReport:
        wanted = set(metrics) if metrics else set(RUN_ORDER)
        unknown = wanted - set(METRICS_BY_KEY)
        if unknown:
            raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}")
        plan = [(METRICS_BY_KEY[key], self.engine_for(key, engines)) for key in RUN_ORDER if key in wanted]

        async with self._slots:
            started = time.perf_counter()
            scores = [await self._run_metric(spec, engine, sample) for spec, engine in plan]
            scores.sort(key=lambda s: s.number)
            status, failed = decide_status(scores)
            report = EvaluationReport(
                sample_id=sample.sample_id,
                question=sample.question,
                generated_answer=sample.generated_answer,
                status=status,
                passed=status == "pass",
                overall_score=overall_score(scores),
                failed_metrics=failed,
                scores=scores,
                judge_model=self.settings.judge_model,
                tracking="disabled",
                seconds=round(time.perf_counter() - started, 3),
            )
        await self._track(sample, report, track)
        return report

    async def evaluate_batch(
        self,
        samples: list[EvaluationSample],
        metrics: Iterable[str] | None = None,
        engines: dict[str, str] | None = None,
        track: bool = True,
    ) -> BatchReport:
        metrics = list(metrics) if metrics else None
        reports = [await self.evaluate(s, metrics, engines, track) for s in samples]
        averages: dict[str, float | None] = {}
        for spec in METRICS:
            values = [m.score for r in reports for m in r.scores if m.metric == spec.key and m.status == "scored"]
            if any(m.metric == spec.key for r in reports for m in r.scores):
                averages[spec.key] = sum(values) / len(values) if values else None
        passed = sum(r.status == "pass" for r in reports)
        return BatchReport(
            total=len(reports),
            passed=passed,
            failed=sum(r.status == "fail" for r in reports),
            incomplete=sum(r.status == "incomplete" for r in reports),
            pass_rate=passed / len(reports),
            metric_averages=averages,
            reports=reports,
        )

    # ----------------------------------------------------------------- private
    async def _run_metric(self, spec: MetricSpec, engine: str, sample: EvaluationSample) -> MetricScore:
        threshold = self.settings.threshold_for(spec.key)
        base = dict(metric=spec.key, number=spec.number, label=spec.label, priority=spec.priority,
                    gating=spec.gating, framework=engine, threshold=threshold, formula=spec.formula)
        missing = spec.missing_inputs(sample)
        if spec.key == "answer_completeness" and sample.required_facts:
            missing = [m for m in missing if m != "reference_answer"]  # required_facts are enough
        if missing:
            return MetricScore(**base, status="skipped", error=f"Needs {' and '.join(missing)}.")

        started = time.perf_counter()
        try:
            if spec.llm_judged:
                outcome = await self.judges[engine].score(spec.key, sample)
            elif spec.key == "semantic_similarity":
                outcome = await asyncio.to_thread(self.similarity.score, sample.generated_answer, sample.reference_answer)
            else:
                outcome = TEXT_METRICS[spec.key](sample.generated_answer, sample.reference_answer)
        except Exception as exc:  # one failing metric must not lose the others
            logger.warning("%s (%s) failed for sample %s: %s", spec.key, engine, sample.sample_id, exc)
            return MetricScore(**base, status="error", error=_short_error(exc),
                               seconds=round(time.perf_counter() - started, 3))

        score = min(1.0, max(0.0, outcome.score))
        return MetricScore(
            **base,
            framework_version=outcome.framework_version,
            status="scored",
            score=score,
            passed=score >= threshold,
            explanation=outcome.explanation,
            details=outcome.details,
            seconds=round(time.perf_counter() - started, 3),
        )

    async def _track(self, sample: EvaluationSample, report: EvaluationReport, track: bool) -> None:
        if not track:
            report.tracking = "off for this request"
            return
        if not self.tracker.enabled:
            report.tracking = "disabled"
            return
        try:
            report.trace_url = await asyncio.to_thread(self.tracker.record, sample, report)
            report.tracking = "sent"
        except Exception as exc:  # tracking must never break an evaluation
            logger.warning("Langfuse tracking failed: %s", exc)
            report.tracking = f"error: {_short_error(exc)}"


def _short_error(exc: Exception) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {text}"[:300] if text else type(exc).__name__
