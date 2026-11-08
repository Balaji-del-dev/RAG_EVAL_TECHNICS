"""Evaluation tracking in Langfuse (SDK v5, OpenTelemetry based).

Each evaluated sample becomes one trace with an "evaluator" observation:
  input   question, answer, retrieved contexts, reference
  output  status, overall score, failed metrics
  scores  one numeric score per metric (comment = the working), plus
          overall_score and passed (1 or 0)
Tracking is off until LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.
"""

import threading

from ..config import Settings
from ..schemas import EvaluationReport, EvaluationSample


class LangfuseTracker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.settings.tracking_enabled and self.settings.langfuse_configured

    def _get_client(self):
        with self._lock:
            if self._client is None:
                from langfuse import Langfuse

                s = self.settings
                self._client = Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host)
            return self._client

    def check(self) -> bool:
        """True when Langfuse accepts the keys (used by /health)."""
        return bool(self.enabled and self._get_client().auth_check())

    def record(self, sample: EvaluationSample, report: EvaluationReport) -> str | None:
        client = self._get_client()
        with client.start_as_current_observation(
            name="generation-evaluation",
            as_type="evaluator",
            input={
                "question": sample.question,
                "generated_answer": sample.generated_answer,
                "retrieved_contexts": sample.retrieved_contexts,
                "reference_answer": sample.reference_answer,
            },
            metadata={"sample_id": sample.sample_id, "judge_model": report.judge_model, **sample.metadata},
        ) as observation:
            observation.update(output={
                "status": report.status,
                "overall_score": report.overall_score,
                "failed_metrics": report.failed_metrics,
            })
            for score in report.scores:
                if score.status == "scored" and score.score is not None:
                    observation.score_trace(
                        name=score.metric, value=score.score,
                        comment=f"[{score.framework}] {score.explanation or ''}"[:1000],
                    )
            if report.overall_score is not None:
                observation.score_trace(name="overall_score", value=report.overall_score)
            observation.score_trace(name="passed", value=1.0 if report.passed else 0.0, comment=report.status)
            trace_id = client.get_current_trace_id()
        client.flush()
        return client.get_trace_url(trace_id=trace_id)
