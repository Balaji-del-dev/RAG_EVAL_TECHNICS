"""Request and report models (Pydantic), shared by the evaluator, the API and the CLI."""

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from .catalog import METRICS_BY_KEY

MetricKey = Literal[
    "faithfulness", "answer_relevancy", "answer_correctness", "answer_completeness",
    "semantic_similarity", "exact_match", "bleu", "rouge",
]
EngineName = Literal["ragas", "deepeval", "custom"]


class EvaluationSample(BaseModel):
    """One question, the context the retriever found, the LLM's answer and the expected answer."""

    sample_id: str = Field(default_factory=lambda: uuid4().hex[:12], max_length=100)
    question: str = Field(min_length=1, max_length=4000)
    generated_answer: str = Field(min_length=1, max_length=8000)
    retrieved_contexts: list[str] = Field(default_factory=list, max_length=50)
    reference_answer: str | None = Field(default=None, max_length=8000)
    # Optional: the details a complete answer must contain. Used by Answer Completeness.
    required_facts: list[str] | None = Field(default=None, max_length=50)
    metadata: dict = Field(default_factory=dict)

    @field_validator("retrieved_contexts")
    @classmethod
    def _drop_blank_contexts(cls, value: list[str]) -> list[str]:
        return [c for c in value if c.strip()]


class _RunOptions(BaseModel):
    metrics: list[MetricKey] | None = Field(default=None, description="Metrics to run; all eight when omitted.")
    engines: dict[MetricKey, EngineName] = Field(
        default_factory=dict, description="Override the framework per LLM-judged metric, e.g. {\"faithfulness\": \"deepeval\"}."
    )
    track: bool = Field(default=True, description="Send the scores to Langfuse (when configured).")

    @model_validator(mode="after")
    def _engines_must_support_metric(self):
        for metric, engine in self.engines.items():
            spec = METRICS_BY_KEY[metric]
            if not spec.llm_judged:
                raise ValueError(f"{metric} is not LLM-judged; it always uses {spec.frameworks[0]}.")
            if engine not in spec.frameworks:
                raise ValueError(f"{metric} can use {' or '.join(spec.frameworks)}, not {engine}.")
        return self


class EvaluateRequest(_RunOptions):
    sample: EvaluationSample


class BatchEvaluateRequest(_RunOptions):
    samples: list[EvaluationSample] = Field(min_length=1, max_length=200)


class MetricScore(BaseModel):
    metric: MetricKey
    number: int
    label: str
    priority: Literal["critical", "high", "low"]
    gating: bool = Field(description="True when this metric decides pass/fail.")
    framework: str
    framework_version: str | None = None
    status: Literal["scored", "skipped", "error"]
    score: float | None = None
    threshold: float
    passed: bool | None = None
    formula: str
    explanation: str | None = None
    details: dict = Field(default_factory=dict)
    error: str | None = None
    seconds: float = 0.0


class EvaluationReport(BaseModel):
    sample_id: str
    question: str
    generated_answer: str
    status: Literal["pass", "fail", "incomplete"]
    passed: bool
    overall_score: float | None = Field(description="Weighted mean of the scored metrics: critical 3, high 2, low 1.")
    failed_metrics: list[MetricKey]
    scores: list[MetricScore]
    judge_model: str
    tracking: str = Field(description="'sent', 'disabled', 'off for this request' or 'error: …'.")
    trace_url: str | None = None
    seconds: float


class BatchReport(BaseModel):
    total: int
    passed: int
    failed: int
    incomplete: int
    pass_rate: float
    metric_averages: dict[MetricKey, float | None]
    reports: list[EvaluationReport]
