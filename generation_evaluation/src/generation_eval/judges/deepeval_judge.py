"""DeepEval adapter, with a local Ollama model.

    faithfulness         FaithfulnessMetric:    claims the context does not contradict / all claims
    answer_relevancy     AnswerRelevancyMetric: relevant statements / all statements in the answer
    answer_correctness   GEval with correctness steps (facts vs the expected output)
    answer_completeness  GEval with completeness steps (required details vs the expected output)

DeepEval metrics keep state between calls, so each call builds a fresh metric.
DeepEval is synchronous here (async_mode=False) and runs in a worker thread.
"""

import asyncio

from ..config import Settings
from ..metrics.base import MetricOutcome, package_version
from ..schemas import EvaluationSample
from .base import check_score

CORRECTNESS_STEPS = [
    "Compare every fact in 'actual output' (numbers, names, dates, conditions) with 'expected output'.",
    "Heavily penalize any fact in 'actual output' that contradicts 'expected output'.",
    "Do not penalize different wording when the meaning is the same.",
    "Do not penalize missing details: this checks correctness, not completeness.",
]
COMPLETENESS_STEPS = [
    "List every detail in 'expected output' that is needed to answer 'input'.",
    "Check whether 'actual output' states each detail, in any wording.",
    "The score is the share of required details that 'actual output' includes.",
    "Do not penalize extra correct details or different wording.",
]


def _verdict(v) -> str:
    value = getattr(v, "verdict", v)
    return str(getattr(value, "value", value)).lower()


class DeepEvalJudge:
    name = "deepeval"
    supports = frozenset({"faithfulness", "answer_relevancy", "answer_correctness", "answer_completeness"})

    def __init__(self, settings: Settings):
        self.settings = settings

    def _build(self, metric: str):
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
        from deepeval.models import OllamaModel
        from deepeval.test_case import SingleTurnParams

        s = self.settings
        model = OllamaModel(model=s.judge_model, base_url=s.ollama_base_url, temperature=s.judge_temperature)
        threshold = s.threshold_for(metric)
        if metric == "faithfulness":
            return FaithfulnessMetric(model=model, threshold=threshold, async_mode=False, include_reason=s.include_reasons)
        if metric == "answer_relevancy":
            return AnswerRelevancyMetric(model=model, threshold=threshold, async_mode=False, include_reason=s.include_reasons)
        params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT]
        if metric == "answer_correctness":
            return GEval(name="Answer Correctness", evaluation_steps=CORRECTNESS_STEPS, evaluation_params=params,
                         model=model, threshold=threshold, async_mode=False)
        return GEval(name="Answer Completeness", evaluation_steps=COMPLETENESS_STEPS, evaluation_params=params,
                     model=model, threshold=threshold, async_mode=False)

    def _measure(self, metric: str, sample: EvaluationSample) -> MetricOutcome:
        from deepeval.test_case import LLMTestCase

        expected = sample.reference_answer
        if metric == "answer_completeness" and sample.required_facts:
            expected = f"{expected or ''}\nRequired details:\n" + "\n".join(f"- {f}" for f in sample.required_facts)
        case = LLMTestCase(
            input=sample.question,
            actual_output=sample.generated_answer,
            expected_output=expected,
            retrieval_context=sample.retrieved_contexts or None,
        )
        scorer = self._build(metric)
        scorer.measure(case)
        value = check_score(scorer.score, "DeepEval")

        details: dict = {"judge_model": self.settings.judge_model}
        if metric in ("faithfulness", "answer_relevancy"):
            verdicts = [v for v in (getattr(scorer, "verdicts", None) or []) if v is not None]
            passed = sum(1 for v in verdicts if _verdict(v) != "no")
            unit = "claims" if metric == "faithfulness" else "statements"
            good = "supported" if metric == "faithfulness" else "relevant"
            details[unit] = list(getattr(scorer, "claims" if metric == "faithfulness" else "statements", None) or [])
            details["verdicts"] = [_verdict(v) for v in verdicts]
            explanation = (
                f"{passed} {good} {unit} / {len(verdicts)} total {unit} = {value:.3f}"
                if verdicts else f"The judge found no {unit}; DeepEval scores this as {value:.3f}"
            )
        else:
            explanation = f"G-Eval: the judge scored the answer against the {metric.split('_')[1]} steps, scaled to 0-1 = {value:.3f}"
            details["evaluation_steps"] = CORRECTNESS_STEPS if metric == "answer_correctness" else COMPLETENESS_STEPS
        if getattr(scorer, "reason", None):
            details["reason"] = scorer.reason
        return MetricOutcome(score=value, explanation=explanation, details=details,
                             framework_version=package_version("deepeval"))

    async def score(self, metric: str, sample: EvaluationSample) -> MetricOutcome:
        return await asyncio.to_thread(self._measure, metric, sample)
