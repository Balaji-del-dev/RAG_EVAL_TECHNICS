"""Custom LLM-as-a-judge for 14. Answer Completeness.

Completeness = Required Details Provided / Total Required Details

The judge returns JSON that must match a schema (Ollama structured output), one
entry per required detail. The details come from `required_facts` when the sample
has them; otherwise the judge lists them from the reference answer first.
"""

from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..metrics.base import MetricOutcome, package_version
from ..schemas import EvaluationSample
from .base import JudgeError

SYSTEM_PROMPT = (
    "You are a strict evaluator of answers. Reply only with JSON that matches the given schema. "
    "A detail counts as provided only if the answer states it with the same meaning; "
    "a different number, name or condition is not provided."
)


class DetailCheck(BaseModel):
    detail: str
    provided: bool


class CompletenessVerdict(BaseModel):
    details: list[DetailCheck]


def build_prompt(sample: EvaluationSample) -> str:
    if sample.required_facts:
        listed = "\n".join(f"{i}. {fact}" for i, fact in enumerate(sample.required_facts, start=1))
        return (
            f"Question: {sample.question}\n\nAnswer to check: {sample.generated_answer}\n\n"
            f"Required details:\n{listed}\n\n"
            "For each required detail, in the same order, copy it into 'detail' and set 'provided' "
            "to true if the answer states it, otherwise false."
        )
    return (
        f"Question: {sample.question}\n\nReference answer: {sample.reference_answer}\n\n"
        f"Answer to check: {sample.generated_answer}\n\n"
        "First list every separate detail in the reference answer that is needed to answer the question "
        "(one fact per detail; each number is its own detail). Then, for each detail, set 'provided' "
        "to true if the answer to check states it, otherwise false."
    )


def score_verdict(verdict: CompletenessVerdict, sample: EvaluationSample) -> MetricOutcome:
    details = verdict.details
    if sample.required_facts:
        if len(details) != len(sample.required_facts):
            raise JudgeError(
                f"The judge checked {len(details)} details, but the sample has {len(sample.required_facts)} required facts."
            )
        details = [DetailCheck(detail=fact, provided=d.provided) for fact, d in zip(sample.required_facts, details)]
    if not details:
        raise JudgeError("The judge found no required details in the reference answer.")
    provided = sum(1 for d in details if d.provided)
    value = provided / len(details)
    return MetricOutcome(
        score=value,
        explanation=f"{provided} required details provided / {len(details)} total required details = {value:.3f}",
        details={"required_details": [d.model_dump() for d in details],
                 "source": "required_facts" if sample.required_facts else "reference_answer"},
        framework_version=f"custom (ollama {package_version('ollama')})",
    )


class CustomJudge:
    name = "custom"
    supports = frozenset({"answer_completeness"})

    def __init__(self, settings: Settings):
        self.settings = settings

    async def score(self, metric: str, sample: EvaluationSample) -> MetricOutcome:
        import httpx
        import ollama

        s = self.settings
        client = ollama.AsyncClient(host=s.ollama_base_url, timeout=s.judge_timeout_seconds)
        try:
            response = await client.chat(
                model=s.judge_model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_prompt(sample)}],
                format=CompletenessVerdict.model_json_schema(),
                options={"temperature": s.judge_temperature, "num_predict": s.judge_max_tokens},
            )
        except (ollama.ResponseError, httpx.HTTPError, ConnectionError) as exc:
            raise JudgeError(f"Ollama call failed: {exc}") from exc
        try:
            verdict = CompletenessVerdict.model_validate_json(response.message.content)
        except ValidationError as exc:
            raise JudgeError("The judge's reply did not match the JSON schema.") from exc
        outcome = score_verdict(verdict, sample)
        outcome.details["judge_model"] = s.judge_model
        return outcome
