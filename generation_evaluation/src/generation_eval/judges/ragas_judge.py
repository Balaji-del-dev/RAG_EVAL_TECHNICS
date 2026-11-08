"""RAGAS 0.4 adapter, with a local Ollama model through its OpenAI-compatible API.

    faithfulness        ragas Faithfulness:      claims supported by the context / all claims
    answer_relevancy    ragas AnswerRelevancy:   mean cosine(question, questions generated from the answer)
    answer_correctness  ragas AnswerCorrectness: 0.75 × factual F1 + 0.25 × semantic similarity
"""

from ..config import Settings
from ..metrics.base import MetricOutcome, package_version
from ..schemas import EvaluationSample
from .base import check_score

HOW_RAGAS_SCORES = {
    "faithfulness": "RAGAS split the answer into claims and asked the judge which ones the context supports: supported claims / total claims",
    "answer_relevancy": "RAGAS generated questions from the answer and compared them with the real question: mean cosine similarity",
    "answer_correctness": "RAGAS compared the answer's facts with the reference: 0.75 × factual F1 + 0.25 × semantic similarity",
}


class RagasJudge:
    name = "ragas"
    supports = frozenset(HOW_RAGAS_SCORES)

    def __init__(self, settings: Settings):
        self.settings = settings
        self._metrics: dict | None = None

    def _build_metrics(self) -> dict:
        from openai import AsyncOpenAI
        from ragas.embeddings import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerCorrectness, AnswerRelevancy, Faithfulness

        s = self.settings
        client = AsyncOpenAI(base_url=f"{s.ollama_base_url.rstrip('/')}/v1", api_key="ollama", timeout=s.judge_timeout_seconds)
        llm = llm_factory(s.judge_model, provider="openai", client=client, max_tokens=s.judge_max_tokens)
        embeddings = embedding_factory("openai", model=s.ragas_embedding_model, client=client)
        return {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=s.ragas_relevancy_strictness),
            "answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings),
        }

    async def score(self, metric: str, sample: EvaluationSample) -> MetricOutcome:
        if self._metrics is None:
            self._metrics = self._build_metrics()
        scorer = self._metrics[metric]
        if metric == "faithfulness":
            result = await scorer.ascore(
                user_input=sample.question, response=sample.generated_answer, retrieved_contexts=sample.retrieved_contexts
            )
        elif metric == "answer_relevancy":
            result = await scorer.ascore(user_input=sample.question, response=sample.generated_answer)
        else:
            result = await scorer.ascore(
                user_input=sample.question, response=sample.generated_answer, reference=sample.reference_answer
            )
        value = check_score(result.value, "RAGAS")
        details = {"judge_model": self.settings.judge_model}
        if getattr(result, "reason", None):
            details["reason"] = result.reason
        if metric != "faithfulness":
            details["embedding_model"] = self.settings.ragas_embedding_model
        return MetricOutcome(
            score=value,
            explanation=f"{HOW_RAGAS_SCORES[metric]} = {value:.3f}",
            details=details,
            framework_version=package_version("ragas"),
        )
