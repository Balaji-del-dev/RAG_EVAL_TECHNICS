"""LLM-as-a-judge adapters for the four LLM-judged metrics (11-14).

    ragas     RAGAS 0.4 metrics: Faithfulness, AnswerRelevancy, AnswerCorrectness
    deepeval  DeepEval: FaithfulnessMetric, AnswerRelevancyMetric, GEval (correctness, completeness)
    custom    Our own LLM-as-a-judge for Answer Completeness (required details provided / total)

All three use the same local Ollama model. Frameworks are imported on first use,
so importing this package stays fast.
"""

from .base import Judge, JudgeError, build_judges

__all__ = ["Judge", "JudgeError", "build_judges"]
