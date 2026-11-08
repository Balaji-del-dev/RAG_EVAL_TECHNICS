"""The RAGAS and DeepEval metric objects build with the installed versions (no LLM call).

This catches API changes in the frameworks (renamed classes or arguments) without Ollama.
"""

from generation_eval import Settings
from generation_eval.judges.deepeval_judge import DeepEvalJudge
from generation_eval.judges.ragas_judge import RagasJudge


def test_ragas_metrics_build():
    metrics = RagasJudge(Settings(_env_file=None))._build_metrics()
    assert {k: type(v).__name__ for k, v in metrics.items()} == {
        "faithfulness": "Faithfulness", "answer_relevancy": "AnswerRelevancy", "answer_correctness": "AnswerCorrectness",
    }


def test_deepeval_metrics_build():
    judge = DeepEvalJudge(Settings(_env_file=None))
    built = {m: judge._build(m) for m in judge.supports}
    assert type(built["faithfulness"]).__name__ == "FaithfulnessMetric"
    assert type(built["answer_relevancy"]).__name__ == "AnswerRelevancyMetric"
    assert built["answer_correctness"].name.startswith("Answer Correctness")
    assert built["answer_completeness"].threshold == 0.8
