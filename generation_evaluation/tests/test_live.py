"""End to end with the real frameworks and the local Ollama judge. Slow on a CPU.

Run: RUN_LIVE=1 uv run pytest tests/test_live.py -s
"""

import asyncio
import os

import pytest

from generation_eval import GenerationEvaluator, Settings

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="needs Ollama and loads the judge model; set RUN_LIVE=1"),
]

ENGINES = [
    ("ragas", {}),
    ("deepeval", {"faithfulness": "deepeval", "answer_relevancy": "deepeval", "answer_correctness": "deepeval"}),
]


@pytest.mark.parametrize("label, engines", ENGINES, ids=[e[0] for e in ENGINES])
def test_correct_answer_scores_high(hr_samples, label, engines):
    evaluator = GenerationEvaluator(Settings(tracking_enabled=False))
    sample = hr_samples["hr-01-correct-complete"]
    metrics = ["faithfulness", "answer_relevancy", "answer_correctness", "answer_completeness"]
    report = asyncio.run(evaluator.evaluate(sample, metrics=metrics, engines=engines, track=False))
    for s in report.scores:
        print(f"{label:9} {s.metric:20} {s.framework:9} {s.status:7} {s.score} {s.explanation or s.error} ({s.seconds}s)")
    scored = [s for s in report.scores if s.status == "scored"]
    assert scored, "no metric could be scored"
    assert all(s.score >= 0.5 for s in scored if s.metric == "faithfulness")
