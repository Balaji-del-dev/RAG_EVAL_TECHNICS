"""Test doubles: fake judges and a fake embedder, so tests run without Ollama or torch."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from generation_eval import EvaluationSample, GenerationEvaluator, Settings
from generation_eval.metrics import MetricOutcome, SentenceTransformerSimilarity

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_FILE = ROOT / "data" / "hr_leave_samples.json"


class FakeJudge:
    """Returns preset scores: scores[(sample_id, metric)] or scores[metric], default 1.0."""

    def __init__(self, name: str, supports: set[str], scores: dict | None = None):
        self.name, self.supports, self.scores = name, frozenset(supports), scores or {}
        self.calls: list[tuple[str, str]] = []

    async def score(self, metric, sample):
        self.calls.append((sample.sample_id, metric))
        value = self.scores.get((sample.sample_id, metric), self.scores.get(metric, 1.0))
        if isinstance(value, Exception):
            raise value
        return MetricOutcome(score=value, explanation=f"{self.name} says {value}", framework_version="fake")


def bag_of_words(texts):
    """A tiny deterministic 'embedding': word counts over the words in both texts."""
    tokens = [re.findall(r"[a-z0-9]+", t.lower()) for t in texts]
    vocab = sorted({w for ts in tokens for w in ts})
    return np.array([[ts.count(w) for w in vocab] for ts in tokens], dtype=float)


class FakeTracker:
    def __init__(self, enabled=False, fail=False):
        self.enabled, self.fail, self.recorded = enabled, fail, []

    def record(self, sample, report):
        if self.fail:
            raise ConnectionError("Langfuse is down")
        self.recorded.append(report)
        return f"http://localhost:3000/trace/{sample.sample_id}"

    def check(self):
        return self.enabled


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, tracking_enabled=False, ollama_base_url="http://127.0.0.1:9")


@pytest.fixture
def judges():
    return {
        "ragas": FakeJudge("ragas", {"faithfulness", "answer_relevancy", "answer_correctness"}),
        "deepeval": FakeJudge("deepeval", {"faithfulness", "answer_relevancy", "answer_correctness", "answer_completeness"}),
        "custom": FakeJudge("custom", {"answer_completeness"}),
    }


@pytest.fixture
def make_evaluator(settings, judges):
    def build(tracker=None, **judge_scores):
        for name, scores in judge_scores.items():
            judges[name].scores = scores
        return GenerationEvaluator(
            settings,
            judges=judges,
            similarity=SentenceTransformerSimilarity("fake-bag-of-words", encoder=bag_of_words),
            tracker=tracker or FakeTracker(),
        )

    return build


@pytest.fixture
def hr_samples() -> dict[str, EvaluationSample]:
    rows = json.loads(SAMPLES_FILE.read_text())["samples"]
    return {row["sample_id"]: EvaluationSample(**row) for row in rows}
