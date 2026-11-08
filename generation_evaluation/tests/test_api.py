"""FastAPI service: /health, /metrics, /tree, /evaluate, /evaluate/batch."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(make_evaluator):
    app.state.evaluator = make_evaluator()
    with TestClient(app) as c:
        yield c
    del app.state.evaluator


SAMPLE = {
    "sample_id": "api-1",
    "question": "How many casual leaves and sick leaves do employees receive per year?",
    "generated_answer": "Employees receive 12 casual leaves and 12 sick leaves per year.",
    "retrieved_contexts": ["Employees receive 12 casual leaves per year. Employees receive 12 sick leaves per year."],
    "reference_answer": "Employees receive 12 casual leaves and 12 sick leaves per year.",
}


def test_metrics_catalog(client):
    rows = client.get("/metrics").json()
    assert [r["number"] for r in rows] == list(range(11, 19))
    faithfulness = rows[0]
    assert faithfulness["formula"] == "Supported Claims / Total Claims"
    assert faithfulness["frameworks"] == ["ragas", "deepeval"]
    assert faithfulness["needs"] == ["question", "generated_answer", "retrieved_contexts"]
    assert rows[-1]["frameworks"] == ["rouge-score"] and rows[-1]["decides_pass_fail"] is False


def test_tree(client):
    text = client.get("/tree").text
    assert text.startswith("3. GENERATION EVALUATION")
    assert "Framework: RAGAS / DeepEval" in text and "Library: SacreBLEU" in text
    assert "Backend API: FastAPI" in text and "Evaluation Tracking: Langfuse" in text


def test_evaluate(client):
    response = client.post("/evaluate", json={"sample": SAMPLE, "track": False})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pass" and body["tracking"] == "off for this request"
    assert len(body["scores"]) == 8


def test_evaluate_subset_with_engine(client):
    response = client.post("/evaluate", json={"sample": SAMPLE, "metrics": ["faithfulness", "rouge"],
                                              "engines": {"faithfulness": "deepeval"}})
    scores = {s["metric"]: s for s in response.json()["scores"]}
    assert set(scores) == {"faithfulness", "rouge"}
    assert scores["faithfulness"]["framework"] == "deepeval"
    assert scores["rouge"]["framework"] == "rouge-score"


@pytest.mark.parametrize("body, message", [
    ({"engines": {"answer_completeness": "ragas"}}, "answer_completeness can use deepeval or custom, not ragas"),
    ({"engines": {"bleu": "deepeval"}}, "bleu is not LLM-judged"),
    ({"metrics": ["perplexity"]}, "Input should be"),
])
def test_invalid_requests(client, body, message):
    response = client.post("/evaluate", json={"sample": SAMPLE, **body})
    assert response.status_code == 422
    assert message in response.text


def test_batch(client):
    second = {**SAMPLE, "sample_id": "api-2", "generated_answer": "Employees can apply for work from home."}
    body = client.post("/evaluate/batch", json={"samples": [SAMPLE, second], "metrics": ["exact_match", "rouge"]}).json()
    assert (body["total"], body["passed"], body["failed"]) == (2, 1, 1)
    assert body["metric_averages"]["exact_match"] == 0.5


def test_health_reports_unreachable_ollama(client):
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["ollama"]["reachable"] is False
    assert body["tracking"]["enabled"] is False

