"""Web UI pages: evaluate, compare, formulas."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

FAST = ["exact_match", "bleu", "rouge", "semantic_similarity"]


@pytest.fixture
def client(make_evaluator):
    app.state.evaluator = make_evaluator()
    with TestClient(app) as c:
        yield c
    del app.state.evaluator


def form(**overrides):
    data = {
        "question": "How many casual leaves and sick leaves do employees receive per year?",
        "contexts": "Employees receive 12 casual leaves per year.\n\nEmployees receive 12 sick leaves per year.",
        "reference": "Employees receive 12 casual leaves and 12 sick leaves per year.",
        "facts": "Casual leave is 12 days per year\nSick leave is 12 days per year",
        "answer": "Employees receive 12 casual leaves.",
        "metrics": FAST,
    }
    data.update(overrides)
    return data


def test_evaluate_page_with_examples(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Evaluate an answer" in page.text
    for label in ("Correct &amp; complete", "Paraphrase", "Incomplete", "Hallucinated", "Irrelevant"):
        assert label in page.text
    page = client.get("/?example=hr-04-hallucinated")
    assert "Employees receive 20 casual leaves and 15 sick leaves per year." in page.text


def test_run_from_the_form(client):
    page = client.post("/", data=form())
    assert page.status_code == 200
    assert "How each score was calculated" in page.text
    assert "5 overlapping reference unigrams / 11 total reference unigrams = 0.455" in page.text
    assert 'class="card verdict verdict--' in page.text


def test_llm_metric_with_chosen_framework(client, judges):
    page = client.post("/", data=form(metrics=["answer_completeness"], engine_answer_completeness="custom"))
    assert page.status_code == 200
    assert [metric for _, metric in judges["custom"].calls] == ["answer_completeness"]
    assert "Custom LLM judge" in page.text


@pytest.mark.parametrize("overrides, message", [
    ({"metrics": []}, "Tick at least one metric."),
    ({"question": "  "}, "Please fill in: Question"),
])
def test_form_errors(client, overrides, message):
    page = client.post("/", data=form(**overrides))
    assert page.status_code == 400
    assert message in page.text


def test_compare_page(client):
    assert "Compare the 5 answers" in client.get("/compare").text
    page = client.post("/compare", data={"metrics": ["exact_match", "rouge"]})
    assert page.status_code == 200
    assert "Scores side by side" in page.text
    assert page.text.count('class="score-cell"') == 10  # 5 answers × 2 metrics


def test_formulas_page(client):
    page = client.get("/formulas")
    assert "Don&#39;t confuse these metrics" in page.text or "Don't confuse these metrics" in page.text
    assert "Supported Claims / Total Claims" in page.text
    assert "3. GENERATION EVALUATION" in page.text


def test_static_files(client):
    assert client.get("/static/css/generation.css").status_code == 200
    assert client.get("/static/js/app.js").status_code == 200
