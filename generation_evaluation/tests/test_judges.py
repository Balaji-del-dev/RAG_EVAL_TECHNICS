"""Judge helpers that need no LLM: the custom completeness judge's scoring and prompts."""

import math

import pytest

from generation_eval import EvaluationSample
from generation_eval.judges import JudgeError
from generation_eval.judges.base import check_score
from generation_eval.judges.custom_judge import CompletenessVerdict, build_prompt, score_verdict
from generation_eval.judges.deepeval_judge import _verdict

SAMPLE = EvaluationSample(
    question="How many casual leaves and sick leaves do employees receive per year?",
    generated_answer="Employees receive 12 casual leaves.",
    reference_answer="Employees receive 12 casual leaves and 12 sick leaves per year.",
    required_facts=["Casual leave is 12 days per year", "Sick leave is 12 days per year"],
)


def verdict(*provided):
    return CompletenessVerdict.model_validate(
        {"details": [{"detail": f"detail {i}", "provided": p} for i, p in enumerate(provided)]})


def test_completeness_is_provided_over_required():
    outcome = score_verdict(verdict(True, False), SAMPLE)
    assert outcome.score == 0.5
    assert outcome.explanation == "1 required details provided / 2 total required details = 0.500"
    # The sample's own wording of each fact is kept, not the judge's copy.
    assert [d["detail"] for d in outcome.details["required_details"]] == SAMPLE.required_facts


def test_judge_must_check_every_required_fact():
    with pytest.raises(JudgeError, match="checked 1 details, but the sample has 2"):
        score_verdict(verdict(True), SAMPLE)


def test_without_required_facts_the_judge_lists_them():
    sample = SAMPLE.model_copy(update={"required_facts": None})
    assert "First list every separate detail" in build_prompt(sample)
    assert score_verdict(verdict(True, True, False), sample).score == pytest.approx(2 / 3)
    with pytest.raises(JudgeError):
        score_verdict(verdict(), sample)


def test_prompt_lists_required_facts_in_order():
    prompt = build_prompt(SAMPLE)
    assert "1. Casual leave is 12 days per year\n2. Sick leave is 12 days per year" in prompt


def test_unparsable_framework_score_is_an_error():
    assert check_score(0.75, "RAGAS") == 0.75
    for bad in (None, math.nan):
        with pytest.raises(JudgeError, match="RAGAS returned no score"):
            check_score(bad, "RAGAS")


def test_deepeval_verdict_values():
    class Enumish:
        value = "YES"

    class V:
        verdict = Enumish()

    assert _verdict(V()) == "yes"
    assert _verdict(type("W", (), {"verdict": "no"})()) == "no"
