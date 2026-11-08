"""The evaluator: which metrics run, engines, thresholds, pass / fail / incomplete, overall score."""

import asyncio

import pytest

from generation_eval import EvaluationSample
from generation_eval.catalog import METRIC_KEYS
from generation_eval.evaluation import overall_score

from .conftest import FakeTracker


def run(coro):
    return asyncio.run(coro)


def by_metric(report):
    return {s.metric: s for s in report.scores}


def test_correct_complete_answer_passes(make_evaluator, hr_samples, judges):
    report = run(make_evaluator().evaluate(hr_samples["hr-01-correct-complete"]))
    scores = by_metric(report)
    assert report.status == "pass" and report.passed
    assert [s.number for s in report.scores] == list(range(11, 19))
    assert scores["exact_match"].score == 1.0
    assert scores["bleu"].score == pytest.approx(1.0)
    assert scores["semantic_similarity"].score == pytest.approx(1.0)
    # Default engines: RAGAS for 11-13, DeepEval for 14.
    assert {s.metric: s.framework for s in report.scores if s.number <= 14} == {
        "faithfulness": "ragas", "answer_relevancy": "ragas", "answer_correctness": "ragas", "answer_completeness": "deepeval",
    }
    assert ("hr-01-correct-complete", "answer_completeness") in judges["deepeval"].calls


def test_low_priority_metrics_do_not_fail_a_paraphrase(make_evaluator, hr_samples):
    # (The fake bag-of-words embedder cannot see paraphrases, so semantic similarity is left out here.)
    metrics = ["faithfulness", "answer_relevancy", "exact_match", "bleu"]
    report = run(make_evaluator().evaluate(hr_samples["hr-02-paraphrase"], metrics=metrics))
    scores = by_metric(report)
    assert scores["exact_match"].score == 0.0 and scores["exact_match"].passed is False
    assert scores["bleu"].passed is False
    assert not scores["exact_match"].gating
    assert report.status == "pass"


def test_incomplete_answer_fails_on_completeness(make_evaluator, hr_samples):
    evaluator = make_evaluator(deepeval={"answer_completeness": 0.5})
    report = run(evaluator.evaluate(hr_samples["hr-03-incomplete"]))
    assert report.status == "fail"
    assert report.failed_metrics == ["answer_completeness"]


def test_hallucinated_answer_fails_on_faithfulness_and_correctness(make_evaluator, hr_samples):
    evaluator = make_evaluator(ragas={"faithfulness": 0.0, "answer_correctness": 0.2})
    report = run(evaluator.evaluate(hr_samples["hr-04-hallucinated"],
                                    metrics=["faithfulness", "answer_relevancy", "answer_correctness"]))
    assert report.status == "fail"
    assert report.failed_metrics == ["faithfulness", "answer_correctness"]


def test_missing_reference_skips_reference_metrics(make_evaluator):
    sample = EvaluationSample(question="How many sick leaves?", generated_answer="12 sick leaves.",
                              retrieved_contexts=["Employees receive 12 sick leaves per year."])
    report = run(make_evaluator().evaluate(sample))
    scores = by_metric(report)
    assert scores["faithfulness"].status == "scored"
    assert scores["answer_relevancy"].status == "scored"
    for key in ("answer_correctness", "answer_completeness", "semantic_similarity", "exact_match", "bleu", "rouge"):
        assert scores[key].status == "skipped"
        assert scores[key].error == "Needs reference_answer."
    assert report.status == "incomplete"


def test_required_facts_are_enough_for_completeness(make_evaluator):
    sample = EvaluationSample(question="How many sick leaves?", generated_answer="12 sick leaves.",
                              required_facts=["Sick leave is 12 days per year"])
    report = run(make_evaluator().evaluate(sample, metrics=["answer_completeness"]))
    assert by_metric(report)["answer_completeness"].status == "scored"


def test_a_failing_judge_does_not_stop_the_others(make_evaluator, hr_samples):
    evaluator = make_evaluator(ragas={"faithfulness": TimeoutError("judge took too long")})
    report = run(evaluator.evaluate(hr_samples["hr-01-correct-complete"]))
    scores = by_metric(report)
    assert scores["faithfulness"].status == "error"
    assert scores["faithfulness"].error == "TimeoutError: judge took too long"
    assert scores["answer_relevancy"].status == "scored"
    assert report.status == "incomplete"


def test_engine_override(make_evaluator, hr_samples, judges):
    evaluator = make_evaluator(custom={"answer_completeness": 0.5})
    report = run(evaluator.evaluate(hr_samples["hr-03-incomplete"], metrics=["answer_completeness", "faithfulness"],
                                    engines={"answer_completeness": "custom", "faithfulness": "deepeval"}))
    scores = by_metric(report)
    assert scores["answer_completeness"].framework == "custom"
    assert scores["faithfulness"].framework == "deepeval"
    with pytest.raises(ValueError, match="answer_completeness can use deepeval or custom, not ragas"):
        run(evaluator.evaluate(hr_samples["hr-03-incomplete"], engines={"answer_completeness": "ragas"}))


def test_only_low_priority_metrics_decide_when_alone(make_evaluator, hr_samples):
    report = run(make_evaluator().evaluate(hr_samples["hr-02-paraphrase"], metrics=["exact_match", "rouge"]))
    assert report.status == "fail"
    assert report.failed_metrics == ["exact_match"]


def test_overall_score_is_weighted_by_priority(make_evaluator, hr_samples):
    evaluator = make_evaluator(ragas={"faithfulness": 0.5})
    report = run(evaluator.evaluate(hr_samples["hr-01-correct-complete"], metrics=["faithfulness", "exact_match"]))
    # critical 3 × 0.5 + low 1 × 1.0, divided by 4
    assert report.overall_score == pytest.approx((3 * 0.5 + 1 * 1.0) / 4)
    assert overall_score([]) is None


def test_batch_report(make_evaluator, hr_samples):
    evaluator = make_evaluator(deepeval={("hr-03-incomplete", "answer_completeness"): 0.5})
    metrics = [m for m in METRIC_KEYS if m != "semantic_similarity"]  # see the paraphrase test
    batch = run(evaluator.evaluate_batch(list(hr_samples.values()), metrics=metrics))
    assert (batch.total, batch.passed, batch.failed) == (5, 4, 1)
    assert batch.pass_rate == pytest.approx(0.8)
    assert batch.metric_averages["exact_match"] == pytest.approx(0.2)
    assert set(batch.metric_averages) == {s.metric for s in batch.reports[0].scores}


class TestTracking:
    def test_disabled_without_keys(self, make_evaluator, hr_samples):
        report = run(make_evaluator().evaluate(hr_samples["hr-01-correct-complete"], metrics=["exact_match"]))
        assert report.tracking == "disabled"

    def test_sent_when_enabled(self, make_evaluator, hr_samples):
        tracker = FakeTracker(enabled=True)
        report = run(make_evaluator(tracker=tracker).evaluate(hr_samples["hr-01-correct-complete"], metrics=["bleu"]))
        assert report.tracking == "sent"
        assert report.trace_url == "http://localhost:3000/trace/hr-01-correct-complete"
        assert tracker.recorded == [report]

    def test_off_for_one_request(self, make_evaluator, hr_samples):
        tracker = FakeTracker(enabled=True)
        report = run(make_evaluator(tracker=tracker).evaluate(hr_samples["hr-01-correct-complete"], metrics=["bleu"], track=False))
        assert report.tracking == "off for this request" and tracker.recorded == []

    def test_tracking_failure_keeps_the_scores(self, make_evaluator, hr_samples):
        report = run(make_evaluator(tracker=FakeTracker(enabled=True, fail=True)).evaluate(
            hr_samples["hr-01-correct-complete"], metrics=["bleu"]))
        assert report.tracking == "error: ConnectionError: Langfuse is down"
        assert report.status == "pass"
