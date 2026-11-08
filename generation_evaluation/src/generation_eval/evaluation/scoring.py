"""Pass / fail rules and the overall score.

* A metric passes when its score >= its threshold.
* Critical and high metrics (faithfulness, relevancy, correctness, completeness,
  semantic similarity) decide pass/fail. Exact Match, BLEU and ROUGE are reported
  but do not fail an answer: they are supporting text-comparison scores.
* Status:
    fail        a deciding metric scored below its threshold
    incomplete  no deciding metric failed, but one could not be scored (error or missing input)
    pass        every deciding metric scored and passed
  If only low-priority metrics were requested, those decide instead.
* Overall score = weighted mean of the scored metrics: critical 3, high 2, low 1.
"""

from ..catalog import PRIORITY_WEIGHT
from ..schemas import MetricScore


def deciding(scores: list[MetricScore]) -> list[MetricScore]:
    gating = [s for s in scores if s.gating]
    return gating or scores


def decide_status(scores: list[MetricScore]) -> tuple[str, list[str]]:
    decisive = deciding(scores)
    failed = [s.metric for s in decisive if s.status == "scored" and not s.passed]
    if failed:
        return "fail", failed
    if any(s.status != "scored" for s in decisive) or not decisive:
        return "incomplete", []
    return "pass", []


def overall_score(scores: list[MetricScore]) -> float | None:
    scored = [s for s in scores if s.status == "scored" and s.score is not None]
    if not scored:
        return None
    total_weight = sum(PRIORITY_WEIGHT[s.priority] for s in scored)
    return sum(PRIORITY_WEIGHT[s.priority] * s.score for s in scored) / total_weight
