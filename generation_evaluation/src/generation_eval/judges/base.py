from typing import Protocol

from ..config import Settings
from ..metrics.base import MetricOutcome
from ..schemas import EvaluationSample


class JudgeError(RuntimeError):
    """The judge could not produce a score (model unreachable, unparsable output, …)."""


class Judge(Protocol):
    name: str
    supports: frozenset[str]

    async def score(self, metric: str, sample: EvaluationSample) -> MetricOutcome: ...


def build_judges(settings: Settings) -> dict[str, Judge]:
    from .custom_judge import CustomJudge
    from .deepeval_judge import DeepEvalJudge
    from .ragas_judge import RagasJudge

    return {"ragas": RagasJudge(settings), "deepeval": DeepEvalJudge(settings), "custom": CustomJudge(settings)}


def check_score(value, framework: str) -> float:
    """Frameworks return NaN or None when the judge's output could not be parsed."""
    if value is None or value != value:  # NaN != NaN
        raise JudgeError(f"{framework} returned no score: the judge model's output could not be parsed.")
    return float(value)
