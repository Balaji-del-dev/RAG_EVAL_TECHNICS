"""Runs the six metric functions for a query and aggregates them over a run.

The engine only knows about chunk IDs and grades. Matching the retriever's
output against the ground truth happens before this, in
services/ground_truth_matcher.py.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

from .base import top_k_unique, validate_k
from .f1_score import f1_from_precision_recall
from .hit_rate import hit_rate_at_k
from .mrr import first_relevant_rank, mean_reciprocal_rank, reciprocal_rank
from .ndcg import dcg_terms, ideal_ranking, ndcg_at_k
from .precision import precision_at_k
from .recall import recall_at_k


@dataclass(frozen=True)
class MetricInfo:
    key: str
    name: str
    at_k: bool
    display: str  # 'percent' or 'decimal'
    formula: str
    relevance: str
    question: str

    def label(self, k: int | None = None) -> str:
        return f"{self.name}@{k}" if self.at_k and k else self.name


METRICS: tuple[MetricInfo, ...] = (
    MetricInfo("precision", "Precision", True, "percent",
               "Relevant in top K / K", "binary",
               "How much of what was retrieved is relevant?"),
    MetricInfo("recall", "Recall", True, "percent",
               "Relevant in top K / total relevant", "binary",
               "How much of the relevant knowledge was found?"),
    MetricInfo("f1", "F1", False, "percent",
               "2 x P x R / (P + R)", "binary",
               "Are precision and recall balanced?"),
    MetricInfo("hit_rate", "Hit Rate", True, "percent",
               "1 if any relevant chunk in top K, else 0", "binary",
               "Did the query get at least one useful chunk?"),
    MetricInfo("mrr", "MRR", False, "decimal",
               "1 / rank of the first relevant chunk", "rank of first relevant",
               "How high is the first useful chunk?"),
    MetricInfo("ndcg", "NDCG", True, "decimal",
               "DCG@K / ideal DCG@K", "graded (0-3)",
               "Are the most relevant chunks ranked first?"),
)
METRIC_KEYS = tuple(m.key for m in METRICS)
METRICS_BY_KEY = {m.key: m for m in METRICS}


@dataclass(frozen=True)
class QueryScores:
    precision: float
    recall: float
    f1: float
    hit_rate: float
    reciprocal_rank: float
    ndcg: float
    first_relevant_rank: int | None

    def as_metric_dict(self) -> dict[str, float]:
        """Scores keyed like METRICS ('mrr' holds this query's reciprocal rank)."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "hit_rate": self.hit_rate,
            "mrr": self.reciprocal_rank,
            "ndcg": self.ndcg,
        }


def relevant_ids_from_grades(grades: Mapping[str, int]) -> set[str]:
    """Binary relevance: grade >= 1 is relevant."""
    return {chunk_id for chunk_id, grade in grades.items() if grade >= 1}


def compute_query_metrics(
    ranked_ids: Sequence[str], grades: Mapping[str, int], k: int, gain_type: str = "exponential"
) -> QueryScores:
    """All six scores for one query.

    ranked_ids: the retriever's chunk IDs, best first.
    grades: ground truth, {chunk_id: grade 0-3}. Needs at least one grade >= 1.
    """
    validate_k(k)
    relevant = relevant_ids_from_grades(grades)
    if not relevant:
        raise ValueError("The query has no relevant chunk (grade >= 1) in its ground truth.")
    precision = precision_at_k(ranked_ids, relevant, k)
    recall = recall_at_k(ranked_ids, relevant, k)
    return QueryScores(
        precision=precision,
        recall=recall,
        f1=f1_from_precision_recall(precision, recall),
        hit_rate=hit_rate_at_k(ranked_ids, relevant, k),
        reciprocal_rank=reciprocal_rank(ranked_ids, relevant, k),
        ndcg=ndcg_at_k(ranked_ids, grades, k, gain_type),
        first_relevant_rank=first_relevant_rank(ranked_ids, relevant, k),
    )


def aggregate(scores: Iterable[QueryScores]) -> dict[str, float | None]:
    """Run-level scores: the mean of each metric over the queries.

    MRR is the mean of the reciprocal ranks; the other five are macro averages
    (every query counts the same, however many relevant chunks it has).
    """
    scores = list(scores)
    if not scores:
        return {key: None for key in METRIC_KEYS}
    n = len(scores)
    return {
        "precision": sum(s.precision for s in scores) / n,
        "recall": sum(s.recall for s in scores) / n,
        "f1": sum(s.f1 for s in scores) / n,
        "hit_rate": sum(s.hit_rate for s in scores) / n,
        "mrr": mean_reciprocal_rank(s.reciprocal_rank for s in scores),
        "ndcg": sum(s.ndcg for s in scores) / n,
    }


def explain_query_metrics(
    ranked_ids: Sequence[str], grades: Mapping[str, int], k: int, gain_type: str = "exponential"
) -> dict:
    """The working behind each score, for the query detail page."""
    scores = compute_query_metrics(ranked_ids, grades, k, gain_type)
    relevant = relevant_ids_from_grades(grades)
    top = top_k_unique(ranked_ids, k)
    found = sum(1 for cid in top if cid in relevant)
    total = len(relevant)
    p, r = scores.precision, scores.recall

    terms = dcg_terms(ranked_ids, grades, k, gain_type)
    ideal_terms = dcg_terms(ideal_ranking(grades), grades, k, gain_type)
    dcg = sum(t.contribution for t in terms)
    idcg = sum(t.contribution for t in ideal_terms)
    gain_text = "2^grade - 1" if gain_type == "exponential" else "grade"
    rank = scores.first_relevant_rank

    def fmt(value: float) -> str:
        return f"{value:.3f}"

    return {
        "scores": scores,
        "found": found,
        "total_relevant": total,
        "steps": {
            "precision": f"{found} relevant in top {k} / {k} = {fmt(p)}",
            "recall": f"{found} relevant found / {total} relevant in total = {fmt(r)}",
            "f1": (
                f"2 x {fmt(p)} x {fmt(r)} / ({fmt(p)} + {fmt(r)}) = {fmt(scores.f1)}"
                if p + r
                else "precision and recall are both 0, so F1 = 0.000"
            ),
            "hit_rate": (
                f"a relevant chunk appears at rank {rank} (within the top {k}) = 1"
                if rank
                else f"no relevant chunk in the top {k} = 0"
            ),
            "mrr": (
                f"first relevant chunk at rank {rank}: 1 / {rank} = {fmt(scores.reciprocal_rank)}"
                if rank
                else f"no relevant chunk in the top {k}: reciprocal rank = 0.000"
            ),
            "ndcg": f"DCG {fmt(dcg)} / ideal DCG {fmt(idcg)} = {fmt(scores.ndcg)}",
        },
        "gain_text": gain_text,
        "dcg_terms": [asdict(t) | {"contribution": t.contribution} for t in terms],
        "ideal_terms": [asdict(t) | {"contribution": t.contribution} for t in ideal_terms],
        "dcg": dcg,
        "idcg": idcg,
    }
