"""The six retrieval metrics, implemented in plain Python (no evaluation frameworks)."""

from .f1_score import f1_at_k, f1_from_precision_recall
from .hit_rate import hit_rate_at_k
from .metrics_engine import (
    METRIC_KEYS,
    METRICS,
    METRICS_BY_KEY,
    QueryScores,
    aggregate,
    compute_query_metrics,
    explain_query_metrics,
)
from .mrr import first_relevant_rank, mean_reciprocal_rank, reciprocal_rank
from .ndcg import dcg_at_k, idcg_at_k, ndcg_at_k
from .precision import precision_at_k
from .recall import recall_at_k

__all__ = [
    "METRICS", "METRIC_KEYS", "METRICS_BY_KEY", "QueryScores",
    "aggregate", "compute_query_metrics", "explain_query_metrics",
    "precision_at_k", "recall_at_k", "f1_at_k", "f1_from_precision_recall",
    "hit_rate_at_k", "reciprocal_rank", "mean_reciprocal_rank", "first_relevant_rank",
    "dcg_at_k", "idcg_at_k", "ndcg_at_k",
]
