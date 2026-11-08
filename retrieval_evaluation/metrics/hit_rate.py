"""Hit Rate@K: did the retriever find at least one relevant chunk?

    Hit@K = 1 if any of the top K chunks is relevant, otherwise 0

Averaged over queries it becomes the share of queries with at least one useful
result. Binary relevance: a chunk with ground-truth grade >= 1 is relevant.
"""

from collections.abc import Collection, Sequence

from .base import top_k_unique


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    relevant = set(relevant_ids)
    return 1.0 if any(chunk_id in relevant for chunk_id in top_k_unique(retrieved_ids, k)) else 0.0
