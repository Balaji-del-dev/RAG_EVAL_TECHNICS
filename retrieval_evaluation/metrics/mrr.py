"""Reciprocal rank and Mean Reciprocal Rank (MRR@K).

For one query:

    RR = 1 / (rank of the first relevant chunk)     (0 if none in the top K)

Over a set of queries:

    MRR = mean of RR over all queries

MRR only looks at the first relevant chunk: it tells you how far down the user
has to read before something useful appears.
"""

from collections.abc import Collection, Iterable, Sequence

from .base import top_k_unique


def first_relevant_rank(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> int | None:
    """1-based rank of the first relevant chunk in the top K, or None."""
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(top_k_unique(retrieved_ids, k), start=1):
        if chunk_id in relevant:
            return rank
    return None


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    rank = first_relevant_rank(retrieved_ids, relevant_ids, k)
    return 1.0 / rank if rank else 0.0


def mean_reciprocal_rank(reciprocal_ranks: Iterable[float]) -> float:
    values = list(reciprocal_ranks)
    if not values:
        raise ValueError("MRR needs at least one query.")
    return sum(values) / len(values)
