"""Precision@K: how much of what the retriever returned is relevant.

    Precision@K = (relevant chunks in the top K) / K

K is the denominator, not the number of chunks returned: if the retriever
returns fewer than K chunks, the empty slots count as not relevant. This is the
standard IR definition (trec_eval uses it too).

Binary relevance: a chunk with ground-truth grade >= 1 is relevant.
"""

from collections.abc import Collection, Sequence

from .base import count_relevant, top_k_unique


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    top = top_k_unique(retrieved_ids, k)
    return count_relevant(top, relevant_ids) / k
