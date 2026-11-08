"""Recall@K: how much of the relevant knowledge the retriever found.

    Recall@K = (relevant chunks in the top K) / (total relevant chunks)

Recall is undefined for a query with no relevant chunks, so that raises
ValueError instead of silently returning 0 or 1.

Binary relevance: a chunk with ground-truth grade >= 1 is relevant.
"""

from collections.abc import Collection, Sequence

from .base import count_relevant, top_k_unique


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Recall is undefined: the query has no relevant chunks.")
    top = top_k_unique(retrieved_ids, k)
    return count_relevant(top, relevant) / len(relevant)
