"""Helpers shared by the metric functions."""

from collections.abc import Iterable, Sequence


def validate_k(k: int) -> int:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return k


def top_k_unique(retrieved_ids: Sequence[str], k: int) -> list[str]:
    """The first k distinct chunk IDs, in rank order.

    A retriever that returns the same chunk twice must not get credit twice,
    so a repeated ID is dropped and the later chunks move up one rank.
    """
    validate_k(k)
    seen: set[str] = set()
    top: list[str] = []
    for chunk_id in retrieved_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        top.append(chunk_id)
        if len(top) == k:
            break
    return top


def count_relevant(top_ids: Iterable[str], relevant_ids: Iterable[str]) -> int:
    relevant = set(relevant_ids)
    return sum(1 for chunk_id in top_ids if chunk_id in relevant)
