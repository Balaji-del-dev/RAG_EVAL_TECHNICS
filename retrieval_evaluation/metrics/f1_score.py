"""F1@K: the harmonic mean of Precision@K and Recall@K.

    F1 = 2 * P * R / (P + R)        (0 when P + R = 0)

The harmonic mean is pulled towards the smaller value, so a retriever cannot
get a good F1 by maximising only one of precision or recall.
"""

from collections.abc import Collection, Sequence

from .precision import precision_at_k
from .recall import recall_at_k


def f1_from_precision_recall(precision: float, recall: float) -> float:
    if precision < 0 or recall < 0:
        raise ValueError("Precision and recall must be non-negative.")
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def f1_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    return f1_from_precision_recall(
        precision_at_k(retrieved_ids, relevant_ids, k),
        recall_at_k(retrieved_ids, relevant_ids, k),
    )
