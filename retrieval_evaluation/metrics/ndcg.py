"""NDCG@K: Normalized Discounted Cumulative Gain, with graded relevance.

    gain(grade)  = 2^grade - 1          ('exponential', default)
                 = grade                ('linear')
    DCG@K        = sum over ranks i = 1..K of  gain(grade_i) / log2(i + 1)
    IDCG@K       = DCG@K of the ideal ranking (all labelled grades, best first)
    NDCG@K       = DCG@K / IDCG@K       (between 0 and 1)

Unlike the binary metrics, NDCG rewards putting the *most* relevant chunk
first: a grade-3 chunk at rank 1 is worth more than the same chunk at rank 4.
A chunk that is not in the ground truth has grade 0 and adds nothing.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .base import top_k_unique, validate_k

GAIN_FUNCTIONS = ("exponential", "linear")


def gain(grade: int, gain_type: str = "exponential") -> float:
    if grade < 0:
        raise ValueError("Relevance grades must be >= 0.")
    if gain_type == "exponential":
        return float(2**grade - 1)
    if gain_type == "linear":
        return float(grade)
    raise ValueError(f"Unknown gain type {gain_type!r}; use one of {GAIN_FUNCTIONS}.")


def discount(rank: int) -> float:
    """log2(rank + 1): rank 1 -> 1.0, rank 2 -> 1.585, rank 3 -> 2.0 ..."""
    return math.log2(rank + 1)


@dataclass(frozen=True)
class DcgTerm:
    rank: int
    chunk_id: str
    grade: int
    gain: float
    discount: float

    @property
    def contribution(self) -> float:
        return self.gain / self.discount


def dcg_terms(ranked_ids: Sequence[str], grades: Mapping[str, int], k: int, gain_type: str = "exponential") -> list[DcgTerm]:
    """The per-rank pieces of DCG@K, for showing the calculation."""
    return [
        DcgTerm(rank, chunk_id, grades.get(chunk_id, 0), gain(grades.get(chunk_id, 0), gain_type), discount(rank))
        for rank, chunk_id in enumerate(top_k_unique(ranked_ids, k), start=1)
    ]


def ideal_ranking(grades: Mapping[str, int]) -> list[str]:
    """Labelled chunks sorted best grade first (ties by chunk ID, for stable output)."""
    return [cid for cid, g in sorted(grades.items(), key=lambda item: (-item[1], item[0])) if g > 0]


def dcg_at_k(ranked_ids: Sequence[str], grades: Mapping[str, int], k: int, gain_type: str = "exponential") -> float:
    return sum(term.contribution for term in dcg_terms(ranked_ids, grades, k, gain_type))


def idcg_at_k(grades: Mapping[str, int], k: int, gain_type: str = "exponential") -> float:
    validate_k(k)
    return dcg_at_k(ideal_ranking(grades), grades, k, gain_type)


def ndcg_at_k(ranked_ids: Sequence[str], grades: Mapping[str, int], k: int, gain_type: str = "exponential") -> float:
    ideal = idcg_at_k(grades, k, gain_type)
    if ideal == 0:
        raise ValueError("NDCG is undefined: the query has no chunk with grade >= 1.")
    return dcg_at_k(ranked_ids, grades, k, gain_type) / ideal
