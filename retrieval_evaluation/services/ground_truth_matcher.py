"""Matches the retriever's ranked chunk IDs against the expected relevant chunks.

The result says, for every rank, whether the chunk is relevant and with which
grade, and which relevant chunks were missed. That comparison is the input to
the six metric functions.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .retrieval_adapter import RetrievedItem


@dataclass(frozen=True)
class RankedMatch:
    rank: int
    chunk_id: str
    score: float | None
    grade: int
    is_relevant: bool
    within_k: bool


@dataclass
class MatchResult:
    k: int
    ranked: list[RankedMatch]
    expected: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def ranked_ids(self) -> list[str]:
        return [m.chunk_id for m in self.ranked]

    @property
    def top_k(self) -> list[RankedMatch]:
        return [m for m in self.ranked if m.within_k]

    @property
    def relevant_ids(self) -> set[str]:
        return {cid for cid, grade in self.expected.items() if grade >= 1}

    @property
    def relevant_found(self) -> list[str]:
        return [m.chunk_id for m in self.top_k if m.is_relevant]

    @property
    def irrelevant_found(self) -> list[str]:
        return [m.chunk_id for m in self.top_k if not m.is_relevant]

    @property
    def missed(self) -> list[str]:
        found = set(self.relevant_found)
        # Best grade first, so the most important misses are listed first.
        return [
            cid
            for cid, grade in sorted(self.expected.items(), key=lambda item: (-item[1], item[0]))
            if grade >= 1 and cid not in found
        ]


def match(retrieved: Sequence[RetrievedItem], expected: Mapping[str, int], k: int) -> MatchResult:
    """Compare a ranked retrieval with the ground truth of one query.

    Repeated chunk IDs are dropped (the first occurrence keeps its rank) and
    reported as a warning, so a retriever cannot score twice for one chunk.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    expected = {str(cid): int(grade) for cid, grade in expected.items()}
    seen: set[str] = set()
    ranked: list[RankedMatch] = []
    duplicates: list[str] = []
    for item in retrieved:
        if item.chunk_id in seen:
            duplicates.append(item.chunk_id)
            continue
        seen.add(item.chunk_id)
        rank = len(ranked) + 1
        grade = expected.get(item.chunk_id, 0)
        ranked.append(RankedMatch(rank, item.chunk_id, item.score, grade, grade >= 1, rank <= k))

    warnings = []
    if duplicates:
        warnings.append(
            "The retriever returned the same chunk more than once; repeats were ignored: "
            + ", ".join(sorted(set(duplicates)))
        )
    if len(ranked) < k:
        warnings.append(
            f"Only {len(ranked)} chunk(s) returned for K = {k}; the empty slots count as not relevant."
        )
    return MatchResult(k=k, ranked=ranked, expected=expected, warnings=warnings)
