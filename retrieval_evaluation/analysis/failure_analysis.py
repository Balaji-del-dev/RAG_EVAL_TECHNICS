"""Query-level failure analysis: which queries fail, and which metric shows why.

    Issue                        Detected by
    Irrelevant chunks            Precision
    Missing chunks               Recall
    No useful result             Hit Rate
    Relevant result ranked low   MRR
    Poor relevance ordering      NDCG

Thresholds live in settings.RETRIEVAL_EVAL["FAILURE_THRESHOLDS"]. Precision and
recall are judged against the best score the query can reach at K (its
ceiling): with 3 relevant chunks and K = 10, Precision@10 can never exceed 0.3,
and that is not the retriever's fault.
"""

from collections import Counter
from dataclasses import dataclass

from django.conf import settings


@dataclass(frozen=True)
class Issue:
    code: str
    label: str
    metric: str
    explanation: str


ISSUES: dict[str, Issue] = {
    issue.code: issue
    for issue in (
        Issue("no_useful_result", "No useful result", "Hit Rate",
              "None of the top K chunks is relevant."),
        Issue("irrelevant_chunks", "Irrelevant chunks", "Precision",
              "Too many of the top K chunks are not relevant."),
        Issue("missing_chunks", "Missing chunks", "Recall",
              "Relevant chunks that would fit in the top K were not retrieved."),
        Issue("ranked_low", "Relevant result ranked low", "MRR",
              "The first relevant chunk appears too far down the list."),
        Issue("poor_ordering", "Poor relevance ordering", "NDCG",
              "The most relevant chunks are not near the top."),
    )
}

STATUS_PASS, STATUS_WEAK, STATUS_FAILED, STATUS_ERROR = "pass", "weak", "failed", "error"
STATUS_LABELS = {
    STATUS_PASS: "Pass",
    STATUS_WEAK: "Needs attention",
    STATUS_FAILED: "Failed",
    STATUS_ERROR: "Error",
}


def thresholds() -> dict[str, float]:
    return dict(settings.RETRIEVAL_EVAL["FAILURE_THRESHOLDS"])


def precision_ceiling(total_relevant: int, k: int) -> float:
    return min(total_relevant, k) / k


def recall_ceiling(total_relevant: int, k: int) -> float:
    return min(total_relevant, k) / total_relevant if total_relevant else 0.0


def detect_issues(scores: dict[str, float], total_relevant: int, k: int, limits: dict | None = None) -> list[str]:
    """Issue codes for one query. scores uses the METRICS keys ('mrr' = reciprocal rank)."""
    limits = limits or thresholds()
    if scores["hit_rate"] == 0:
        # Nothing relevant came back: every other issue follows from this one.
        return ["no_useful_result"]
    issues = []
    if scores["precision"] < limits["precision_vs_ceiling"] * precision_ceiling(total_relevant, k) - 1e-9:
        issues.append("irrelevant_chunks")
    if scores["recall"] < limits["recall_vs_ceiling"] * recall_ceiling(total_relevant, k) - 1e-9:
        issues.append("missing_chunks")
    if scores["mrr"] < limits["mrr"] - 1e-9:
        issues.append("ranked_low")
    if scores["ndcg"] < limits["ndcg"] - 1e-9:
        issues.append("poor_ordering")
    return issues


def query_status(result) -> str:
    """Status of a MetricResult: error, failed (no relevant chunk), weak (other issues) or pass."""
    if result.error:
        return STATUS_ERROR
    if "no_useful_result" in (result.issues or []):
        return STATUS_FAILED
    return STATUS_WEAK if result.issues else STATUS_PASS


def summarize_run(run) -> dict:
    """Issue counts, status counts and the queries that need attention."""
    results = list(run.results.all())
    issue_counts = Counter(code for r in results for code in (r.issues or []))
    statuses = Counter(query_status(r) for r in results)
    evaluated = sum(1 for r in results if not r.error)
    rows = [
        {
            "issue": issue,
            "count": issue_counts.get(code, 0),
            "share": (issue_counts.get(code, 0) / evaluated) if evaluated else 0.0,
        }
        for code, issue in ISSUES.items()
    ]
    severity = {STATUS_ERROR: 0, STATUS_FAILED: 1, STATUS_WEAK: 2, STATUS_PASS: 3}
    attention = sorted(
        (r for r in results if query_status(r) != STATUS_PASS),
        key=lambda r: (severity[query_status(r)], r.ndcg if r.ndcg is not None else -1, r.pk),
    )
    return {
        "issue_rows": rows,
        "status_counts": {code: statuses.get(code, 0) for code in STATUS_LABELS},
        "attention": attention,
        "thresholds": thresholds(),
    }
