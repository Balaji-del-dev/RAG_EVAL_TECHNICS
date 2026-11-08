"""Experiment comparison and regression testing between two evaluation runs.

Averages are recomputed over the queries both runs evaluated, so a query that
errored in one run (or exists in only one dataset version) cannot skew the
comparison. A metric whose average drops by more than the tolerance is a
regression.
"""

from dataclasses import dataclass, field

from django.conf import settings

from ..metrics import METRICS

IMPROVED, REGRESSED, UNCHANGED = "improved", "regressed", "unchanged"
# Per-query verdicts use NDCG: it is graded and rank-aware, so it moves whenever
# relevant chunks are gained, lost or reordered.
QUERY_VERDICT_METRIC = "ndcg"


@dataclass
class MetricDelta:
    key: str
    label: str
    display: str
    baseline: float
    candidate: float
    verdict: str

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline


@dataclass
class QueryDelta:
    query_id: str
    query_text: str
    baseline: dict[str, float]
    candidate: dict[str, float]
    verdict: str
    baseline_result_id: int
    candidate_result_id: int

    @property
    def deltas(self) -> dict[str, float]:
        return {key: self.candidate[key] - self.baseline[key] for key in self.baseline}

    @property
    def lost_hit(self) -> bool:
        return self.baseline["hit_rate"] == 1 and self.candidate["hit_rate"] == 0

    @property
    def gained_hit(self) -> bool:
        return self.baseline["hit_rate"] == 0 and self.candidate["hit_rate"] == 1


@dataclass
class Comparison:
    baseline: object
    candidate: object
    tolerance: float
    metrics: list[MetricDelta] = field(default_factory=list)
    queries: list[QueryDelta] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    only_in_baseline: list[str] = field(default_factory=list)
    only_in_candidate: list[str] = field(default_factory=list)

    @property
    def common_count(self) -> int:
        return len(self.queries)

    @property
    def regressed_metrics(self) -> list[MetricDelta]:
        return [m for m in self.metrics if m.verdict == REGRESSED]

    @property
    def improved_metrics(self) -> list[MetricDelta]:
        return [m for m in self.metrics if m.verdict == IMPROVED]

    @property
    def is_regression(self) -> bool:
        return bool(self.regressed_metrics)

    @property
    def verdict(self) -> str:
        if not self.metrics:
            return "not comparable"
        if self.regressed_metrics and self.improved_metrics:
            return "mixed"
        if self.regressed_metrics:
            return "regression"
        if self.improved_metrics:
            return "improvement"
        return "no change"

    def query_counts(self) -> dict[str, int]:
        counts = {IMPROVED: 0, REGRESSED: 0, UNCHANGED: 0}
        for q in self.queries:
            counts[q.verdict] += 1
        return counts


def _verdict(delta: float, tolerance: float) -> str:
    if delta > tolerance + 1e-12:
        return IMPROVED
    if delta < -tolerance - 1e-12:
        return REGRESSED
    return UNCHANGED


def compare_runs(baseline, candidate, tolerance: float | None = None) -> Comparison:
    if tolerance is None:
        tolerance = settings.RETRIEVAL_EVAL["REGRESSION_TOLERANCE"]
    comparison = Comparison(baseline=baseline, candidate=candidate, tolerance=tolerance)

    if baseline.pk == candidate.pk:
        comparison.warnings.append("Both sides are the same run.")
    if baseline.dataset_id != candidate.dataset_id:
        comparison.warnings.append(
            "The runs use different datasets; only queries with the same query ID are compared, "
            "and their ground truth may differ."
        )
    if baseline.top_k != candidate.top_k:
        comparison.warnings.append(
            f"The runs use different K ({baseline.top_k} vs {candidate.top_k}), so @K metrics are not like for like."
        )

    base_rows = {r.query_external_id: r for r in baseline.results.filter(error="")}
    cand_rows = {r.query_external_id: r for r in candidate.results.filter(error="")}
    common = [qid for qid in base_rows if qid in cand_rows]
    comparison.only_in_baseline = sorted(set(base_rows) - set(cand_rows))
    comparison.only_in_candidate = sorted(set(cand_rows) - set(base_rows))
    if comparison.only_in_baseline or comparison.only_in_candidate:
        comparison.warnings.append(
            f"{len(common)} queries were evaluated in both runs; "
            f"{len(comparison.only_in_baseline) + len(comparison.only_in_candidate)} appear in only one "
            "and are left out of this comparison."
        )
    if not common:
        comparison.warnings.append("The runs have no evaluated query in common.")
        return comparison

    for qid in common:
        b, c = base_rows[qid].scores(), cand_rows[qid].scores()
        comparison.queries.append(
            QueryDelta(
                query_id=qid,
                query_text=cand_rows[qid].query_text,
                baseline=b,
                candidate=c,
                verdict=_verdict(c[QUERY_VERDICT_METRIC] - b[QUERY_VERDICT_METRIC], tolerance),
                baseline_result_id=base_rows[qid].pk,
                candidate_result_id=cand_rows[qid].pk,
            )
        )

    n = len(common)
    for metric in METRICS:
        base_avg = sum(q.baseline[metric.key] for q in comparison.queries) / n
        cand_avg = sum(q.candidate[metric.key] for q in comparison.queries) / n
        comparison.metrics.append(
            MetricDelta(
                key=metric.key,
                label=metric.label(candidate.top_k),
                display=metric.display,
                baseline=base_avg,
                candidate=cand_avg,
                verdict=_verdict(cand_avg - base_avg, tolerance),
            )
        )
    # Biggest NDCG losses first, biggest gains last; ties keep the dataset order.
    comparison.queries.sort(key=lambda q: (round(q.deltas[QUERY_VERDICT_METRIC], 9), q.candidate_result_id))
    return comparison
