import json

from django.test import TestCase, override_settings

from retrieval_evaluation.analysis.experiment_comparison import compare_runs
from retrieval_evaluation.analysis.failure_analysis import detect_issues, query_status, summarize_run
from retrieval_evaluation.services.dataset_loader import import_dataset, parse_dataset_file
from retrieval_evaluation.services.evaluation_runner import create_run, execute_run

from .helpers import WORKED_DATASET

LIMITS = {"precision_vs_ceiling": 0.5, "recall_vs_ceiling": 0.8, "mrr": 0.5, "ndcg": 0.7}


class FailureAnalysisTests(TestCase):
    def scores(self, **kw):
        base = {"precision": 1, "recall": 1, "f1": 1, "hit_rate": 1, "mrr": 1, "ndcg": 1}
        return base | kw

    def test_no_hit_is_the_only_issue(self):
        self.assertEqual(detect_issues(self.scores(hit_rate=0, precision=0, mrr=0), 3, 5, LIMITS), ["no_useful_result"])

    def test_precision_judged_against_ceiling(self):
        # 1 relevant chunk at K = 5: ceiling is 0.2, so P@5 = 0.2 is perfect.
        self.assertEqual(detect_issues(self.scores(precision=0.2), 1, 5, LIMITS), [])
        self.assertEqual(detect_issues(self.scores(precision=0.2), 5, 5, LIMITS), ["irrelevant_chunks"])

    def test_each_issue(self):
        self.assertEqual(detect_issues(self.scores(recall=0.5), 2, 5, LIMITS), ["missing_chunks"])
        self.assertEqual(detect_issues(self.scores(mrr=1 / 3), 1, 5, LIMITS), ["ranked_low"])
        self.assertEqual(detect_issues(self.scores(mrr=0.5), 1, 5, LIMITS), [])  # rank 2 is fine
        self.assertEqual(detect_issues(self.scores(ndcg=0.6), 1, 5, LIMITS), ["poor_ordering"])


@override_settings(RETRIEVAL_EVAL={
    "TOP_K_CHOICES": [5], "NDCG_GAIN": "exponential", "HTTP_TIMEOUT": 5, "MAX_UPLOAD_BYTES": 10**6,
    "FAILURE_THRESHOLDS": LIMITS, "REGRESSION_TOLERANCE": 0.01,
})
class ComparisonTests(TestCase):
    def setUp(self):
        data = dict(WORKED_DATASET, queries=WORKED_DATASET["queries"] + [
            {"query_id": "Q2", "text": "work from home", "ground_truth": [{"chunk_id": "C105", "grade": 3}]},
        ])
        self.dataset = import_dataset(parse_dataset_file("d.json", json.dumps(data).encode()))

    def make_run(self, rankings):
        results = {q: [{"chunk_id": c, "score": None} for c in ids] for q, ids in rankings.items()}
        return execute_run(create_run(self.dataset, name="r", top_k=5, retriever_type="precomputed",
                                      retriever_config={"results": results}))

    def test_improvement_and_regression(self):
        baseline = self.make_run({"Q1": ["C104", "C101", "C105", "C102", "C106"], "Q2": ["C105"]})
        better = self.make_run({"Q1": ["C101", "C102", "C103"], "Q2": ["C105"]})
        cmp = compare_runs(baseline, better)
        self.assertEqual(cmp.verdict, "improvement")
        self.assertFalse(cmp.is_regression)
        self.assertEqual(cmp.query_counts(), {"improved": 1, "regressed": 0, "unchanged": 1})

        worse = compare_runs(better, baseline)
        self.assertTrue(worse.is_regression)
        self.assertEqual(worse.queries[0].query_id, "Q1")  # biggest drop first

        lost = self.make_run({"Q1": ["C101", "C102", "C103"], "Q2": ["C106"]})
        q2 = next(q for q in compare_runs(better, lost).queries if q.query_id == "Q2")
        self.assertTrue(q2.lost_hit)

    def test_only_common_queries_are_compared(self):
        a = self.make_run({"Q1": ["C101"], "Q2": ["C105"]})
        b = self.make_run({"Q1": ["C101"]})  # Q2 errors in b
        cmp = compare_runs(a, b)
        self.assertEqual(cmp.common_count, 1)
        self.assertEqual(cmp.only_in_baseline, ["Q2"])
        self.assertEqual(cmp.verdict, "no change")

    def test_summary(self):
        run = self.make_run({"Q1": ["C104", "C101", "C105", "C102", "C106"], "Q2": ["C106"]})
        summary = summarize_run(run)
        self.assertEqual(summary["status_counts"]["failed"], 1)
        self.assertEqual(summary["status_counts"]["weak"], 1)
        self.assertEqual(query_status(summary["attention"][0]), "failed")
