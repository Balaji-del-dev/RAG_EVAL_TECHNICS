from django.test import SimpleTestCase

from retrieval_evaluation.metrics import aggregate, compute_query_metrics, explain_query_metrics
from retrieval_evaluation.services.ground_truth_matcher import match
from retrieval_evaluation.services.retrieval_adapter import RetrievedItem

from .helpers import WORKED_GRADES, WORKED_RETRIEVED


class MetricsEngineTests(SimpleTestCase):
    def test_all_six_for_worked_example(self):
        s = compute_query_metrics(WORKED_RETRIEVED, WORKED_GRADES, 5)
        self.assertAlmostEqual(s.precision, 0.4)
        self.assertAlmostEqual(s.recall, 2 / 3)
        self.assertAlmostEqual(s.f1, 0.5)
        self.assertEqual(s.hit_rate, 1.0)
        self.assertAlmostEqual(s.reciprocal_rank, 0.5)
        self.assertAlmostEqual(s.ndcg, 0.6077574819, places=9)
        self.assertEqual(s.first_relevant_rank, 2)

    def test_grade_zero_label_is_not_relevant(self):
        s = compute_query_metrics(["C9", "C1"], {"C1": 2, "C9": 0}, 2)
        self.assertAlmostEqual(s.precision, 0.5)
        self.assertEqual(s.first_relevant_rank, 2)

    def test_query_without_relevant_chunk_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_query_metrics(["C1"], {"C1": 0}, 5)

    def test_aggregate_is_macro_average(self):
        a = compute_query_metrics(["C1"], {"C1": 1}, 1)            # all 1.0
        b = compute_query_metrics(["X", "C1"], {"C1": 1}, 2)       # P .5, RR .5
        avg = aggregate([a, b])
        self.assertAlmostEqual(avg["precision"], 0.75)
        self.assertAlmostEqual(avg["mrr"], 0.75)
        self.assertEqual(aggregate([])["ndcg"], None)

    def test_explanation_text(self):
        e = explain_query_metrics(WORKED_RETRIEVED, WORKED_GRADES, 5)
        self.assertEqual(e["steps"]["precision"], "2 relevant in top 5 / 5 = 0.400")
        self.assertEqual(e["steps"]["mrr"], "first relevant chunk at rank 2: 1 / 2 = 0.500")
        self.assertEqual(e["steps"]["ndcg"], "DCG 5.709 / ideal DCG 9.393 = 0.608")
        self.assertEqual([t["chunk_id"] for t in e["ideal_terms"]], ["C101", "C102", "C103"])


class GroundTruthMatcherTests(SimpleTestCase):
    def items(self, ids):
        return [RetrievedItem(cid, None) for cid in ids]

    def test_worked_example_comparison(self):
        m = match(self.items(WORKED_RETRIEVED), WORKED_GRADES, 5)
        self.assertEqual(m.relevant_found, ["C101", "C102"])
        self.assertEqual(m.irrelevant_found, ["C104", "C105", "C106"])
        self.assertEqual(m.missed, ["C103"])
        self.assertEqual([r.grade for r in m.ranked], [0, 3, 0, 2, 0])
        self.assertEqual(m.warnings, [])

    def test_ranks_beyond_k_are_kept_but_not_scored(self):
        m = match(self.items(WORKED_RETRIEVED + ["C103"]), WORKED_GRADES, 5)
        self.assertFalse(m.ranked[-1].within_k)
        self.assertEqual(m.missed, ["C103"])

    def test_duplicates_and_short_lists_warn(self):
        m = match(self.items(["C101", "C101"]), WORKED_GRADES, 5)
        self.assertEqual(m.ranked_ids, ["C101"])
        self.assertEqual(len(m.warnings), 2)
