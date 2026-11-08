import math

from django.test import SimpleTestCase

from retrieval_evaluation.metrics import dcg_at_k, idcg_at_k, ndcg_at_k
from retrieval_evaluation.metrics.ndcg import gain

from .helpers import WORKED_GRADES, WORKED_RETRIEVED


class NDCGTests(SimpleTestCase):
    def test_worked_example_matches_design(self):
        # DCG  = 7/log2(3) + 3/log2(5)          = 5.709
        # IDCG = 7/1 + 3/log2(3) + 1/log2(4)    = 9.393
        dcg = 7 / math.log2(3) + 3 / math.log2(5)
        idcg = 7 + 3 / math.log2(3) + 1 / 2
        self.assertAlmostEqual(dcg_at_k(WORKED_RETRIEVED, WORKED_GRADES, 5), dcg)
        self.assertAlmostEqual(idcg_at_k(WORKED_GRADES, 5), idcg)
        self.assertAlmostEqual(ndcg_at_k(WORKED_RETRIEVED, WORKED_GRADES, 5), dcg / idcg)
        self.assertEqual(round(ndcg_at_k(WORKED_RETRIEVED, WORKED_GRADES, 5), 3), 0.608)

    def test_linear_gain(self):
        dcg = 3 / math.log2(3) + 2 / math.log2(5)
        idcg = 3 + 2 / math.log2(3) + 1 / 2
        self.assertAlmostEqual(ndcg_at_k(WORKED_RETRIEVED, WORKED_GRADES, 5, "linear"), dcg / idcg)

    def test_perfect_and_reversed_ranking(self):
        self.assertAlmostEqual(ndcg_at_k(["C101", "C102", "C103"], WORKED_GRADES, 3), 1.0)
        reversed_score = ndcg_at_k(["C103", "C102", "C101"], WORKED_GRADES, 3)
        self.assertLess(reversed_score, 1.0)
        self.assertGreater(reversed_score, 0.0)

    def test_ideal_dcg_is_cut_at_k(self):
        # With K = 1 only the grade-3 chunk fits in the ideal ranking.
        self.assertAlmostEqual(idcg_at_k(WORKED_GRADES, 1), 7.0)
        self.assertAlmostEqual(ndcg_at_k(["C101"], WORKED_GRADES, 1), 1.0)

    def test_grade_zero_adds_nothing(self):
        grades = {"C1": 3, "C2": 0}
        self.assertAlmostEqual(ndcg_at_k(["C2", "C1"], grades, 2), (7 / math.log2(3)) / 7)

    def test_undefined_without_relevant_chunks(self):
        with self.assertRaises(ValueError):
            ndcg_at_k(["C1"], {"C1": 0}, 5)

    def test_gain_validation(self):
        self.assertEqual(gain(3), 7.0)
        self.assertEqual(gain(3, "linear"), 3.0)
        with self.assertRaises(ValueError):
            gain(-1)
        with self.assertRaises(ValueError):
            gain(1, "cubic")
