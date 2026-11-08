from django.test import SimpleTestCase

from retrieval_evaluation.metrics import precision_at_k

from .helpers import WORKED_RELEVANT, WORKED_RETRIEVED


class PrecisionAtKTests(SimpleTestCase):
    def test_worked_example(self):
        # 2 relevant (C101, C102) in the top 5 -> 2 / 5
        self.assertAlmostEqual(precision_at_k(WORKED_RETRIEVED, WORKED_RELEVANT, 5), 0.4)

    def test_cut_at_k(self):
        # Top 2 = C104, C101 -> 1 relevant / 2
        self.assertAlmostEqual(precision_at_k(WORKED_RETRIEVED, WORKED_RELEVANT, 2), 0.5)

    def test_fewer_results_than_k_divides_by_k(self):
        self.assertAlmostEqual(precision_at_k(["C101"], WORKED_RELEVANT, 5), 0.2)

    def test_duplicates_count_once(self):
        self.assertAlmostEqual(precision_at_k(["C101", "C101", "C102"], WORKED_RELEVANT, 2), 1.0)
        self.assertAlmostEqual(precision_at_k(["C101", "C101", "C104"], WORKED_RELEVANT, 2), 0.5)

    def test_empty_and_invalid_k(self):
        self.assertEqual(precision_at_k([], WORKED_RELEVANT, 5), 0.0)
        for bad in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                precision_at_k(WORKED_RETRIEVED, WORKED_RELEVANT, bad)
