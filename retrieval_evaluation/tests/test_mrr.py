from django.test import SimpleTestCase

from retrieval_evaluation.metrics import first_relevant_rank, mean_reciprocal_rank, reciprocal_rank

from .helpers import WORKED_RELEVANT, WORKED_RETRIEVED


class MRRTests(SimpleTestCase):
    def test_worked_example(self):
        # First relevant chunk at rank 2 -> 1 / 2
        self.assertEqual(first_relevant_rank(WORKED_RETRIEVED, WORKED_RELEVANT, 5), 2)
        self.assertAlmostEqual(reciprocal_rank(WORKED_RETRIEVED, WORKED_RELEVANT, 5), 0.5)

    def test_rank_one_and_none(self):
        self.assertEqual(reciprocal_rank(["C101"], WORKED_RELEVANT, 5), 1.0)
        self.assertEqual(reciprocal_rank(["C104", "C105"], WORKED_RELEVANT, 5), 0.0)
        self.assertIsNone(first_relevant_rank(["C104"], WORKED_RELEVANT, 5))

    def test_relevant_beyond_k_counts_as_zero(self):
        self.assertEqual(reciprocal_rank(WORKED_RETRIEVED, WORKED_RELEVANT, 1), 0.0)

    def test_mean_over_queries(self):
        # Queries with first relevant at rank 1, 2 and none: (1 + 0.5 + 0) / 3
        self.assertAlmostEqual(mean_reciprocal_rank([1.0, 0.5, 0.0]), 0.5)
        with self.assertRaises(ValueError):
            mean_reciprocal_rank([])

    def test_duplicates_do_not_shift_rank(self):
        self.assertEqual(first_relevant_rank(["C104", "C104", "C101"], WORKED_RELEVANT, 5), 2)
