from django.test import SimpleTestCase

from retrieval_evaluation.metrics import recall_at_k

from .helpers import WORKED_RELEVANT, WORKED_RETRIEVED


class RecallAtKTests(SimpleTestCase):
    def test_worked_example(self):
        # 2 of the 3 relevant chunks are in the top 5
        self.assertAlmostEqual(recall_at_k(WORKED_RETRIEVED, WORKED_RELEVANT, 5), 2 / 3)

    def test_k_limits_recall(self):
        self.assertAlmostEqual(recall_at_k(WORKED_RETRIEVED, WORKED_RELEVANT, 2), 1 / 3)
        self.assertAlmostEqual(recall_at_k(WORKED_RETRIEVED, WORKED_RELEVANT, 1), 0.0)

    def test_all_found(self):
        self.assertAlmostEqual(recall_at_k(["C103", "C102", "C101"], WORKED_RELEVANT, 3), 1.0)

    def test_undefined_without_relevant_chunks(self):
        with self.assertRaises(ValueError):
            recall_at_k(WORKED_RETRIEVED, set(), 5)
