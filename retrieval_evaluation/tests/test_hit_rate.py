from django.test import SimpleTestCase

from retrieval_evaluation.metrics import hit_rate_at_k

from .helpers import WORKED_RELEVANT, WORKED_RETRIEVED


class HitRateTests(SimpleTestCase):
    def test_worked_example(self):
        self.assertEqual(hit_rate_at_k(WORKED_RETRIEVED, WORKED_RELEVANT, 5), 1.0)

    def test_relevant_outside_k_is_a_miss(self):
        # First relevant chunk (C101) is at rank 2
        self.assertEqual(hit_rate_at_k(WORKED_RETRIEVED, WORKED_RELEVANT, 1), 0.0)

    def test_nothing_relevant(self):
        self.assertEqual(hit_rate_at_k(["C104", "C105"], WORKED_RELEVANT, 5), 0.0)
        self.assertEqual(hit_rate_at_k([], WORKED_RELEVANT, 5), 0.0)
