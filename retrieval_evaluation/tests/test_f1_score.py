from django.test import SimpleTestCase

from retrieval_evaluation.metrics import f1_at_k, f1_from_precision_recall

from .helpers import WORKED_RELEVANT, WORKED_RETRIEVED


class F1Tests(SimpleTestCase):
    def test_worked_example(self):
        # 2 x 0.4 x 0.667 / (0.4 + 0.667) = 0.5
        self.assertAlmostEqual(f1_at_k(WORKED_RETRIEVED, WORKED_RELEVANT, 5), 0.5)

    def test_harmonic_mean(self):
        self.assertAlmostEqual(f1_from_precision_recall(1.0, 0.5), 2 / 3)
        self.assertEqual(f1_from_precision_recall(0.0, 0.0), 0.0)
        self.assertEqual(f1_from_precision_recall(1.0, 0.0), 0.0)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            f1_from_precision_recall(-0.1, 0.5)
