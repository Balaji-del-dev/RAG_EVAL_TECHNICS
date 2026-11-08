import io
from pathlib import Path

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from retrieval_evaluation.models import EvaluationDataset, EvaluationRun

SAMPLES = Path(__file__).resolve().parent.parent / "sample_data"


class CommandTests(TestCase):
    def call(self, *args):
        out = io.StringIO()
        call_command(*args, stdout=out)
        return out.getvalue()

    def test_load_run_and_compare(self):
        self.call("load_sample_data")
        self.assertEqual(EvaluationDataset.objects.count(), 3)
        out = self.call("run_evaluation", "--dataset", "HR Worked Example", "--retriever", "precomputed",
                        "--results", str(SAMPLES / "worked_example_results.json"), "--k", "5")
        self.assertIn("NDCG@5         0.608", out)
        out = self.call("run_evaluation", "--dataset", "HR Leave Policy Example", "--retriever", "precomputed",
                        "--results", str(SAMPLES / "leave_example_results.json"), "--k", "5")
        self.assertIn("MRR            0.500", out)

        self.call("run_evaluation", "--dataset", "HR Policy Evaluation@1.0", "--retriever", "tfidf", "--name", "tfidf")
        self.call("run_evaluation", "--dataset", "HR Policy Evaluation", "--retriever", "fusion", "--stemming", "--name", "fusion")
        tfidf, fusion = EvaluationRun.objects.get(name="tfidf"), EvaluationRun.objects.get(name="fusion")
        self.assertIn("Verdict: improvement", self.call("compare_runs", str(tfidf.pk), str(fusion.pk)))
        with self.assertRaises(CommandError):
            self.call("compare_runs", str(fusion.pk), str(tfidf.pk), "--fail-on-regression")

    def test_load_dataset_csv(self):
        out = self.call("load_dataset", str(SAMPLES / "hr_policy_ground_truth.csv"), "--name", "CSV GT")
        self.assertIn("14 queries", out)
        with self.assertRaises(CommandError):
            self.call("load_dataset", str(SAMPLES / "hr_policy_ground_truth.csv"))  # no name
