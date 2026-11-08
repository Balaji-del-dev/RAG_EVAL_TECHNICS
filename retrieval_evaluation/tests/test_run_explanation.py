import json

from django.test import TestCase
from django.urls import reverse

from retrieval_evaluation.analysis.run_explanation import explain_run
from retrieval_evaluation.services.dataset_loader import import_dataset, parse_dataset_file
from retrieval_evaluation.services.evaluation_runner import create_run, execute_run

from .helpers import WORKED_DATASET, WORKED_RETRIEVED


class RunExplanationTests(TestCase):
    def setUp(self):
        data = dict(WORKED_DATASET, queries=WORKED_DATASET["queries"] + [
            {"query_id": "Q2", "text": "insurance", "ground_truth": [{"chunk_id": "C106", "grade": 3}]},
        ])
        dataset = import_dataset(parse_dataset_file("d.json", json.dumps(data).encode()))
        results = {"Q1": [{"chunk_id": c, "score": None} for c in WORKED_RETRIEVED],
                   "Q2": [{"chunk_id": "C105", "score": None}]}
        self.run_obj = execute_run(create_run(dataset, name="e", top_k=5, retriever_type="precomputed",
                                              retriever_config={"results": results}))

    def test_numbers_come_from_the_run(self):
        x = explain_run(self.run_obj)
        self.assertEqual(x["counts"], {"slots": 10, "relevant": 2, "irrelevant": 4, "empty": 4,
                                       "total_relevant": 4, "precision_ceiling": 0.4})
        stories = {s.key: s for s in x["stories"]}
        self.assertEqual(stories["precision"].calculation, "2 relevant chunks in 2 × 5 = 10 slots = 20%")
        self.assertEqual(stories["hit_rate"].calculation, "1 of 2 queries had a hit = 1 ÷ 2 = 50%")
        self.assertEqual(stories["mrr"].calculation, "(1 × 1/2 + 1 × 0) ÷ 2 = 0.250")
        self.assertTrue(all(rule["holds"] for rule in x["rules"]))
        self.assertIn("Q2", x["priorities"][0])

    def test_page_renders(self):
        response = self.client.get(reverse("retrieval_evaluation:run_explain", args=[self.run_obj.pk]))
        self.assertContains(response, "What these scores mean")
        self.assertContains(response, "How they link together")
        self.assertContains(response, "1/2 = 0.5")  # MRR table: Q1's first relevant chunk is at rank 2
