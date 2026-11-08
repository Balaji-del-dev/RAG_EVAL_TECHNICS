import json

from django.test import TestCase

from retrieval_evaluation.models import EvaluationQuery, EvaluationRun
from retrieval_evaluation.services.dataset_loader import import_dataset, parse_dataset_file
from retrieval_evaluation.services.evaluation_runner import create_run, execute_run, rerun

from .helpers import WORKED_DATASET, WORKED_RETRIEVED


class EvaluationRunnerTests(TestCase):
    def setUp(self):
        self.dataset = import_dataset(parse_dataset_file("d.json", json.dumps(WORKED_DATASET).encode()))

    def run_precomputed(self, results, **kwargs):
        run = create_run(self.dataset, name="t", top_k=5, retriever_type="precomputed",
                         retriever_config={"results": results}, **kwargs)
        return execute_run(run)

    def test_worked_example_end_to_end(self):
        run = self.run_precomputed({"Q1": [{"chunk_id": c, "score": None} for c in WORKED_RETRIEVED]})
        self.assertEqual(run.status, EvaluationRun.Status.COMPLETED, run.error_message)
        self.assertAlmostEqual(run.avg_precision, 0.4)
        self.assertAlmostEqual(run.avg_recall, 2 / 3)
        self.assertAlmostEqual(run.avg_f1, 0.5)
        self.assertAlmostEqual(run.avg_hit_rate, 1.0)
        self.assertAlmostEqual(run.mrr, 0.5)
        self.assertAlmostEqual(run.avg_ndcg, 0.6078, places=4)

        result = run.results.get()
        self.assertEqual((result.relevant_found, result.irrelevant_found, result.relevant_missed), (2, 3, 1))
        self.assertEqual(result.missed_chunk_ids, ["C103"])
        self.assertEqual(result.expected, {"C101": 3, "C102": 2, "C103": 1})
        self.assertEqual(list(result.retrieved.values_list("chunk_id", flat=True)), WORKED_RETRIEVED)
        self.assertEqual(sorted(result.issues), ["missing_chunks", "poor_ordering"])

    def test_builtin_bm25_runs(self):
        run = execute_run(create_run(self.dataset, name="bm25", top_k=5, retriever_type="bm25",
                                     retriever_config={"k1": 1.5, "b": 0.75}))
        self.assertEqual(run.status, "completed", run.error_message)
        self.assertEqual(run.results.get().retrieved.first().chunk_id is not None, True)

    def test_errored_queries_are_excluded_from_averages(self):
        q2 = EvaluationQuery.objects.create(dataset=self.dataset, query_id="Q2", text="unlabelled")
        run = self.run_precomputed({"Q1": [{"chunk_id": "C101", "score": None}]})
        self.assertEqual((run.evaluated_count, run.error_count), (1, 1))
        self.assertAlmostEqual(run.avg_precision, 0.2)
        self.assertIn("no relevant chunk", run.results.get(query=q2).error)

    def test_run_fails_when_nothing_can_be_evaluated(self):
        run = self.run_precomputed({"OTHER": [{"chunk_id": "C1", "score": None}]})
        self.assertEqual(run.status, EvaluationRun.Status.FAILED)
        self.assertIn("No query could be evaluated", run.error_message)
        self.assertFalse(run.results.exists())

    def test_builtin_without_corpus_fails_cleanly(self):
        self.dataset.chunks.all().delete()
        run = execute_run(create_run(self.dataset, name="x", top_k=5, retriever_type="bm25"))
        self.assertEqual(run.status, "failed")
        self.assertIn("no chunk corpus", run.error_message)

    def test_old_scores_survive_ground_truth_edits(self):
        run = self.run_precomputed({"Q1": [{"chunk_id": c, "score": None} for c in WORKED_RETRIEVED]})
        query = self.dataset.queries.get()
        query.ground_truth.filter(chunk_id="C103").delete()
        new = rerun(run)
        self.assertAlmostEqual(run.results.get().recall, 2 / 3)
        self.assertAlmostEqual(new.results.get().recall, 1.0)
        query.delete()
        self.assertEqual(run.results.get().query_text, "What is the maternity leave policy?")

    def test_linear_gain_is_stored_per_run(self):
        run = self.run_precomputed({"Q1": [{"chunk_id": c, "score": None} for c in WORKED_RETRIEVED]},
                                   ndcg_gain="linear")
        self.assertAlmostEqual(run.avg_ndcg, 0.5784, places=4)
