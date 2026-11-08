import json

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from retrieval_evaluation.models import EvaluationDataset, EvaluationRun, GroundTruth

from .helpers import WORKED_DATASET, WORKED_RETRIEVED


def url(name, *args):
    return reverse(f"retrieval_evaluation:{name}", args=args)


class ViewFlowTests(TestCase):
    def upload_dataset(self, data=WORKED_DATASET, **extra):
        f = SimpleUploadedFile("d.json", json.dumps(data).encode(), "application/json")
        return self.client.post(url("dataset_list"), {"file": f, **extra})

    def test_empty_pages_render(self):
        for name in ("dashboard", "dataset_list", "evaluation_create", "run_list", "comparison"):
            with self.subTest(name=name):
                self.assertEqual(self.client.get(url(name)).status_code, 200)

    def test_sample_datasets_load_once(self):
        self.client.post(url("load_sample"))
        self.client.post(url("load_sample"))
        self.assertEqual(EvaluationDataset.objects.count(), 3)

    def test_upload_errors_are_shown(self):
        bad = SimpleUploadedFile("d.json", b'{"queries": []}', "application/json")
        response = self.client.post(url("dataset_list"), {"file": bad})
        self.assertContains(response, "The dataset has no queries.")

    def test_full_flow(self):
        response = self.upload_dataset()
        dataset = EvaluationDataset.objects.get()
        self.assertRedirects(response, dataset.get_absolute_url())

        # Uploaded-results run reproduces the worked example.
        results = SimpleUploadedFile("r.json", json.dumps({"Q1": WORKED_RETRIEVED}).encode())
        response = self.client.post(url("evaluation_create"), {
            "dataset": dataset.pk, "name": "Worked", "top_k": 5, "retriever_type": "precomputed",
            "results_file": results,
        })
        run = EvaluationRun.objects.get()
        self.assertRedirects(response, run.get_absolute_url())
        page = self.client.get(run.get_absolute_url())
        for text in ("40%", "66.7%", "50%", "100%", "0.500", "0.608", "Missing chunks"):
            self.assertContains(page, text)

        result = run.results.get()
        detail = self.client.get(url("result_detail", run.pk, result.pk))
        self.assertContains(detail, "DCG 5.709 / ideal DCG 9.393 = 0.608")
        self.assertContains(detail, "Found at rank 2")

        # Built-in run, then compare.
        self.client.post(url("evaluation_create"), {
            "dataset": dataset.pk, "name": "BM25", "top_k": 10, "retriever_type": "bm25",
            "k1": 1.2, "b": 0.75, "stemming": "on",
        })
        bm25 = EvaluationRun.objects.get(name="BM25")
        self.assertEqual(bm25.status, "completed")
        self.assertEqual(bm25.retriever_config, {"stemming": True, "k1": 1.2, "b": 0.75})
        cmp = self.client.get(url("comparison"), {"baseline": run.pk, "candidate": bm25.pk})
        self.assertContains(cmp, "different K")
        self.assertContains(cmp, "Per-query changes")

        # Exports
        csv_response = self.client.get(url("run_export", run.pk, "csv"))
        self.assertIn("C104 C101 C105 C102 C106", csv_response.content.decode())
        payload = json.loads(self.client.get(url("run_export", run.pk, "json")).content)
        self.assertAlmostEqual(payload["run"]["averages"]["ndcg"], 0.6078, places=4)
        self.assertEqual(self.client.get(url("run_export", run.pk, "xml")).status_code, 404)

        # Re-run and delete
        self.client.post(url("run_rerun", run.pk))
        self.assertEqual(EvaluationRun.objects.filter(name__endswith="(re-run)").count(), 1)
        self.client.post(url("run_delete", bm25.pk))
        self.assertFalse(EvaluationRun.objects.filter(pk=bm25.pk).exists())

    def test_ground_truth_management(self):
        self.upload_dataset()
        dataset = EvaluationDataset.objects.get()
        query = dataset.queries.get()
        page = url("query_detail", dataset.pk, query.pk)
        self.client.post(page, {"action": "save_label", "label-chunk_id": "C104", "label-grade": 1})
        self.client.post(page, {"action": "save_label", "label-chunk_id": "C101", "label-grade": 2})
        self.assertEqual(dict(query.ground_truth.values_list("chunk_id", "grade")),
                         {"C101": 2, "C102": 2, "C103": 1, "C104": 1})
        label = GroundTruth.objects.get(query=query, chunk_id="C104")
        self.client.post(page, {"action": "delete_label", "label_id": label.pk})
        self.client.post(page, {"action": "edit_query", "edit-text": "Maternity leave rules?"})
        query.refresh_from_db()
        self.assertEqual(query.text, "Maternity leave rules?")
        self.assertFalse(query.ground_truth.filter(chunk_id="C104").exists())

        # Add a query, then delete it
        self.client.post(url("dataset_detail", dataset.pk),
                         {"query_id": "Q2", "text": "Sick leave?", "chunk_id": "C105", "grade": 3})
        q2 = dataset.queries.get(query_id="Q2")
        self.client.post(url("query_delete", dataset.pk, q2.pk))
        self.assertFalse(dataset.queries.filter(query_id="Q2").exists())

    def test_evaluation_form_validation(self):
        self.upload_dataset(data=dict(WORKED_DATASET, chunks=[]))
        dataset = EvaluationDataset.objects.get()
        base = {"dataset": dataset.pk, "name": "x", "top_k": 5}
        response = self.client.post(url("evaluation_create"), base | {"retriever_type": "bm25"})
        self.assertContains(response, "no chunk corpus")
        response = self.client.post(url("evaluation_create"), base | {"retriever_type": "http", "http_url": "file:///x"})
        self.assertContains(response, "must start with http")
        response = self.client.post(url("evaluation_create"), base | {"retriever_type": "precomputed"})
        self.assertContains(response, "Upload the ranked results")
        response = self.client.post(url("evaluation_create"), base | {"retriever_type": "bm25", "top_k": 7})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(EvaluationRun.objects.exists())

    def test_csv_export_neutralises_formulas(self):
        data = json.loads(json.dumps(WORKED_DATASET))
        data["queries"][0]["text"] = "=HYPERLINK(\"http://x\")"
        self.upload_dataset(data=data)
        dataset = EvaluationDataset.objects.get()
        results = SimpleUploadedFile("r.json", json.dumps({"Q1": WORKED_RETRIEVED}).encode())
        self.client.post(url("evaluation_create"), {"dataset": dataset.pk, "name": "x", "top_k": 5,
                                                     "retriever_type": "precomputed", "results_file": results})
        body = self.client.get(url("run_export", EvaluationRun.objects.get().pk, "csv")).content.decode()
        self.assertIn("'=HYPERLINK", body)

    def test_dataset_delete_cascades(self):
        self.upload_dataset()
        dataset = EvaluationDataset.objects.get()
        self.client.post(url("dataset_delete", dataset.pk))
        self.assertFalse(EvaluationDataset.objects.exists())

    def test_get_does_not_delete(self):
        self.upload_dataset()
        dataset = EvaluationDataset.objects.get()
        self.assertEqual(self.client.get(url("dataset_delete", dataset.pk)).status_code, 405)
