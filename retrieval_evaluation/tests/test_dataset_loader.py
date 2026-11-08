import json

from django.test import TestCase

from retrieval_evaluation.models import EvaluationDataset, GroundTruth
from retrieval_evaluation.services.dataset_loader import (
    DatasetFormatError, export_dataset, import_dataset, load_for_evaluation, parse_dataset_file,
)

from .helpers import WORKED_DATASET


class DatasetLoaderTests(TestCase):
    def parse(self, data, name="d.json"):
        content = json.dumps(data).encode() if not isinstance(data, bytes) else data
        return parse_dataset_file(name, content)

    def test_json_import_and_round_trip(self):
        dataset = import_dataset(self.parse(WORKED_DATASET), source_filename="d.json")
        self.assertEqual(dataset.chunks.count(), 6)
        self.assertEqual(GroundTruth.objects.filter(query__dataset=dataset).count(), 3)
        items = load_for_evaluation(dataset)
        self.assertEqual(items[0].expected, {"C101": 3, "C102": 2, "C103": 1})
        exported = export_dataset(dataset)
        exported["version"] = "2.0"
        again = import_dataset(self.parse(exported))
        self.assertEqual(load_for_evaluation(again)[0].expected, items[0].expected)

    def test_alternative_label_shapes(self):
        parsed = self.parse({"name": "x", "queries": [
            {"query_id": "A", "text": "a", "ground_truth": {"C1": 2}},
            {"query_id": "B", "query": "b", "relevant_chunks": ["C2", "C3"]},
        ]})
        self.assertEqual(parsed.queries[0]["ground_truth"], {"C1": 2})
        self.assertEqual(parsed.queries[1]["ground_truth"], {"C2": 1, "C3": 1})

    def test_csv(self):
        csv = b"query_id,query,chunk_id,grade\nQ1,What?,C1,3\nQ1,What?,C2,1\nQ2,Why?,C3,\n"
        parsed = self.parse(csv, "gt.csv")
        self.assertEqual(parsed.queries[0]["ground_truth"], {"C1": 3, "C2": 1})
        self.assertEqual(parsed.queries[1]["ground_truth"], {"C3": 1})
        with self.assertRaises(DatasetFormatError):
            import_dataset(parsed)  # CSV has no name
        self.assertEqual(import_dataset(parsed, name="CSV set").queries.count(), 2)

    def test_validation_errors(self):
        cases = [
            {"queries": []},
            {"queries": [{"query_id": "Q1", "text": "t", "ground_truth": [{"chunk_id": "C1", "grade": 5}]}]},
            {"queries": [{"query_id": "Q1", "text": "t", "ground_truth": [{"chunk_id": "C1", "grade": 0}]}]},
            {"queries": [{"query_id": "Q1", "text": "t", "relevant_chunks": ["C1"]},
                         {"query_id": "Q1", "text": "u", "relevant_chunks": ["C1"]}]},
            {"queries": [{"query_id": "Q1", "text": "t", "ground_truth": [{"chunk_id": "C1"}, {"chunk_id": "C1"}]}]},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(DatasetFormatError):
                self.parse(case)
        with self.assertRaises(DatasetFormatError):
            self.parse(b"{not json")
        with self.assertRaises(DatasetFormatError):
            self.parse(b"x", "data.txt")

    def test_unknown_chunk_ids_warn(self):
        data = dict(WORKED_DATASET, queries=[{"query_id": "Q", "text": "t", "relevant_chunks": ["NOPE"]}])
        self.assertIn("NOPE", self.parse(data).warnings[0])

    def test_duplicate_name_and_version_rejected(self):
        import_dataset(self.parse(WORKED_DATASET))
        with self.assertRaises(DatasetFormatError):
            import_dataset(self.parse(WORKED_DATASET))
        self.assertEqual(EvaluationDataset.objects.count(), 1)
