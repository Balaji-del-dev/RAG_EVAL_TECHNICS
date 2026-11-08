"""The step-by-step lesson on the explanation page (analysis/run_lesson.py)."""

import json
from pathlib import Path

from django.test import TestCase
from django.urls import reverse

from retrieval_evaluation.analysis.run_lesson import build_lesson
from retrieval_evaluation.services.dataset_loader import import_dataset, parse_dataset_file
from retrieval_evaluation.services.evaluation_runner import create_run, execute_run
from retrieval_evaluation.services.retrieval_adapter import parse_results_file

from .helpers import WORKED_DATASET, WORKED_RETRIEVED

SAMPLES = Path(__file__).resolve().parent.parent / "sample_data"


def flat(parts) -> str:
    """Formula parts as one line: fractions become 'a/b', sums 'Σ[i=1..N]'."""
    out = []
    for p in parts:
        if "frac" in p:
            out.append("{}/{}".format(*p["frac"]))
        elif "sum" in p:
            out.append("Σ[{}..{}]".format(*p["sum"]))
        else:
            out.append(p["t"])
    return " ".join(out)


def precomputed_run(data: dict, ranked: dict, k=5):
    dataset = import_dataset(parse_dataset_file("d.json", json.dumps(data).encode()))
    results = {qid: [{"chunk_id": c, "score": None} for c in ids] for qid, ids in ranked.items()}
    return execute_run(create_run(dataset, name="t", top_k=k, retriever_type="precomputed",
                                  retriever_config={"results": results}))


class LeavePolicyExampleTests(TestCase):
    """The bundled HR leave sample reproduces the hand-worked lesson exactly."""

    def setUp(self):
        dataset = import_dataset(parse_dataset_file(
            "hr_leave_example.json", (SAMPLES / "hr_leave_example.json").read_bytes()))
        results = parse_results_file(
            "leave_example_results.json", (SAMPLES / "leave_example_results.json").read_bytes())
        self.run_obj = execute_run(create_run(dataset, name="lesson", top_k=5, retriever_type="precomputed",
                                              retriever_config={"results": results}))
        self.lesson = build_lesson(self.run_obj)

    def test_example_question_and_values(self):
        ex = self.lesson["example"]
        self.assertEqual(ex["result"].query_external_id, "Q1")
        self.assertEqual((ex["total_relevant"], ex["returned"], ex["found"]), (3, 5, 2))
        self.assertEqual(ex["missed_ids"], ["D3"])
        self.assertEqual([r["chunk_id"] for r in ex["retrieved_rows"]], ["D1", "D4", "D2", "D5", "D6"])

    def test_six_calculations(self):
        steps = {s["key"]: s for s in self.lesson["steps"]}
        calc = {key: " | ".join(flat(c["parts"]) for c in s["calc"]) for key, s in steps.items()}
        self.assertEqual(calc["precision"], "= 2/5 = 0.40")
        self.assertEqual(calc["recall"], "= 2/3 = 0.667")
        self.assertEqual(calc["f1"], "= 2 × 0.40 × 0.667/0.40 + 0.667 = 0.50")
        self.assertEqual(calc["hit_rate"], "= 2/3 = 66.7%")
        self.assertEqual(calc["mrr"], "= 1 + 0.5 + 0/3 = 0.50")
        self.assertEqual(
            calc["ndcg"],
            "DCG = 1/log₂(2) + 0 + 1/log₂(4) = 1 + 0 + 0.5 = 1.5"
            " | IDCG = 1/log₂(2) + 1/log₂(3) + 1/log₂(4) = 1 + 0.631 + 0.5 = 2.131"
            " | NDCG = 1.5/2.131 = 0.704",
        )
        self.assertEqual(flat(steps["mrr"]["formula"][0]), "MRR = 1/N Σ[i=1..N] 1/rankᵢ")
        self.assertIn("One correct chunk is missing (D3)", steps["recall"]["meaning"])

    def test_summary_table(self):
        rows = [(r["label"], r["calc"], r["result"], r["run"]) for r in self.lesson["summary_rows"]]
        self.assertEqual(rows, [
            ("Precision@5", "2 / 5", "40%", "26.7%"),
            ("Recall@5", "2 / 3", "66.7%", "55.6%"),
            ("F1 Score", "2 × (0.40 × 0.667) / (0.40 + 0.667)", "50%", "35.7%"),
            ("Hit Rate@5", "2 / 3 questions", "66.7%", "66.7%"),
            ("MRR", "(1 + 0.5 + 0) / 3", "0.50", "0.500"),
            ("NDCG@5", "1.5 / 2.131", "0.704", "0.452"),
        ])

    def test_all_wrong_question_and_fix(self):
        fail = self.lesson["failure"]
        self.assertFalse(fail["hypothetical"])
        self.assertEqual(fail["q"]["result"].query_external_id, "Q3")
        self.assertEqual(fail["wrong_count"], 5)
        self.assertEqual({r["result"] for r in fail["zero_rows"]}, {"0%", "0"})
        self.assertEqual([r["after"] for r in fail["before_after"]], ["40%", "100%", "57.1%", "100%", "1", "1"])
        # The question shares words with D11 and D12, so no single failure point is blamed.
        self.assertIsNone(fail["likely"])
        self.assertEqual(len(fail["fix_steps"]), 9)

    def test_page(self):
        url = reverse("retrieval_evaluation:run_explain", args=[self.run_obj.pk])
        page = self.client.get(url)
        for text in ("Step 1: Take one question from the dataset", "Step 2: Calculate all 6 metrics",
                     "Step 3: Final summary", "Step 4: When all retrieved chunks are wrong",
                     "Step 5: How to fix it, step by step", "Very important interview point",
                     "How they link together", "2.131"):
            self.assertContains(page, text)
        q2 = self.run_obj.results.get(query_external_id="Q2")
        page = self.client.get(url, {"q": q2.pk})
        self.assertEqual(page.context["lesson"]["example"]["result"].pk, q2.pk)
        page = self.client.get(url, {"q": "999999", "fail": "abc"})
        self.assertEqual(page.context["lesson"]["example"]["result"].query_external_id, "Q1")


class GradedAndHypotheticalTests(TestCase):
    def test_graded_ndcg_and_no_failed_question(self):
        run = precomputed_run(WORKED_DATASET, {"Q1": WORKED_RETRIEVED})
        lesson = build_lesson(run)
        ndcg = next(s for s in lesson["steps"] if s["key"] == "ndcg")
        self.assertEqual(flat(ndcg["formula"][1]), "DCG@5 = Σ[i=1..5] gainᵢ/log₂(i + 1)")
        self.assertEqual(flat(ndcg["calc"][0]["parts"]),
                         "DCG = 0 + 7/log₂(3) + 0 + 3/log₂(5) = 0 + 4.417 + 0 + 1.292 = 5.709")
        self.assertEqual(flat(ndcg["calc"][2]["parts"]), "NDCG = 5.709/9.393 = 0.608")
        # Every question had a hit, so Step 4 is a "suppose" built on the example question.
        fail = lesson["failure"]
        self.assertTrue(fail["hypothetical"])
        self.assertEqual([r["before"] for r in fail["before_after"]], ["0%", "0%", "0%", "0%", "0", "0"])
        self.assertEqual([r["after"] for r in fail["before_after"]], ["60%", "100%", "75%", "100%", "1", "1"])

    def test_grade_order_is_explained(self):
        run = precomputed_run(WORKED_DATASET, {"Q1": ["C103", "C101", "C102", "C104", "C105"]})
        ndcg = next(s for s in build_lesson(run)["steps"] if s["key"] == "ndcg")
        self.assertIn("C103 (grade 1) is at rank 1, above C101 (grade 3)", ndcg["meaning"])


class DiagnosisTests(TestCase):
    """For built-in retrievers the app re-ranks the whole corpus to see where a missed chunk is."""

    def make_run(self, chunks, question, target):
        data = {
            "name": "Diag", "version": "1",
            "chunks": [{"chunk_id": cid, "title": title, "text": text} for cid, title, text in chunks],
            "queries": [{"query_id": "Q1", "text": question, "ground_truth": [{"chunk_id": target, "grade": 3}]}],
        }
        dataset = import_dataset(parse_dataset_file("d.json", json.dumps(data).encode()))
        return execute_run(create_run(dataset, name="bm25", top_k=5, retriever_type="bm25", retriever_config={}))

    def test_vocabulary_mismatch(self):
        run = self.make_run([
            ("P1", "Paternity leave", "Male employees can take five days of paternity leave."),
            ("A1", "Annual leave", "Employees receive twenty days of annual leave and time off each year."),
            ("W1", "Working hours", "Office time is nine to six."),
        ], "Do new fathers get time off when a baby is born?", "P1")
        fail = build_lesson(run)["failure"]
        self.assertFalse(fail["hypothetical"])
        self.assertEqual(fail["diagnosis"]["likely"], "query_mismatch")
        texts = " ".join(c["text"] for c in fail["diagnosis"]["checks"])
        self.assertIn("share no keywords", texts)
        self.assertIn("P1 is never matched", texts)
        self.assertEqual(fail["words"][:2], ["paternity", "leave"])

    def test_ranked_low(self):
        others = [(f"L{i}", f"Leave policy {i}", f"Leave policy rules, leave policy part {i}.") for i in range(5)]
        run = self.make_run(others + [
            ("T1", "Fathers", "New fathers get some leave."),
            ("X1", "Canteen", "Lunch is served at noon."),
        ], "What is the leave policy?", "T1")
        fail = build_lesson(run)["failure"]
        self.assertEqual(fail["diagnosis"]["likely"], "ranked_low")
        self.assertIn("T1 comes at rank 6", " ".join(c["text"] for c in fail["diagnosis"]["checks"]))
