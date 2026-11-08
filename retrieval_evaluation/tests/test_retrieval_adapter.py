import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from django.test import SimpleTestCase

from retrieval_evaluation.services.retrieval_adapter import (
    HttpRetriever, PrecomputedRetriever, QueryInput, ResultsFormatError, RetrieverError,
    normalize_ranked_list, parse_results_file,
)


class ResultsParsingTests(SimpleTestCase):
    def test_json_shapes(self):
        expected = ["C104", "C101"]
        shapes = [
            {"Q1": ["C104", "C101"]},
            {"retriever": "x", "results": {"Q1": [{"chunk_id": "C104"}, {"id": "C101"}]}},
            [{"query_id": "Q1", "retrieved": ["C104", "C101"]}],
            {"Q1": [{"chunk_id": "C101", "rank": 2}, {"chunk_id": "C104", "rank": 1}]},
        ]
        for shape in shapes:
            with self.subTest(shape=shape):
                parsed = parse_results_file("r.json", json.dumps(shape).encode())
                self.assertEqual([i["chunk_id"] for i in parsed["Q1"]], expected)

    def test_csv(self):
        content = b"query_id,chunk_id,rank,score\nQ1,C101,2,0.5\nQ1,C104,1,0.9\nQ2,C108,1,\n"
        parsed = parse_results_file("r.csv", content)
        self.assertEqual(parsed["Q1"], [{"chunk_id": "C104", "score": 0.9}, {"chunk_id": "C101", "score": 0.5}])
        self.assertEqual(parsed["Q2"], [{"chunk_id": "C108", "score": None}])

    def test_bad_files(self):
        bad = [
            ("r.json", b"[1, 2]"),
            ("r.json", b"{}"),
            ("r.json", b'{"Q1": "C1"}'),
            ("r.json", b'{"Q1": [{"chunk_id": "C1", "rank": 1}, {"chunk_id": "C2"}]}'),
            ("r.json", b'{"Q1": [{"score": 1}]}'),
            ("r.csv", b"query,chunk\nQ1,C1\n"),
            ("r.json", b"\xff\xfe"),
        ]
        for name, content in bad:
            with self.subTest(content=content), self.assertRaises(ResultsFormatError):
                parse_results_file(name, content)

    def test_precomputed_missing_query(self):
        adapter = PrecomputedRetriever({"results": {"Q1": [{"chunk_id": "C1", "score": None}]}})
        self.assertEqual(adapter.retrieve(QueryInput("Q1", "t"), 5)[0].chunk_id, "C1")
        with self.assertRaises(RetrieverError):
            adapter.retrieve(QueryInput("Q2", "t"), 5)

    def test_normalize_object_with_documents_key(self):
        items = normalize_ranked_list({"documents": [{"doc_id": "D1", "score": "0.5"}]})
        self.assertEqual((items[0].chunk_id, items[0].score), ("D1", 0.5))


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - http.server API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).last_request = {"body": body, "auth": self.headers.get("Authorization")}
        if body["query_id"] == "boom":
            self.send_response(500)
            self.end_headers()
            return
        payload = {"results": [{"chunk_id": "C104", "score": 0.9}, {"chunk_id": "C101", "score": 0.8}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class HttpRetrieverTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/retrieve"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        super().tearDownClass()

    def test_posts_query_and_reads_ranking(self):
        adapter = HttpRetriever({"url": self.url, "timeout": 5}, {"authorization": "Bearer t"})
        items = adapter.retrieve(QueryInput("Q1", "What is the maternity leave policy?"), 5)
        self.assertEqual([i.chunk_id for i in items], ["C104", "C101"])
        self.assertEqual(_Handler.last_request["body"]["top_k"], 5)
        self.assertEqual(_Handler.last_request["auth"], "Bearer t")

    def test_http_error_becomes_retriever_error(self):
        adapter = HttpRetriever({"url": self.url, "timeout": 5})
        with self.assertRaises(RetrieverError):
            adapter.retrieve(QueryInput("boom", "x"), 5)

    def test_url_scheme_checked(self):
        for url in ("file:///etc/passwd", "ftp://x", ""):
            with self.subTest(url=url), self.assertRaises(RetrieverError):
                HttpRetriever({"url": url})
