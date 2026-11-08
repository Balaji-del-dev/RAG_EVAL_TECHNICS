from django.test import SimpleTestCase

from retrieval_evaluation.services.builtin_retrievers import (
    BM25Index, Document, ScoredChunk, TfidfIndex, light_stem, reciprocal_rank_fusion, tokenize,
)

DOCS = [
    Document("A", "maternity leave lasts 26 weeks"),
    Document("B", "sick leave is 12 days"),
    Document("C", "password policies and account security"),
]


class BuiltinRetrieverTests(SimpleTestCase):
    def test_tokenize_and_stem(self):
        self.assertEqual(tokenize("What is the Maternity leave?"), ["maternity", "leave"])
        self.assertEqual(light_stem("policies"), "policy")
        self.assertEqual(light_stem("leaves"), "leave")
        self.assertEqual(light_stem("process"), "process")

    def test_bm25_ranks_matching_document_first(self):
        ranked = BM25Index(DOCS).search("maternity leave")
        self.assertEqual(ranked[0].chunk_id, "A")
        self.assertEqual({r.chunk_id for r in ranked}, {"A", "B"})  # zero scores are dropped

    def test_stemming_changes_matches(self):
        self.assertEqual(BM25Index(DOCS).search("password policy"), BM25Index(DOCS).search("password"))
        stemmed = BM25Index(DOCS, stemming=True).search("policy")
        self.assertEqual([r.chunk_id for r in stemmed], ["C"])

    def test_tfidf(self):
        ranked = TfidfIndex(DOCS).search("sick days")
        self.assertEqual(ranked[0].chunk_id, "B")
        self.assertLessEqual(ranked[0].score, 1.0 + 1e-9)

    def test_rrf(self):
        fused = reciprocal_rank_fusion([[ScoredChunk("A", 2), ScoredChunk("B", 1)], [ScoredChunk("B", 5)]], rrf_k=60)
        self.assertEqual(fused[0].chunk_id, "B")
        self.assertAlmostEqual(fused[0].score, 1 / 62 + 1 / 61)

    def test_no_overlap_returns_nothing(self):
        self.assertEqual(BM25Index(DOCS).search("zebra"), [])
