"""15 Semantic Similarity: cosine of the two answer embeddings."""

import os

import numpy as np
import pytest

from generation_eval.metrics import SentenceTransformerSimilarity, cosine_similarity

from .conftest import bag_of_words


def test_cosine_formula():
    cosine, dot, norm_a, norm_b = cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
    assert dot == pytest.approx(1.0)
    assert norm_a == pytest.approx(1.0)
    assert norm_b == pytest.approx(2 ** 0.5)
    assert cosine == pytest.approx(1 / 2 ** 0.5)


def test_zero_vector_gives_zero():
    assert cosine_similarity(np.zeros(3), np.ones(3))[0] == 0.0


def test_explanation_shows_the_working():
    outcome = SentenceTransformerSimilarity("fake", encoder=bag_of_words).score("a b", "a c")
    assert outcome.score == pytest.approx(0.5)
    assert outcome.explanation == "(A · B) / (‖A‖ × ‖B‖) = 1.000 / (1.414 × 1.414) = 0.500"


def test_opposite_vectors_score_zero():
    outcome = SentenceTransformerSimilarity("fake", encoder=lambda texts: np.array([[1.0, 0.0], [-1.0, 0.0]])).score("x", "y")
    assert outcome.score == 0.0
    assert outcome.details["cosine"] == pytest.approx(-1.0)


@pytest.mark.models
@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="loads torch + all-MiniLM-L6-v2; set RUN_MODEL_TESTS=1")
def test_real_model_understands_paraphrase():
    similarity = SentenceTransformerSimilarity("sentence-transformers/all-MiniLM-L6-v2", threads=1)
    reference = "Employees receive 12 casual leaves and 12 sick leaves per year."
    paraphrase = similarity.score("Each employee gets 12 days of casual leave and 12 days of sick leave annually.", reference)
    unrelated = similarity.score("Employees can apply for work from home.", reference)
    assert paraphrase.score > 0.8
    assert unrelated.score < paraphrase.score - 0.3
