"""16 Exact Match (Python), 17 BLEU (SacreBLEU), 18 ROUGE (rouge-score)."""

import pytest

from generation_eval.metrics import bleu, exact_match, normalize_answer, rouge

REFERENCE = "Employees receive 12 casual leaves and 12 sick leaves per year."


class TestExactMatch:
    def test_same_answer_after_normalization(self):
        assert exact_match("employees receive 12 casual leaves and 12 sick leaves per year", REFERENCE).score == 1.0
        assert exact_match("The 12 casual leaves", "12 casual leaves.").score == 1.0

    def test_one_different_number_is_no_match(self):
        outcome = exact_match("12 casual leaves and 10 sick leaves", "12 casual leaves and 12 sick leaves")
        assert outcome.score == 0.0
        assert "No match" in outcome.explanation

    def test_normalization(self):
        assert normalize_answer("  The Casual-Leave,   is 12!  ") == "casualleave is 12"


class TestBleu:
    def test_identical_sentence_scores_one(self):
        text = "Employees receive 12 casual leaves per year"
        outcome = bleu(text, text)
        assert outcome.score == pytest.approx(1.0)
        assert outcome.details["ngram_precisions"] == pytest.approx([1.0, 1.0, 1.0, 1.0])
        assert outcome.details["brevity_penalty"] == pytest.approx(1.0)

    def test_short_answer_gets_a_brevity_penalty(self):
        outcome = bleu("Employees receive 12 casual leaves.", REFERENCE)
        assert outcome.details["brevity_penalty"] < 1.0
        assert 0 < outcome.score < 0.5

    def test_paraphrase_scores_low(self):
        outcome = bleu("Each employee gets 12 days of casual leave and 12 days of sick leave annually.", REFERENCE)
        assert outcome.score < 0.3
        assert outcome.explanation.startswith("BP ")


class TestRouge:
    def test_partial_answer_recall(self):
        # The reference has 11 words; the answer covers 5 of them
        # (employees, receive, 12, casual, leaves): 5 / 11 = 0.4545.
        outcome = rouge("Employees receive 12 casual leaves", "Employees receive 12 casual leaves and 12 sick leaves per year")
        assert outcome.score == pytest.approx(5 / 11)
        assert outcome.explanation.startswith("5 overlapping reference unigrams / 11 total reference unigrams")
        assert outcome.details["rouge1"]["precision"] == pytest.approx(1.0)

    def test_identical_answer(self):
        assert rouge(REFERENCE, REFERENCE).score == pytest.approx(1.0)

    def test_stemming_matches_word_forms(self):
        assert rouge("leave", "leaves").score == pytest.approx(1.0)
