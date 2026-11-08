"""18. ROUGE, with rouge-score.

ROUGE-1 Recall = overlapping reference unigrams / total reference unigrams.
This is the score used for pass/fail. ROUGE-1, ROUGE-2 and ROUGE-L precision,
recall and F1 are in the details. Words are lowercased and Porter-stemmed
("leaves" and "leave" match).
"""

from collections import Counter
from functools import lru_cache

from rouge_score import rouge_scorer, tokenizers

from .base import MetricOutcome, package_version


@lru_cache(maxsize=1)
def _scorer() -> rouge_scorer.RougeScorer:
    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


@lru_cache(maxsize=1)
def _tokenizer() -> tokenizers.DefaultTokenizer:
    return tokenizers.DefaultTokenizer(use_stemmer=True)


def rouge(generated: str, reference: str) -> MetricOutcome:
    scores = _scorer().score(target=reference, prediction=generated)
    # Count the overlap with the same tokenizer, to show the numbers behind ROUGE-1 recall.
    ref_tokens = _tokenizer().tokenize(reference)
    gen_tokens = _tokenizer().tokenize(generated)
    overlap = sum((Counter(ref_tokens) & Counter(gen_tokens)).values())
    recall = scores["rouge1"].recall
    return MetricOutcome(
        score=recall,
        explanation=f"{overlap} overlapping reference unigrams / {len(ref_tokens)} total reference unigrams = {recall:.3f}",
        details={
            "overlapping_unigrams": overlap,
            "reference_unigrams": len(ref_tokens),
            **{
                name: {"precision": s.precision, "recall": s.recall, "f1": s.fmeasure}
                for name, s in scores.items()
            },
        },
        framework_version=package_version("rouge-score"),
    )
