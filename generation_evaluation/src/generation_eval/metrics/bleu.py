"""17. BLEU, with SacreBLEU.

BLEU = BP × exp(Σ wₙ log pₙ), n = 1..4, wₙ = 1/4
  pₙ  = n-gram precision: answer n-grams found in the reference / answer n-grams
  BP  = brevity penalty: 1 if the answer is longer than the reference, else exp(1 − ref_len / answer_len)

Sentence-level BLEU uses SacreBLEU's default 'exp' smoothing, so a missing
4-gram does not force the whole score to 0. SacreBLEU reports 0-100; we divide by 100.
"""

import sacrebleu

from .base import MetricOutcome, package_version


def bleu(generated: str, reference: str) -> MetricOutcome:
    result = sacrebleu.sentence_bleu(generated, [reference])
    precisions = [p / 100 for p in result.precisions]
    p_text = ", ".join(f"p{n} = {p:.3f}" for n, p in enumerate(precisions, start=1))
    return MetricOutcome(
        score=result.score / 100,
        explanation=f"BP {result.bp:.3f} × exp(¼ × Σ log pₙ) with {p_text} = {result.score / 100:.3f}",
        details={
            "ngram_precisions": precisions,
            "matching_ngrams": list(result.counts),
            "total_ngrams": list(result.totals),
            "brevity_penalty": result.bp,
            "answer_length": result.sys_len,
            "reference_length": result.ref_len,
            "max_ngram_order": 4,
            "weights": [0.25, 0.25, 0.25, 0.25],
        },
        framework_version=package_version("sacrebleu"),
    )
