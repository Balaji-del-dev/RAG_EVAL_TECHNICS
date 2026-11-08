"""16. Exact Match (EM), plain Python.

EM = 1 if Generated = Reference (after normalization), else 0.
Normalization follows the SQuAD convention: lowercase, remove punctuation and
the articles a/an/the, and collapse whitespace.
"""

import re
import string

from .base import MetricOutcome

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)

NORMALIZATION = "lowercase, remove punctuation, remove a/an/the, collapse spaces"


def normalize_answer(text: str) -> str:
    text = text.lower().translate(_PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(generated: str, reference: str) -> MetricOutcome:
    gen, ref = normalize_answer(generated), normalize_answer(reference)
    matched = gen == ref
    return MetricOutcome(
        score=1.0 if matched else 0.0,
        explanation=(
            "Match: the normalized answer equals the normalized reference, so EM = 1."
            if matched else
            "No match: the normalized answer differs from the normalized reference, so EM = 0."
        ),
        details={"normalized_generated": gen, "normalized_reference": ref, "normalization": NORMALIZATION},
        framework_version="python",
    )
