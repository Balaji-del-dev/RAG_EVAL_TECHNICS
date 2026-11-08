"""Metrics that need no LLM judge: Exact Match (Python), BLEU (SacreBLEU), ROUGE (rouge-score)
and Semantic Similarity (Sentence Transformers)."""

from .base import MetricOutcome
from .bleu import bleu
from .exact_match import exact_match, normalize_answer
from .rouge import rouge
from .semantic_similarity import SentenceTransformerSimilarity, cosine_similarity

# Pure text-comparison metrics: (generated answer, reference answer) -> MetricOutcome
TEXT_METRICS = {"exact_match": exact_match, "bleu": bleu, "rouge": rouge}

__all__ = [
    "TEXT_METRICS", "MetricOutcome", "SentenceTransformerSimilarity",
    "bleu", "cosine_similarity", "exact_match", "normalize_answer", "rouge",
]
