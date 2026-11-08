"""15. Semantic Similarity, with Sentence Transformers.

Similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
  A = embedding of the generated answer, B = embedding of the reference answer.

The cosine is computed here from the two vectors so the working is visible.
Negative cosines (opposite meaning) are reported as a score of 0.
"""

from collections.abc import Callable, Sequence
from functools import lru_cache

import numpy as np

from .base import MetricOutcome, package_version

Encoder = Callable[[Sequence[str]], np.ndarray]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """(cosine, dot product, ‖a‖, ‖b‖)."""
    dot = float(np.dot(a, b))
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cosine = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
    return cosine, dot, norm_a, norm_b


@lru_cache(maxsize=2)
def _load_model(name: str, threads: int):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    return SentenceTransformer(name, device="cpu")


class SentenceTransformerSimilarity:
    """Loads the model on first use and keeps it for later requests."""

    def __init__(self, model_name: str, threads: int = 2, encoder: Encoder | None = None):
        self.model_name = model_name
        self.threads = threads
        self._encoder = encoder

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._encoder is not None:
            return np.asarray(self._encoder(texts), dtype=float)
        model = _load_model(self.model_name, self.threads)
        return model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False)

    def score(self, generated: str, reference: str) -> MetricOutcome:
        a, b = self.encode([generated, reference])
        cosine, dot, norm_a, norm_b = cosine_similarity(a, b)
        return MetricOutcome(
            score=max(0.0, cosine),
            explanation=f"(A · B) / (‖A‖ × ‖B‖) = {dot:.3f} / ({norm_a:.3f} × {norm_b:.3f}) = {cosine:.3f}",
            details={
                "cosine": cosine,
                "dot_product": dot,
                "norm_generated": norm_a,
                "norm_reference": norm_b,
                "dimensions": int(len(a)),
                "model": self.model_name,
            },
            framework_version=package_version("sentence-transformers"),
        )
