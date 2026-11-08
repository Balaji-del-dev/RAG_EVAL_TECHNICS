"""The eight generation metrics: what each checks, its formula, priority and frameworks.

This is the single source for the tree, the API's /metrics list, the CLI and the README.
"""

from dataclasses import dataclass
from typing import Literal

Priority = Literal["critical", "high", "low"]

# Weight of each priority in the overall score.
PRIORITY_WEIGHT: dict[str, int] = {"critical": 3, "high": 2, "low": 1}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    number: int
    label: str
    priority: Priority
    checks: str
    problem: str
    formula: str
    formula_explained: str
    frameworks: tuple[str, ...]  # LLM-judged: the engines that can score it; others: the library used
    llm_judged: bool
    needs_context: bool = False
    needs_reference: bool = False

    @property
    def gating(self) -> bool:
        """Critical and high metrics decide pass/fail; low ones are supporting text-comparison scores."""
        return self.priority != "low"

    def missing_inputs(self, sample) -> list[str]:
        missing = []
        if self.needs_context and not sample.retrieved_contexts:
            missing.append("retrieved_contexts")
        if self.needs_reference and not sample.reference_answer:
            missing.append("reference_answer")
        return missing


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="faithfulness", number=11, label="Faithfulness", priority="critical",
        checks="Is the answer supported by the retrieved context?",
        problem="Prevents hallucinated information.",
        formula="Supported Claims / Total Claims",
        formula_explained="Claims in the answer that the retrieved context supports, divided by all claims in the answer.",
        frameworks=("ragas", "deepeval"), llm_judged=True, needs_context=True,
    ),
    MetricSpec(
        key="answer_relevancy", number=12, label="Answer Relevancy", priority="critical",
        checks="Does the answer address the user's question?",
        problem="Prevents irrelevant answers.",
        formula="Relevant Question Aspects / Total Question Aspects",
        formula_explained=(
            "Parts of the question the answer addresses, divided by all parts of the question. RAGAS measures it by "
            "generating questions from the answer and comparing them with the real question (cosine similarity)."
        ),
        frameworks=("ragas", "deepeval"), llm_judged=True,
    ),
    MetricSpec(
        key="answer_correctness", number=13, label="Answer Correctness", priority="critical",
        checks="Is the answer factually correct compared with the reference answer?",
        problem="Detects incorrect facts and answers.",
        formula="Correct Facts / Total Facts",
        formula_explained=(
            "Facts in the answer that agree with the reference, divided by all facts. RAGAS combines a factual F1 "
            "(weight 0.75) with semantic similarity (weight 0.25)."
        ),
        frameworks=("ragas", "deepeval"), llm_judged=True, needs_reference=True,
    ),
    MetricSpec(
        key="answer_completeness", number=14, label="Answer Completeness", priority="high",
        checks="Does the answer include all the required information?",
        problem="Prevents missing important information.",
        formula="Required Details Provided / Total Required Details",
        formula_explained="Required details (from the reference or required_facts) found in the answer, divided by all required details.",
        frameworks=("deepeval", "custom"), llm_judged=True, needs_reference=True,
    ),
    MetricSpec(
        key="semantic_similarity", number=15, label="Semantic Similarity", priority="high",
        checks="Does the answer mean the same as the reference answer?",
        problem="Handles different wording with the same meaning.",
        formula="(A · B) / (‖A‖ × ‖B‖)",
        formula_explained="Cosine similarity of A = generated answer embedding and B = reference answer embedding.",
        frameworks=("sentence-transformers",), llm_judged=False, needs_reference=True,
    ),
    MetricSpec(
        key="exact_match", number=16, label="Exact Match (EM)", priority="low",
        checks="Does the answer exactly match the reference answer?",
        problem="Detects exact answer mismatches.",
        formula="EM = 1 if Generated = Reference, else 0",
        formula_explained="1 when the two texts are equal after normalization (lowercase, no punctuation, no a/an/the, single spaces), otherwise 0.",
        frameworks=("python",), llm_judged=False, needs_reference=True,
    ),
    MetricSpec(
        key="bleu", number=17, label="BLEU", priority="low",
        checks="How many of the answer's word sequences (n-grams) appear in the reference?",
        problem="Measures text-generation precision.",
        formula="BP × exp(Σ wₙ log pₙ)",
        formula_explained="BP = brevity penalty; pₙ = n-gram precision; wₙ = weight of each order (¼ each); N = 4.",
        frameworks=("sacrebleu",), llm_judged=False, needs_reference=True,
    ),
    MetricSpec(
        key="rouge", number=18, label="ROUGE", priority="low",
        checks="How much of the reference answer does the answer cover?",
        problem="Detects missing content in generated text.",
        formula="ROUGE-1 Recall = Overlapping Reference Unigrams / Total Reference Unigrams",
        formula_explained="Reference words also found in the answer, divided by all reference words. ROUGE-2 and ROUGE-L are reported too.",
        frameworks=("rouge-score",), llm_judged=False, needs_reference=True,
    ),
)

METRICS_BY_KEY: dict[str, MetricSpec] = {m.key: m for m in METRICS}
METRIC_KEYS: tuple[str, ...] = tuple(m.key for m in METRICS)


def production_tree() -> str:
    """The evaluation as a tree, in the same shape as the design."""
    lines = ["3. GENERATION EVALUATION", "   |"]
    for i, m in enumerate(METRICS, start=1):
        last = i == len(METRICS)
        branch, rail = ("   `-- ", "          ") if last else ("   |-- ", "   |      ")
        kind = "Library" if m.key in ("bleu", "rouge") else "Framework"
        engines = " / ".join(_display(f) for f in m.frameworks)
        lines += [
            f"{branch}{i}. {m.label}",
            f"{rail}|-- Purpose: {m.checks}",
            f"{rail}|-- Formula: {m.formula}",
            f"{rail}|-- {kind}: {engines}",
        ]
        if not last:
            lines.append("   |")
    lines += [
        "",
        "   |",
        "   v",
        "PRODUCTION EVALUATION INTEGRATION",
        "   |",
        "   |-- Evaluation Framework: RAGAS + DeepEval",
        "   |-- Custom Metrics: Python (Exact Match, custom completeness judge)",
        "   |-- Backend API: FastAPI",
        "   |-- Evaluation Tracking: Langfuse",
        "   |",
        "   v",
        "FINAL EVALUATION REPORT",
        "   |",
        "   |-- Individual Metric Scores",
        "   |-- Pass / Fail",
        "   `-- Overall Evaluation Results",
    ]
    return "\n".join(lines)


def _display(name: str) -> str:
    return {
        "ragas": "RAGAS", "deepeval": "DeepEval", "custom": "Custom LLM-as-a-Judge",
        "sentence-transformers": "Sentence Transformers", "python": "Python",
        "sacrebleu": "SacreBLEU", "rouge-score": "rouge-score",
    }.get(name, name)
