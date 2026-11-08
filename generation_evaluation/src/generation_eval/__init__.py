"""Generation evaluation for RAG answers.

Eight metrics check whether the LLM's answer is grounded, relevant, correct and
complete, and how close it is to a reference answer:

    11 Faithfulness          RAGAS / DeepEval
    12 Answer Relevancy      RAGAS / DeepEval
    13 Answer Correctness    RAGAS / DeepEval
    14 Answer Completeness   DeepEval / custom LLM-as-a-judge
    15 Semantic Similarity   Sentence Transformers
    16 Exact Match           Python
    17 BLEU                  SacreBLEU
    18 ROUGE                 rouge-score

Scores are served by a FastAPI service and tracked in Langfuse.
"""

import os

# Local-first: the evaluation frameworks must not send questions or answers anywhere.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .catalog import METRICS, METRICS_BY_KEY, MetricSpec  # noqa: E402
from .config import Settings, get_settings  # noqa: E402
from .evaluation.evaluator import GenerationEvaluator  # noqa: E402
from .schemas import BatchReport, EvaluationReport, EvaluationSample, MetricScore  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "METRICS",
    "METRICS_BY_KEY",
    "BatchReport",
    "EvaluationReport",
    "EvaluationSample",
    "GenerationEvaluator",
    "MetricScore",
    "MetricSpec",
    "Settings",
    "get_settings",
    "main",
]


def main() -> int:
    """Entry point of the `generation-eval` command."""
    from .cli import main as cli_main

    return cli_main()
