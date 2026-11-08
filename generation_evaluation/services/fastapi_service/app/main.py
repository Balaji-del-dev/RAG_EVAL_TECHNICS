"""Generation Evaluation API (FastAPI) and its web UI.

Web UI (HTML, see ui.py):  /  (evaluate)   /compare   /formulas
JSON API:
    GET  /health            judge model, Ollama and Langfuse status
    GET  /metrics           the eight metrics: formula, priority, frameworks, threshold
    GET  /tree              the production tree as text
    POST /evaluate          evaluate one sample  -> EvaluationReport
    POST /evaluate/batch    evaluate many        -> BatchReport

Run:  scripts/start.sh   (uvicorn app.main:app --app-dir services/fastapi_service --port 8013)
UI:   http://127.0.0.1:8013/        API docs: http://127.0.0.1:8013/docs
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from generation_eval import __version__
from generation_eval.catalog import METRICS, production_tree
from generation_eval.config import Settings, get_settings
from generation_eval.evaluation import GenerationEvaluator
from generation_eval.schemas import BatchEvaluateRequest, BatchReport, EvaluateRequest, EvaluationReport

from . import ui
from .system import available_memory_mb, ollama_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "evaluator"):  # tests set their own
        app.state.evaluator = GenerationEvaluator(get_settings())
    yield


app = FastAPI(
    title="Generation Evaluation API",
    version=__version__,
    description="Faithfulness, Answer Relevancy, Answer Correctness, Answer Completeness, Semantic Similarity, "
                "Exact Match, BLEU and ROUGE for RAG answers. RAGAS + DeepEval + Python, tracked in Langfuse.",
    lifespan=lifespan,
)


app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")
app.include_router(ui.router)


def get_evaluator(request: Request) -> GenerationEvaluator:
    return request.app.state.evaluator


@app.get("/health")
async def health(evaluator: GenerationEvaluator = Depends(get_evaluator)) -> dict:
    s: Settings = evaluator.settings
    ollama = await ollama_status(s)
    tracking = {"enabled": evaluator.tracker.enabled, "host": s.langfuse_host}
    if evaluator.tracker.enabled:
        try:
            tracking["keys_valid"] = await asyncio.wait_for(asyncio.to_thread(evaluator.tracker.check), timeout=5)
        except Exception as exc:  # noqa: BLE001 - report, don't fail the health check
            tracking["error"] = type(exc).__name__
    return {
        "status": "ok" if ollama["reachable"] else "degraded",
        "version": __version__,
        "judge_model": s.judge_model,
        "sentence_model": s.sentence_model,
        "ollama": ollama,
        "free_memory_mb": available_memory_mb(),
        "tracking": tracking,
    }


@app.get("/metrics")
def metrics(evaluator: GenerationEvaluator = Depends(get_evaluator)) -> list[dict]:
    s = evaluator.settings
    return [
        {
            "number": m.number,
            "metric": m.key,
            "label": m.label,
            "priority": m.priority,
            "decides_pass_fail": m.gating,
            "checks": m.checks,
            "problem_solved": m.problem,
            "formula": m.formula,
            "formula_explained": m.formula_explained,
            "frameworks": list(m.frameworks),
            "default_framework": s.default_engine(m.key) if m.llm_judged else m.frameworks[0],
            "threshold": s.threshold_for(m.key),
            "needs": ["question", "generated_answer"]
            + (["retrieved_contexts"] if m.needs_context else [])
            + (["reference_answer"] if m.needs_reference else []),
        }
        for m in METRICS
    ]


@app.get("/tree", response_class=PlainTextResponse)
def tree() -> str:
    return production_tree()


@app.post("/evaluate", response_model=EvaluationReport)
async def evaluate(body: EvaluateRequest, evaluator: GenerationEvaluator = Depends(get_evaluator)) -> EvaluationReport:
    try:
        return await evaluator.evaluate(body.sample, body.metrics, body.engines, body.track)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/evaluate/batch", response_model=BatchReport)
async def evaluate_batch(body: BatchEvaluateRequest, evaluator: GenerationEvaluator = Depends(get_evaluator)) -> BatchReport:
    try:
        return await evaluator.evaluate_batch(body.samples, body.metrics, body.engines, body.track)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
