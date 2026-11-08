"""The web UI: server-rendered pages (Jinja2) on top of the same evaluator as the JSON API.

    GET  /            Evaluate: one answer, all 8 scores with their formulas and working
    POST /            run the evaluation from the form
    GET  /compare     Compare: the 5 sample answers side by side
    POST /compare     run the comparison
    GET  /formulas    Metrics & formulas: the tree and the consolidated table
"""

import json
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from generation_eval.catalog import METRICS, METRICS_BY_KEY, production_tree
from generation_eval.config import get_settings
from generation_eval.evaluation import GenerationEvaluator
from generation_eval.schemas import EvaluationSample

from .system import LLM_JUDGE_MEMORY_MB, available_memory_mb, ollama_status

APP_DIR = Path(__file__).resolve().parent
SAMPLES_FILE = APP_DIR.parents[2] / "data" / "hr_leave_samples.json"

templates = Jinja2Templates(directory=APP_DIR / "templates")
templates.env.globals["retrieval_url"] = get_settings().retrieval_app_url
router = APIRouter(include_in_schema=False)

FRAMEWORK_NAMES = {
    "ragas": "RAGAS", "deepeval": "DeepEval", "custom": "Custom LLM judge",
    "sentence-transformers": "Sentence Transformers", "python": "Python",
    "sacrebleu": "SacreBLEU", "rouge-score": "rouge-score",
}
FAST_METRICS = [m.key for m in METRICS if not m.llm_judged]
LLM_METRICS = [m.key for m in METRICS if m.llm_judged]


def load_examples() -> list[dict]:
    return json.loads(SAMPLES_FILE.read_text(encoding="utf-8"))["samples"]


def _evaluator(request: Request) -> GenerationEvaluator:
    return request.app.state.evaluator


async def _machine(evaluator: GenerationEvaluator) -> dict:
    free_mb = available_memory_mb()
    return {
        "free_mb": free_mb,
        "needed_mb": LLM_JUDGE_MEMORY_MB,
        "llm_ok": free_mb is None or free_mb >= LLM_JUDGE_MEMORY_MB,
        "ollama": await ollama_status(evaluator.settings),
        "judge_model": evaluator.settings.judge_model,
    }


def _metric_options(evaluator: GenerationEvaluator, selected: list[str], engines: dict[str, str]) -> dict:
    s = evaluator.settings

    def option(m):
        return {
            "spec": m,
            "checked": m.key in selected,
            "engines": [(e, FRAMEWORK_NAMES[e]) for e in m.frameworks],
            "engine": engines.get(m.key) or (s.default_engine(m.key) if m.llm_judged else m.frameworks[0]),
            "threshold": s.threshold_for(m.key),
        }

    return {"llm": [option(m) for m in METRICS if m.llm_judged], "fast": [option(m) for m in METRICS if not m.llm_judged]}


def _read_choices(form) -> tuple[list[str], dict[str, str]]:
    metrics = [m for m in form.getlist("metrics") if m in METRICS_BY_KEY]
    engines = {}
    for key in LLM_METRICS:
        engine = form.get(f"engine_{key}")
        if engine in METRICS_BY_KEY[key].frameworks:
            engines[key] = engine
    return metrics, engines


def _lines(text: str) -> list[str]:
    return [line.strip(" -•\t") for line in text.splitlines() if line.strip(" -•\t")]


def _blocks(text: str) -> list[str]:
    """Context chunks are separated by a blank line."""
    return [" ".join(block.split()) for block in text.replace("\r\n", "\n").split("\n\n") if block.strip()]


def _form_values(sample: dict) -> dict:
    return {
        "question": sample.get("question", ""),
        "contexts": "\n\n".join(sample.get("retrieved_contexts") or []),
        "reference": sample.get("reference_answer") or "",
        "facts": "\n".join(sample.get("required_facts") or []),
        "answer": sample.get("generated_answer", ""),
        "case": (sample.get("metadata") or {}).get("case", ""),
        "example_id": sample.get("sample_id", ""),
    }


def _view_scores(report) -> list[dict]:
    rows = []
    for s in report.scores:
        if s.status == "scored":
            result = ("pass", "Pass") if s.passed else ("failed", "Fail")
        elif s.status == "skipped":
            result = ("pending", "Skipped")
        else:
            result = ("error", "Error")
        details = dict(s.details)
        facts = details.pop("required_details", None)
        rows.append({
            "s": s,
            "result_code": result[0],
            "result_label": result[1],
            "framework": FRAMEWORK_NAMES.get(s.framework, s.framework),
            "meter": round((s.score or 0) * 100, 1),
            "threshold_pct": round(s.threshold * 100, 1),
            "facts": facts,
            "details_json": json.dumps(details, indent=2, ensure_ascii=False, default=str) if details else "",
        })
    return rows


# ---------------------------------------------------------------------- Evaluate
@router.get("/", response_class=HTMLResponse)
async def evaluate_page(request: Request, example: str | None = None):
    evaluator = _evaluator(request)
    examples = load_examples()
    chosen = next((e for e in examples if e["sample_id"] == example), examples[0])
    machine = await _machine(evaluator)
    selected = FAST_METRICS + (LLM_METRICS if machine["llm_ok"] and machine["ollama"]["reachable"] else [])
    return templates.TemplateResponse(request, "evaluate.html", {
        "page": "evaluate",
        "examples": examples,
        "values": _form_values(chosen),
        "options": _metric_options(evaluator, selected, {}),
        "machine": machine,
        "track": evaluator.tracker.enabled,
        "tracking_available": evaluator.tracker.enabled,
    })


@router.post("/", response_class=HTMLResponse)
async def evaluate_run(request: Request):
    evaluator = _evaluator(request)
    form = await request.form()
    metrics, engines = _read_choices(form)
    values = {k: str(form.get(k, "")) for k in ("question", "contexts", "reference", "facts", "answer", "case", "example_id")}
    error, report = None, None
    if not metrics:
        error = "Tick at least one metric."
    else:
        try:
            sample = EvaluationSample(
                sample_id=f"ui-{values['example_id'] or uuid4().hex[:8]}",
                question=values["question"].strip(),
                generated_answer=values["answer"].strip(),
                retrieved_contexts=_blocks(values["contexts"]),
                reference_answer=values["reference"].strip() or None,
                required_facts=_lines(values["facts"]) or None,
                metadata={"source": "ui", "case": values["case"]} if values["case"] else {"source": "ui"},
            )
            report = await evaluator.evaluate(sample, metrics, engines, track=form.get("track") == "on")
        except ValidationError as exc:
            error = "Please fill in: " + ", ".join(
                {"question": "Question", "generated_answer": "Generated answer"}.get(str(e["loc"][0]), str(e["loc"][0]))
                for e in exc.errors())
        except ValueError as exc:
            error = str(exc)
    return templates.TemplateResponse(request, "evaluate.html", {
        "page": "evaluate",
        "examples": load_examples(),
        "values": values,
        "options": _metric_options(evaluator, metrics, engines),
        "machine": await _machine(evaluator),
        "track": form.get("track") == "on",
        "tracking_available": evaluator.tracker.enabled,
        "error": error,
        "report": report,
        "rows": _view_scores(report) if report else [],
        "not_checked": [m.label for m in METRICS if m.gating and m.key not in metrics],
    }, status_code=400 if error else 200)


# ----------------------------------------------------------------------- Compare
@router.get("/compare", response_class=HTMLResponse)
async def compare_page(request: Request):
    evaluator = _evaluator(request)
    return templates.TemplateResponse(request, "compare.html", {
        "page": "compare",
        "examples": load_examples(),
        "options": _metric_options(evaluator, FAST_METRICS, {}),
        "machine": await _machine(evaluator),
    })


@router.post("/compare", response_class=HTMLResponse)
async def compare_run(request: Request):
    evaluator = _evaluator(request)
    form = await request.form()
    metrics, engines = _read_choices(form)
    examples = load_examples()
    batch, error = None, None
    if not metrics:
        error = "Tick at least one metric."
    else:
        samples = [EvaluationSample(**{**e, "sample_id": f"compare-{e['sample_id']}"}) for e in examples]
        batch = await evaluator.evaluate_batch(samples, metrics, engines, track=False)
    columns = [m for m in METRICS if m.key in metrics]
    matrix = []
    if batch:
        for example, report in zip(examples, batch.reports):
            cells = {s.metric: s for s in report.scores}
            matrix.append({"example": example, "report": report, "cells": [cells[c.key] for c in columns]})
    return templates.TemplateResponse(request, "compare.html", {
        "page": "compare",
        "examples": examples,
        "options": _metric_options(evaluator, metrics, engines),
        "machine": await _machine(evaluator),
        "error": error,
        "batch": batch,
        "columns": columns,
        "matrix": matrix,
        "thresholds": {m.key: evaluator.settings.threshold_for(m.key) for m in METRICS},
        "llm_ran": any(m in LLM_METRICS for m in metrics),
    }, status_code=400 if error else 200)


# ---------------------------------------------------------------------- Formulas
@router.get("/formulas", response_class=HTMLResponse)
async def formulas_page(request: Request):
    evaluator = _evaluator(request)
    s = evaluator.settings
    rows = [{
        "m": m,
        "frameworks": " / ".join(FRAMEWORK_NAMES[f] for f in m.frameworks),
        "default": FRAMEWORK_NAMES[s.default_engine(m.key) if m.llm_judged else m.frameworks[0]],
        "threshold": s.threshold_for(m.key),
    } for m in METRICS]
    return templates.TemplateResponse(request, "formulas.html", {"page": "formulas", "rows": rows, "tree": production_tree()})
