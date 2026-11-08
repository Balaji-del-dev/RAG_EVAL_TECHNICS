# Generation Evaluation

Checks whether an LLM's answer in a RAG system is **grounded, relevant, correct and
complete**, and how close it is to a reference answer. It uses eight metrics, scored
with the frameworks used in production, served by FastAPI and tracked in Langfuse.

```
3. GENERATION EVALUATION
   |
   |-- 1. Faithfulness         Supported Claims / Total Claims                      RAGAS / DeepEval
   |-- 2. Answer Relevancy     Relevant Question Aspects / Total Question Aspects   RAGAS / DeepEval
   |-- 3. Answer Correctness   Correct Facts / Total Facts                          RAGAS / DeepEval
   |-- 4. Answer Completeness  Required Details Provided / Total Required Details   DeepEval / Custom LLM-as-a-Judge
   |-- 5. Semantic Similarity  (A · B) / (‖A‖ × ‖B‖)                                Sentence Transformers
   |-- 6. Exact Match (EM)     1 if Generated = Reference, else 0                   Python
   |-- 7. BLEU                 BP × exp(Σ wₙ log pₙ)                                SacreBLEU
   `-- 8. ROUGE                Overlapping Ref. Unigrams / Total Ref. Unigrams      rouge-score
   |
   v
PRODUCTION EVALUATION INTEGRATION
   |-- Evaluation Framework: RAGAS + DeepEval
   |-- Custom Metrics: Python
   |-- Backend API: FastAPI
   `-- Evaluation Tracking: Langfuse
   |
   v
FINAL EVALUATION REPORT: individual metric scores, pass / fail, overall score
```

`uv run generation-eval tree` prints the full tree. Every metric's formula, priority
and frameworks live in one place: `src/generation_eval/catalog.py`.

## The eight metrics

| No. | Metric | Priority | What problem it solves | Formula | Framework (default first) | Pass at |
|---|---|---|---|---|---|---|
| 11 | Faithfulness | Critical | Hallucinated information | Supported claims / total claims | RAGAS, DeepEval | ≥ 0.8 |
| 12 | Answer Relevancy | Critical | Answers that miss the question | Relevant aspects / total aspects | RAGAS, DeepEval | ≥ 0.7 |
| 13 | Answer Correctness | Critical | Wrong facts | Correct facts / total facts | RAGAS, DeepEval | ≥ 0.7 |
| 14 | Answer Completeness | High | Missing details | Required details provided / required | DeepEval, custom judge | ≥ 0.8 |
| 15 | Semantic Similarity | High | Same meaning, different words | (A · B) / (‖A‖ × ‖B‖) | Sentence Transformers | ≥ 0.8 |
| 16 | Exact Match | Low | Exact mismatches | 1 = match, 0 = no match | Python | = 1 |
| 17 | BLEU | Low | Generated n-gram precision | BP × exp(Σ wₙ log pₙ) | SacreBLEU | ≥ 0.3 |
| 18 | ROUGE | Low | Reference content not covered | ROUGE-1 recall | rouge-score | ≥ 0.5 |

The ratios for metrics 11–14 are the simple version of the idea. The frameworks compute them with an LLM judge:

* **RAGAS Faithfulness** splits the answer into claims and asks the judge which ones the context supports.
* **RAGAS Answer Relevancy** generates questions from the answer and measures their cosine similarity to the real question.
* **RAGAS Answer Correctness** = 0.75 × factual F1 + 0.25 × semantic similarity.
* **DeepEval** Faithfulness and Answer Relevancy count the judge's yes/no verdicts per claim or statement. Correctness and Completeness use **G-Eval** with written evaluation steps.
* **Custom judge** (completeness) returns one JSON verdict per required detail, so the score is exactly provided / required.

Every score comes with its working, for example
`5 overlapping reference unigrams / 11 total reference unigrams = 0.455` or
`(A · B) / (‖A‖ × ‖B‖) = 0.878 / (1.000 × 1.000) = 0.878`.

## Pass / fail

* A metric passes when its score is at least its threshold (set in `.env`).
* **Critical and high metrics decide** the result. Exact Match, BLEU and ROUGE are reported,
  but they never fail an answer on their own: they are supporting text-comparison scores,
  not factual checks.
* Each answer gets one status:
  * **fail**: a deciding metric scored below its threshold.
  * **incomplete**: a deciding metric could not be scored, because of a judge error or a missing input such as `reference_answer`.
  * **pass**: every deciding metric scored and passed.
* **Overall score** is the weighted mean of the scored metrics, with weights critical 3, high 2 and low 1.

Why the decision rests on the critical metrics: with real embeddings, the hallucinated answer
"Employees receive 20 casual leaves and 15 sick leaves per year" scores **0.955** semantic
similarity, *higher* than a correct paraphrase (0.817). Similarity and word overlap cannot
spot a wrong number. Faithfulness and correctness can.

## Run it

```bash
uv sync                                   # Python 3.12, CPU-only torch
cp .env.example .env                      # judge model, thresholds, Langfuse keys
ollama pull qwen2.5:1.5b && ollama pull all-minilm

uv run generation-eval tree               # the tree above
uv run generation-eval metrics            # metric table with defaults and thresholds
uv run generation-eval run data/hr_leave_samples.json --no-track
uv run generation-eval run data/hr_leave_samples.json --metrics exact_match,bleu,rouge
uv run generation-eval run data/hr_leave_samples.json --engine faithfulness=deepeval --engine answer_completeness=custom

scripts/start.sh                          # UI on http://127.0.0.1:8013/, API docs on /docs
scripts/stop.sh
```

### Web UI

| Page | What it does |
|---|---|
| http://127.0.0.1:8013/ | **Evaluate**: load one of the five examples or type your own question, context, reference and answer. Pick metrics and the framework for each LLM-judged metric. You get a pass/fail verdict, eight score tiles (bar = score, tick = threshold) and a table with each formula and its working. |
| http://127.0.0.1:8013/compare | **Compare answers**: the five example answers side by side, to see which metric catches which problem. |
| http://127.0.0.1:8013/formulas | **Metrics & formulas**: the consolidated table, "don't confuse these metrics", production priorities and the tree. |

The UI shows free memory. When less than the judge needs (about 1.8 GB) is free, it leaves the
LLM-judged metrics unticked. When a deciding metric did not run, the verdict says so.

`data/hr_leave_samples.json` has one HR question with five answers. Each answer is one case:

1. correct and complete;
2. paraphrased;
3. correct but incomplete;
4. hallucinated (wrong numbers);
5. irrelevant.

### API

| Method | Path | |
|---|---|---|
| GET | `/health` | Ollama reachable, judge model installed, Langfuse keys valid |
| GET | `/metrics` | The eight metrics: formula, priority, frameworks, threshold, required inputs |
| GET | `/tree` | The tree as text |
| POST | `/evaluate` | One sample, returns an `EvaluationReport` |
| POST | `/evaluate/batch` | Up to 200 samples, returns a `BatchReport` with averages and pass rate |

```bash
curl -s -X POST http://127.0.0.1:8013/evaluate -H 'Content-Type: application/json' -d '{
  "sample": {
    "question": "How many casual leaves and sick leaves do employees receive per year?",
    "retrieved_contexts": ["Employees receive 12 casual leaves per year. Employees receive 12 sick leaves per year."],
    "reference_answer": "Employees receive 12 casual leaves and 12 sick leaves per year.",
    "required_facts": ["Casual leave is 12 days per year", "Sick leave is 12 days per year"],
    "generated_answer": "Employees receive 12 casual leaves."
  },
  "metrics": ["faithfulness", "answer_completeness", "rouge"],
  "engines": {"answer_completeness": "custom"}
}'
```

Inputs: `question` and `generated_answer` are always needed. Faithfulness also needs
`retrieved_contexts`. Correctness, similarity, EM, BLEU and ROUGE need `reference_answer`.
Completeness needs `reference_answer` or `required_facts`. A metric without its inputs is
reported as `skipped`, not as a failure.

### Langfuse

Each evaluated sample becomes one trace with an `evaluator` observation. It records the
question, answer, contexts and reference, and the result as output. The trace gets one score
per metric, with the working as the comment, plus `overall_score` and `passed`. To switch it
on, create a project at http://localhost:3000, open Settings > API Keys, and put
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`. A Langfuse outage is reported in
the response (`"tracking": "error: …"`) and never loses the scores.

## Project layout

```
src/generation_eval/
  catalog.py            the 8 metrics: formula, priority, frameworks (single source)
  config.py             settings from .env (judge model, engines, thresholds, Langfuse)
  schemas.py            EvaluationSample, MetricScore, EvaluationReport, BatchReport
  metrics/              exact_match (Python), bleu (SacreBLEU), rouge (rouge-score),
                        semantic_similarity (Sentence Transformers)
  judges/               ragas_judge, deepeval_judge, custom_judge (LLM-as-a-judge, Ollama)
  evaluation/           evaluator (runs metrics), scoring (pass/fail, overall score)
  tracking/             langfuse_tracker
  cli.py                generation-eval tree | metrics | run
services/fastapi_service/app/main.py   the JSON API
services/fastapi_service/app/ui.py     the web UI (Jinja2 templates, static CSS/JS)
data/hr_leave_samples.json             sample dataset
tests/                                 56 tests, plus model and live tests behind flags
```

## Tests

```bash
uv run pytest                                              # fast: fake judges, no Ollama
RUN_MODEL_TESTS=1 uv run pytest -k real_model              # real Sentence Transformers model
RUN_LIVE=1 uv run pytest tests/test_live.py -s             # real RAGAS + DeepEval judges via Ollama
```

## Notes

* **Memory:** the judge model needs about 1.1 GB and the embedding model about 0.5 GB. On a
  small machine, check `free -m` before a live run. `scripts/start.sh` marks the API as the
  first process the OOM killer stops. Samples are evaluated one at a time
  (`MAX_CONCURRENT_EVALUATIONS=1`).
* **Small judges:** a 1.5B judge sometimes writes output the framework cannot parse. That
  metric is then reported as `error`, and the rest still score. Use a larger `JUDGE_MODEL`
  (for example `qwen2.5:3b`) when memory allows.
* **ROUGE example:** "Employees receive 12 casual leaves" against the 11-word reference
  matches 5 words (employees, receive, 12, casual, leaves), so ROUGE-1 recall is
  5/11 = 0.455, not 6/11.
* **Local only:** RAGAS and DeepEval telemetry is switched off.
