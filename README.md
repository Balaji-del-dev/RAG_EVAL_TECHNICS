# Retrieval Evaluation (Django)

A standalone Django app that scores a retriever with six retrieval metrics,
implemented in plain Python (no Ragas, DeepEval or other evaluation frameworks).

You upload a test dataset (questions + the chunk IDs that should come back),
collect your retriever's ranked chunk IDs, and the app compares them with the
ground truth, calculates the metrics, shows which queries fail and why, and
compares experiments. Generation quality is out of scope.

## Run it

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python manage.py migrate
venv/bin/python manage.py load_sample_data     # optional: three sample datasets
scripts/start.sh                               # http://127.0.0.1:8011/
scripts/stop.sh
venv/bin/python manage.py test retrieval_evaluation
```

`createsuperuser` gives you the Django admin at `/admin/`. The app pages have no
login; the server binds to 127.0.0.1 only.

## The six metrics

Worked example from `sample_data/hr_worked_example.json`. The query is "What is
the maternity leave policy?". Ground truth: C101 (grade 3), C102 (grade 2),
C103 (grade 1). The retriever's top 5: C104, **C101**, C105, **C102**, C106.

| Metric | Formula | Example |
|---|---|---|
| Precision@K | relevant in top K / K | 2 / 5 = **40%** |
| Recall@K | relevant in top K / all relevant | 2 / 3 = **66.7%** |
| F1 | 2PR / (P + R) | 2 x 0.4 x 0.667 / 1.067 = **50%** |
| Hit Rate@K | 1 if any relevant chunk in top K | **100%** |
| MRR | mean of 1 / rank of first relevant chunk | 1 / 2 = **0.50** |
| NDCG@K | DCG@K / ideal DCG@K | 5.709 / 9.393 = **0.608** |

NDCG: gain = 2^grade - 1, discount = log2(rank + 1).
DCG = 7/log2(3) + 3/log2(5) = 5.709. Ideal DCG = 7/1 + 3/log2(3) + 1/log2(4) = 9.393.

Conventions (all covered by tests):

* Precision, Recall, F1, Hit Rate and MRR use binary relevance: grade >= 1 is
  relevant. NDCG uses the grade (0-3). Grade 0 marks a known non-relevant chunk.
* Precision@K divides by K, even if the retriever returned fewer chunks.
* A chunk ID returned twice counts once.
* Run averages are macro averages: every query counts the same. MRR is the mean
  of the per-query reciprocal ranks.
* A query with no relevant chunk cannot be scored. It is reported as an error
  and left out of the averages.
* `RETRIEVAL_EVAL["NDCG_GAIN"] = "linear"` switches NDCG to gain = grade.
  The gain is stored on each run, so changing the setting does not change old runs.

Each run has an **Explain these scores** page (`/runs/<id>/explain/`). It walks
through the run step by step, using the run's own data:

1. **One question from the dataset**: its correct chunks, the top K the retriever
   returned, and the values the formulas need. `?q=<result id>` picks another question.
2. **All six metrics**: for each one, the question it answers, the formula, the
   calculation with real numbers, the result and what it means. Hit Rate and MRR
   use every question in the run.
3. **A summary table**: the example question next to the run averages, plus what
   the averages tell you, how the six metrics link, and rules that always hold.
4. **A question where every retrieved chunk was wrong**: all six scores are 0, what
   happens inside a RAG system, and what to check. If no question failed, it shows
   what would happen.
5. **How to fix it, step by step**: the fix steps, a before and after table (the
   "after" is computed for the best possible top K), and the failure points.
   With a chunk corpus, the app also checks the question itself: is the chunk there,
   does it share words with the question, and, for the built-in retrievers, how far
   down the full ranking it sits.

`sample_data/hr_leave_example.json` with `leave_example_results.json` gives the
standard textbook lesson. For the annual leave question it gives Precision@5 2/5 = 40%,
Recall@5 2/3 = 66.7%, F1 50% and NDCG@5 1.5/2.131 = 0.704. Over its three questions, Hit Rate is 2/3
and MRR is (1 + 0.5 + 0)/3 = 0.50. Its maternity question gets five wrong chunks.

Open any query in a run to see this calculation for that query. The page shows
the ranked chunks next to the ground truth, each formula with the numbers filled
in, and the DCG and ideal DCG tables.

## Where the ranked chunk IDs come from

The app scores retriever output; it never makes up retrieval results.

| Retriever | Use it for |
|---|---|
| Built-in BM25, TF-IDF, BM25 + TF-IDF fusion (RRF) | Experiments on a dataset that includes a chunk corpus. Pure Python, `services/builtin_retrievers.py`. |
| HTTP endpoint | Your real retriever (hybrid, dense, reranked). Each query is POSTed as `{"query", "query_id", "top_k"}`. The endpoint returns `["C1", ...]` or `{"results": [{"chunk_id", "score"}]}`. An Authorization header can be sent; it is never stored. |
| Uploaded results | Rankings exported from your pipeline: JSON `{"Q1": ["C104", "C101", ...]}` or CSV `query_id,chunk_id,rank,score`. |

## Dataset formats

JSON (the chunk corpus is optional; it is only needed for the built-in retrievers):

```json
{
  "name": "HR Policy Evaluation", "version": "1.0",
  "chunks": [{"chunk_id": "C101", "title": "Maternity eligibility", "text": "..."}],
  "queries": [{"query_id": "Q1", "text": "What is the maternity leave policy?",
               "ground_truth": [{"chunk_id": "C101", "grade": 3}, {"chunk_id": "C102", "grade": 2}]}]
}
```

CSV, one row per label: `query_id,query,chunk_id,grade`. See `sample_data/hr_policy_ground_truth.csv`.

## Failure analysis

| Issue | Detected by | Flagged when |
|---|---|---|
| No useful result | Hit Rate | nothing relevant in the top K (the query counts as **failed**) |
| Irrelevant chunks | Precision | under 50% of the best precision this query can reach at K |
| Missing chunks | Recall | under 80% of the best recall this query can reach at K |
| Relevant result ranked low | MRR | first relevant chunk below rank 2 |
| Poor relevance ordering | NDCG | NDCG under 0.70 |

Precision and recall are judged against what the query can reach at K. With 1
relevant chunk and K = 5, Precision@5 can never go above 20%, and that is not
the retriever's fault. Thresholds are in `config/settings.py`
(`RETRIEVAL_EVAL["FAILURE_THRESHOLDS"]`).

## Experiment comparison and regression testing

`/compare/` puts a baseline run next to a candidate run. Averages are recomputed
over the queries both runs evaluated. A metric whose average drops by more than
the tolerance (default 0.01) is a regression. Per query, the verdict follows
NDCG, and a query that loses all its relevant chunks is called out.

For CI:

```bash
venv/bin/python manage.py run_evaluation --dataset "HR Policy Evaluation" --retriever http \
    --url http://127.0.0.1:9000/retrieve --k 5 --name "candidate" --auth-env RETRIEVER_TOKEN
venv/bin/python manage.py compare_runs <baseline_id> <candidate_id> --fail-on-regression   # exit 1 on regression
```

Other commands: `load_dataset FILE [--name --dataset-version]`, `load_sample_data`,
and `run_evaluation --retriever precomputed --results FILE`.

## Layout

```
config/                         settings (RETRIEVAL_EVAL block), urls
retrieval_evaluation/
  models.py                     EvaluationDataset, KnowledgeChunk, EvaluationQuery, GroundTruth,
                                EvaluationRun, MetricResult, RetrievedChunk
  metrics/                      precision, recall, f1_score, hit_rate, mrr, ndcg, metrics_engine
  services/                     dataset_loader, retrieval_adapter, builtin_retrievers,
                                ground_truth_matcher, evaluation_runner
  analysis/                     failure_analysis, experiment_comparison, run_explanation, run_lesson
  views/                        dataset_views, evaluation_views, report_views
  templates/retrieval_evaluation/   Django templates (no JavaScript build step)
  static/retrieval_evaluation/      app.css, app.js
  management/commands/          load_dataset, load_sample_data, run_evaluation, compare_runs
  sample_data/                  sample datasets and the worked-example results file
  tests/                        90 tests
```

Each run stores the ranked chunk IDs per query (`RetrievedChunk`) and a snapshot
of the ground truth it was scored with (`MetricResult.expected`). Editing labels
or deleting a query later never changes an old run's scores. Evaluations run
synchronously in the request, which is fine for a few hundred queries. The
database is SQLite; change `DATABASES` to use PostgreSQL.

The HR policy sample text is invented for demonstration.
