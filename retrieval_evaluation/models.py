"""Database models for the retrieval evaluation system.

Relationships:

    EvaluationDataset 1:N EvaluationQuery 1:N GroundTruth
    EvaluationDataset 1:N KnowledgeChunk          (optional corpus, used by
                                                  the built-in retrievers)
    EvaluationDataset 1:N EvaluationRun 1:N MetricResult 1:N RetrievedChunk

A MetricResult keeps a snapshot of the query text and its ground truth at the
time of the run, so editing or deleting a query later never changes the scores
of an old run.
"""

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.urls import reverse

MAX_GRADE = 3


class EvaluationDataset(models.Model):
    """A versioned set of test queries with their ground truth."""

    name = models.CharField(max_length=200)
    version = models.CharField(max_length=50, default="1.0")
    description = models.TextField(blank=True)
    source_filename = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(fields=["name", "version"], name="unique_dataset_version"),
        ]

    def __str__(self):
        return f"{self.name} v{self.version}"

    def get_absolute_url(self):
        return reverse("retrieval_evaluation:dataset_detail", args=[self.pk])


class KnowledgeChunk(models.Model):
    """A chunk of the knowledge base. Optional: needed only by the built-in
    retrievers and to show chunk titles next to IDs."""

    dataset = models.ForeignKey(EvaluationDataset, on_delete=models.CASCADE, related_name="chunks")
    chunk_id = models.CharField(max_length=100)
    title = models.CharField(max_length=300, blank=True)
    text = models.TextField(blank=True)

    class Meta:
        ordering = ["chunk_id"]
        constraints = [
            models.UniqueConstraint(fields=["dataset", "chunk_id"], name="unique_chunk_per_dataset"),
        ]

    def __str__(self):
        return f"{self.chunk_id} {self.title}".strip()


class EvaluationQuery(models.Model):
    """One test question."""

    dataset = models.ForeignKey(EvaluationDataset, on_delete=models.CASCADE, related_name="queries")
    query_id = models.CharField(max_length=100, help_text="Stable ID used to match retriever output.")
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]
        verbose_name_plural = "evaluation queries"
        constraints = [
            models.UniqueConstraint(fields=["dataset", "query_id"], name="unique_query_per_dataset"),
        ]

    def __str__(self):
        return f"{self.query_id}: {self.text[:60]}"


class GroundTruth(models.Model):
    """An expected chunk for a query, with a graded relevance label.

    Grade 0 = labelled not relevant (a hard negative), 1 = somewhat relevant,
    2 = relevant, 3 = highly relevant. Binary metrics treat grade >= 1 as
    relevant; NDCG uses the grade itself.
    """

    query = models.ForeignKey(EvaluationQuery, on_delete=models.CASCADE, related_name="ground_truth")
    chunk_id = models.CharField(max_length=100)
    grade = models.PositiveSmallIntegerField(
        default=1, validators=[MinValueValidator(0), MaxValueValidator(MAX_GRADE)]
    )

    class Meta:
        ordering = ["-grade", "chunk_id"]
        verbose_name_plural = "ground truth labels"
        constraints = [
            models.UniqueConstraint(fields=["query", "chunk_id"], name="unique_label_per_query"),
            models.CheckConstraint(
                condition=models.Q(grade__gte=0, grade__lte=MAX_GRADE), name="grade_in_range"
            ),
        ]

    def __str__(self):
        return f"{self.chunk_id} (grade {self.grade})"

    @property
    def is_relevant(self):
        return self.grade >= 1


class EvaluationRun(models.Model):
    """One experiment: a dataset evaluated with one retriever configuration at one K."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    class Retriever(models.TextChoices):
        BM25 = "bm25", "Built-in BM25"
        TFIDF = "tfidf", "Built-in TF-IDF (cosine)"
        FUSION = "fusion", "Built-in BM25 + TF-IDF fusion (RRF)"
        HTTP = "http", "HTTP retriever endpoint"
        PRECOMPUTED = "precomputed", "Uploaded retriever results"

    dataset = models.ForeignKey(EvaluationDataset, on_delete=models.CASCADE, related_name="runs")
    name = models.CharField(max_length=200)
    notes = models.TextField(blank=True)
    top_k = models.PositiveSmallIntegerField(validators=[MinValueValidator(1), MaxValueValidator(100)])
    retriever_type = models.CharField(max_length=20, choices=Retriever.choices)
    # Parameters of the retriever (BM25 k1/b, URL, ...). For uploaded results it
    # also holds the submitted rankings so the run can be reproduced.
    retriever_config = models.JSONField(default=dict, blank=True)
    # NDCG gain used for this run ('exponential' = 2^grade - 1, or 'linear').
    ndcg_gain = models.CharField(max_length=20, default="exponential")

    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    error_message = models.TextField(blank=True)

    query_count = models.PositiveIntegerField(default=0)
    evaluated_count = models.PositiveIntegerField(default=0)
    error_count = models.PositiveIntegerField(default=0)

    # Averages over the evaluated queries (errored queries are excluded).
    avg_precision = models.FloatField(null=True, blank=True)
    avg_recall = models.FloatField(null=True, blank=True)
    avg_f1 = models.FloatField(null=True, blank=True)
    avg_hit_rate = models.FloatField(null=True, blank=True)
    mrr = models.FloatField(null=True, blank=True)
    avg_ndcg = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    duration_ms = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self):
        return f"#{self.pk} {self.name}"

    def get_absolute_url(self):
        return reverse("retrieval_evaluation:run_detail", args=[self.pk])

    @property
    def is_completed(self):
        return self.status == self.Status.COMPLETED

    def averages(self):
        """The six run-level scores, keyed by metric name."""
        return {
            "precision": self.avg_precision,
            "recall": self.avg_recall,
            "f1": self.avg_f1,
            "hit_rate": self.avg_hit_rate,
            "mrr": self.mrr,
            "ndcg": self.avg_ndcg,
        }

    def retriever_summary(self):
        """Readable retriever parameters, without the uploaded rankings."""
        cfg = self.retriever_config or {}
        if self.retriever_type == self.Retriever.PRECOMPUTED:
            name = cfg.get("source_filename") or "uploaded file"
            return f"{name}, {len(cfg.get('results', {}))} queries"
        if self.retriever_type == self.Retriever.HTTP:
            return cfg.get("url", "")
        parts = []
        if self.retriever_type in (self.Retriever.BM25, self.Retriever.FUSION):
            parts.append(f"k1={cfg.get('k1', 1.5)}, b={cfg.get('b', 0.75)}")
        if self.retriever_type == self.Retriever.FUSION:
            parts.append(f"rrf_k={cfg.get('rrf_k', 60)}")
        parts.append("stemming on" if cfg.get("stemming") else "stemming off")
        return ", ".join(parts)


class MetricResult(models.Model):
    """The six metric scores of one query in one run."""

    run = models.ForeignKey(EvaluationRun, on_delete=models.CASCADE, related_name="results")
    query = models.ForeignKey(
        EvaluationQuery, on_delete=models.SET_NULL, null=True, blank=True, related_name="metric_results"
    )
    query_external_id = models.CharField(max_length=100)
    query_text = models.TextField()

    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1 = models.FloatField(null=True, blank=True)
    hit_rate = models.FloatField(null=True, blank=True)
    reciprocal_rank = models.FloatField(null=True, blank=True)
    ndcg = models.FloatField(null=True, blank=True)

    total_relevant = models.PositiveIntegerField(default=0)
    relevant_found = models.PositiveIntegerField(default=0)
    irrelevant_found = models.PositiveIntegerField(default=0)
    relevant_missed = models.PositiveIntegerField(default=0)
    first_relevant_rank = models.PositiveIntegerField(null=True, blank=True)
    retrieved_count = models.PositiveIntegerField(default=0)

    # Snapshot of the ground truth used for scoring: {chunk_id: grade}.
    expected = models.JSONField(default=dict)
    missed_chunk_ids = models.JSONField(default=list)
    # Failure-analysis issue codes, e.g. ["missing_chunks", "poor_ordering"].
    issues = models.JSONField(default=list)
    warnings = models.JSONField(default=list)

    latency_ms = models.FloatField(null=True, blank=True)
    error = models.TextField(blank=True)

    class Meta:
        ordering = ["id"]
        constraints = [
            models.UniqueConstraint(fields=["run", "query_external_id"], name="unique_result_per_query"),
        ]

    def __str__(self):
        return f"run {self.run_id} / {self.query_external_id}"

    def get_absolute_url(self):
        return reverse("retrieval_evaluation:result_detail", args=[self.run_id, self.pk])

    @property
    def has_error(self):
        return bool(self.error)

    def scores(self):
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "hit_rate": self.hit_rate,
            "mrr": self.reciprocal_rank,
            "ndcg": self.ndcg,
        }


class RetrievedChunk(models.Model):
    """One ranked chunk the retriever returned for a query in a run."""

    result = models.ForeignKey(MetricResult, on_delete=models.CASCADE, related_name="retrieved")
    rank = models.PositiveIntegerField()
    chunk_id = models.CharField(max_length=100)
    score = models.FloatField(null=True, blank=True)
    grade = models.PositiveSmallIntegerField(default=0)
    is_relevant = models.BooleanField(default=False)
    within_k = models.BooleanField(default=True)

    class Meta:
        ordering = ["rank"]
        constraints = [
            models.UniqueConstraint(fields=["result", "rank"], name="unique_rank_per_result"),
        ]

    def __str__(self):
        return f"#{self.rank} {self.chunk_id}"
