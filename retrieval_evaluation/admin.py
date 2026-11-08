from django.contrib import admin

from .models import EvaluationDataset, EvaluationQuery, EvaluationRun, GroundTruth, KnowledgeChunk, MetricResult, RetrievedChunk


class GroundTruthInline(admin.TabularInline):
    model = GroundTruth
    extra = 1


@admin.register(EvaluationDataset)
class EvaluationDatasetAdmin(admin.ModelAdmin):
    list_display = ("name", "version", "created_at")
    search_fields = ("name", "description")


@admin.register(KnowledgeChunk)
class KnowledgeChunkAdmin(admin.ModelAdmin):
    list_display = ("chunk_id", "title", "dataset")
    list_filter = ("dataset",)
    search_fields = ("chunk_id", "title", "text")


@admin.register(EvaluationQuery)
class EvaluationQueryAdmin(admin.ModelAdmin):
    list_display = ("query_id", "text", "dataset")
    list_filter = ("dataset",)
    search_fields = ("query_id", "text")
    inlines = [GroundTruthInline]


@admin.register(EvaluationRun)
class EvaluationRunAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "dataset", "retriever_type", "top_k", "status", "avg_ndcg", "created_at")
    list_filter = ("status", "retriever_type", "dataset")
    readonly_fields = ("created_at", "started_at", "finished_at")


class RetrievedChunkInline(admin.TabularInline):
    model = RetrievedChunk
    extra = 0
    readonly_fields = ("rank", "chunk_id", "score", "grade", "is_relevant", "within_k")
    can_delete = False


@admin.register(MetricResult)
class MetricResultAdmin(admin.ModelAdmin):
    list_display = ("query_external_id", "run", "precision", "recall", "f1", "hit_rate", "reciprocal_rank", "ndcg")
    list_filter = ("run",)
    inlines = [RetrievedChunkInline]
