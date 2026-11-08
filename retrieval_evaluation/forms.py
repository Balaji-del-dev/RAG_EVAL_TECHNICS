from django import forms
from django.conf import settings

from .models import MAX_GRADE, EvaluationDataset, EvaluationRun
from .services.dataset_loader import DatasetFormatError, parse_dataset_file
from .services.retrieval_adapter import ResultsFormatError, RetrieverError, parse_results_file, validate_http_url

GRADE_CHOICES = [
    (3, "3 - highly relevant"),
    (2, "2 - relevant"),
    (1, "1 - somewhat relevant"),
    (0, "0 - not relevant (hard negative)"),
]


def _read_upload(upload) -> bytes:
    limit = settings.RETRIEVAL_EVAL["MAX_UPLOAD_BYTES"]
    if upload.size > limit:
        raise forms.ValidationError(f"The file is larger than {limit // (1024 * 1024)} MB.")
    return upload.read()


class DatasetUploadForm(forms.Form):
    file = forms.FileField(
        help_text="JSON (queries, ground truth and an optional chunk corpus) or CSV (query_id, query, chunk_id, grade).",
        widget=forms.ClearableFileInput(attrs={"accept": ".json,.csv"}),
    )
    name = forms.CharField(max_length=200, required=False, help_text="Required for CSV. Overrides the name in a JSON file.")
    version = forms.CharField(max_length=50, required=False, help_text="Defaults to the file's version, or 1.0.")
    description = forms.CharField(required=False, widget=forms.Textarea(attrs={"rows": 2}))

    def clean(self):
        cleaned = super().clean()
        upload = cleaned.get("file")
        if upload:
            try:
                cleaned["parsed"] = parse_dataset_file(upload.name, _read_upload(upload))
            except DatasetFormatError as exc:
                for error in exc.errors:
                    self.add_error("file", error)
        return cleaned


class QueryForm(forms.Form):
    query_id = forms.CharField(max_length=100, label="Query ID")
    text = forms.CharField(label="Question", widget=forms.Textarea(attrs={"rows": 2}))
    chunk_id = forms.CharField(max_length=100, label="First relevant chunk ID",
                               help_text="A query needs at least one relevant chunk.")
    grade = forms.TypedChoiceField(choices=GRADE_CHOICES[:3], coerce=int, initial=3)

    def __init__(self, *args, dataset=None, **kwargs):
        self.dataset = dataset
        super().__init__(*args, **kwargs)

    def clean_query_id(self):
        query_id = self.cleaned_data["query_id"].strip()
        if self.dataset and self.dataset.queries.filter(query_id=query_id).exists():
            raise forms.ValidationError("This dataset already has a query with this ID.")
        return query_id


class QueryEditForm(forms.Form):
    text = forms.CharField(label="Question", widget=forms.Textarea(attrs={"rows": 2}))


class GroundTruthForm(forms.Form):
    chunk_id = forms.CharField(
        max_length=100, label="Chunk ID",
        widget=forms.TextInput(attrs={"list": "chunk-options", "autocomplete": "off"}),
    )
    grade = forms.TypedChoiceField(choices=GRADE_CHOICES, coerce=int, initial=3)

    def clean_chunk_id(self):
        return self.cleaned_data["chunk_id"].strip()

    def clean_grade(self):
        grade = self.cleaned_data["grade"]
        if not 0 <= grade <= MAX_GRADE:
            raise forms.ValidationError("Grade must be between 0 and 3.")
        return grade


class EvaluationForm(forms.Form):
    dataset = forms.ModelChoiceField(queryset=EvaluationDataset.objects.all(), empty_label=None)
    name = forms.CharField(max_length=200, label="Experiment name",
                           help_text="For example \"BM25 baseline\" or \"Hybrid + reranker v2\".")
    top_k = forms.TypedChoiceField(label="Top K", coerce=int, widget=forms.RadioSelect)
    retriever_type = forms.ChoiceField(label="Retriever", choices=EvaluationRun.Retriever.choices,
                                       widget=forms.RadioSelect, initial=EvaluationRun.Retriever.BM25)

    # Built-in retrievers
    k1 = forms.FloatField(label="BM25 k1", initial=1.5, min_value=0, max_value=5, required=False,
                          help_text="Term-frequency saturation.")
    b = forms.FloatField(label="BM25 b", initial=0.75, min_value=0, max_value=1, required=False,
                         help_text="Document-length normalisation.")
    rrf_k = forms.IntegerField(label="RRF k", initial=60, min_value=1, max_value=1000, required=False,
                               help_text="Fusion constant: higher values flatten rank differences.")
    stemming = forms.BooleanField(label="Light stemming (plurals, -ing, -ed)", required=False)

    # HTTP retriever
    http_url = forms.CharField(label="Endpoint URL", required=False, max_length=500,
                               widget=forms.URLInput(attrs={"placeholder": "http://127.0.0.1:9000/retrieve"}))
    http_timeout = forms.FloatField(label="Timeout (seconds)", required=False, min_value=1, max_value=300)
    http_authorization = forms.CharField(
        label="Authorization header", required=False, max_length=2000,
        widget=forms.PasswordInput(attrs={"autocomplete": "off", "placeholder": "Bearer ..."}),
        help_text="Optional. Used for this run only, never stored.",
    )

    # Uploaded results
    results_file = forms.FileField(
        label="Retriever results file", required=False,
        widget=forms.ClearableFileInput(attrs={"accept": ".json,.csv"}),
        help_text="JSON {\"Q1\": [\"C104\", \"C101\", ...]} or CSV query_id, chunk_id, rank, score.",
    )

    notes = forms.CharField(required=False, widget=forms.Textarea(attrs={"rows": 2}))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        choices = settings.RETRIEVAL_EVAL["TOP_K_CHOICES"]
        self.fields["top_k"].choices = [(k, str(k)) for k in choices]
        self.fields["top_k"].initial = choices[0]
        self.fields["http_timeout"].initial = settings.RETRIEVAL_EVAL["HTTP_TIMEOUT"]

    def clean(self):
        cleaned = super().clean()
        kind = cleaned.get("retriever_type")
        dataset = cleaned.get("dataset")
        config: dict = {}
        if kind in ("bm25", "tfidf", "fusion"):
            if dataset and not dataset.chunks.exists():
                self.add_error("retriever_type", "This dataset has no chunk corpus, so the built-in retrievers "
                                                 "cannot run. Use the HTTP or uploaded-results option.")
            config["stemming"] = bool(cleaned.get("stemming"))
            if kind in ("bm25", "fusion"):
                config["k1"] = cleaned.get("k1") if cleaned.get("k1") is not None else 1.5
                config["b"] = cleaned.get("b") if cleaned.get("b") is not None else 0.75
            if kind == "fusion":
                config["rrf_k"] = cleaned.get("rrf_k") or 60
        elif kind == "http":
            try:
                config["url"] = validate_http_url(cleaned.get("http_url", "").strip())
            except RetrieverError as exc:
                self.add_error("http_url", str(exc))
            config["timeout"] = cleaned.get("http_timeout") or settings.RETRIEVAL_EVAL["HTTP_TIMEOUT"]
        elif kind == "precomputed":
            upload = cleaned.get("results_file")
            if not upload:
                self.add_error("results_file", "Upload the ranked results your retriever produced.")
            else:
                try:
                    config["results"] = parse_results_file(upload.name, _read_upload(upload))
                    config["source_filename"] = upload.name
                except (ResultsFormatError, forms.ValidationError) as exc:
                    self.add_error("results_file", exc.messages[0] if hasattr(exc, "messages") else str(exc))
            if dataset and config.get("results"):
                known = set(dataset.queries.values_list("query_id", flat=True))
                if not known & set(config["results"]):
                    self.add_error("results_file", "None of the query IDs in the file belong to this dataset.")
        cleaned["retriever_config"] = config
        cleaned["secrets"] = {"authorization": cleaned.get("http_authorization", "").strip()} if kind == "http" else {}
        return cleaned


class ComparisonForm(forms.Form):
    baseline = forms.ModelChoiceField(queryset=EvaluationRun.objects.none(), label="Baseline run")
    candidate = forms.ModelChoiceField(queryset=EvaluationRun.objects.none(), label="Candidate run")
    tolerance = forms.FloatField(min_value=0, max_value=1, required=False, label="Regression tolerance",
                                 help_text="A drop in an average larger than this counts as a regression.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        runs = EvaluationRun.objects.filter(status=EvaluationRun.Status.COMPLETED).select_related("dataset")
        for field in ("baseline", "candidate"):
            self.fields[field].queryset = runs
            self.fields[field].label_from_instance = lambda run: (
                f"#{run.pk} {run.name} - {run.dataset} - K={run.top_k}"
            )
        self.fields["tolerance"].initial = settings.RETRIEVAL_EVAL["REGRESSION_TOLERANCE"]
