"""Settings, read from environment variables or a `.env` file in the working directory."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Engine = Literal["ragas", "deepeval", "custom"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- LLM judge: RAGAS, DeepEval and the custom judge all use this local Ollama model ----
    ollama_base_url: str = "http://127.0.0.1:11434"
    judge_model: str = "qwen2.5:1.5b"
    judge_temperature: float = 0.0
    judge_timeout_seconds: float = 300
    judge_max_tokens: int = 2048
    # Embeddings that RAGAS needs for Answer Relevancy and Answer Correctness (served by Ollama).
    ragas_embedding_model: str = "all-minilm"
    # RAGAS Answer Relevancy asks the judge for this many reverse questions (more = steadier, slower).
    ragas_relevancy_strictness: int = Field(default=1, ge=1, le=5)
    # DeepEval: also ask the judge why it gave the score (one extra LLM call per metric).
    include_reasons: bool = False

    # ---- Semantic similarity (Sentence Transformers) ----
    sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    torch_threads: int = Field(default=2, ge=1)

    # ---- Which framework scores each LLM-judged metric by default ----
    faithfulness_engine: Engine = "ragas"
    answer_relevancy_engine: Engine = "ragas"
    answer_correctness_engine: Engine = "ragas"
    answer_completeness_engine: Engine = "deepeval"

    # ---- Pass thresholds, 0 to 1 ----
    threshold_faithfulness: float = Field(default=0.8, ge=0, le=1)
    threshold_answer_relevancy: float = Field(default=0.7, ge=0, le=1)
    threshold_answer_correctness: float = Field(default=0.7, ge=0, le=1)
    threshold_answer_completeness: float = Field(default=0.8, ge=0, le=1)
    threshold_semantic_similarity: float = Field(default=0.8, ge=0, le=1)
    threshold_exact_match: float = Field(default=1.0, ge=0, le=1)
    threshold_bleu: float = Field(default=0.3, ge=0, le=1)
    threshold_rouge: float = Field(default=0.5, ge=0, le=1)

    # ---- Evaluation tracking (Langfuse) ----
    tracking_enabled: bool = True
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"

    # ---- Service ----
    # Link to the Retrieval Evaluation app, shown in the UI's top bar.
    retrieval_app_url: str = "http://127.0.0.1:8011/"
    # LLM judging on a CPU is memory-hungry: evaluate one sample at a time by default.
    max_concurrent_evaluations: int = Field(default=1, ge=1)

    def threshold_for(self, metric: str) -> float:
        return getattr(self, f"threshold_{metric}")

    def default_engine(self, metric: str) -> Engine | None:
        return getattr(self, f"{metric}_engine", None)

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
