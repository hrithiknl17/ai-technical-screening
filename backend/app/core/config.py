"""Application configuration.

All runtime configuration is supplied through environment variables (or a
`.env` file) so the same image can run in any environment without code edits.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- app ---------------------------------------------------------------
    app_name: str = "AI Interviewer"
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:3001,http://127.0.0.1:3001"
    )

    # --- storage -----------------------------------------------------------
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'app.db').as_posix()}"
    knowledge_base_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"
    #: Built frontend (`next build` with STATIC_EXPORT=1). When present the API
    #: serves the UI from its own origin, which removes CORS from deployment.
    static_dir: Path = PROJECT_ROOT / "frontend" / "out"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"

    # --- llm ---------------------------------------------------------------
    gemini_api_key: str = ""
    #: Free-tier daily caps differ sharply per model (gemini-2.5-flash allows only
    #: 20 generate-content requests per day); flash-lite is the practical default.
    llm_model: str = "gemini-3.1-flash-lite"
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 5
    #: Client-side spacing so a burst of calls does not blow the free-tier RPM.
    llm_requests_per_minute: int = 14

    # --- embeddings --------------------------------------------------------
    #: auto | local | gemini | hashing  (see app.rag.embeddings)
    embedding_provider: str = "auto"
    local_embedding_model: str = "minishlab/potion-base-8M"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    embedding_batch_size: int = 16
    #: Starting pause between embedding batches. The client then adapts it up or
    #: down (AIMD) to sit just under whatever rate limit the key actually has.
    embedding_pause_seconds: float = 4.0

    # --- rag ---------------------------------------------------------------
    chunk_size_chars: int = 1400
    chunk_overlap_chars: int = 220
    retrieval_top_k: int = 6
    retrieval_candidate_k: int = 30
    retrieval_mmr_lambda: float = 0.65
    dense_weight: float = 0.7  # remainder goes to the lexical (BM25) score

    # --- interview ---------------------------------------------------------
    default_question_count: int = 6
    max_question_count: int = 12
    max_resume_bytes: int = 5 * 1024 * 1024

    offline_mode: bool = Field(
        default=False,
        description="Skip all network calls; use deterministic local stubs. Useful for tests.",
    )

    @field_validator("knowledge_base_dir", "vector_store_dir", "static_dir", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_enabled(self) -> bool:
        return bool(self.gemini_api_key) and not self.offline_mode


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    settings.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    if settings.database_url.startswith("sqlite:///"):
        Path(settings.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return settings
