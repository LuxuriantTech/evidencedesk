from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"
    database_url: str = (
        "postgresql+asyncpg://evidencedesk:evidencedesk-local-only@127.0.0.1:55432/evidencedesk"
    )
    redis_url: str = "redis://127.0.0.1:56379/0"
    storage_root: Path = Path("storage")
    jwt_secret: str = Field(min_length=32)
    access_token_minutes: int = Field(default=30, ge=1, le=30)
    public_demo_mode: bool = True
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_request_bytes: int = Field(default=11 * 1024 * 1024, ge=1)
    max_document_pages: int = Field(default=200, ge=1, le=10_000)
    max_extracted_chars: int = Field(default=1_000_000, ge=1, le=50_000_000)
    max_document_chunks: int = Field(default=5_000, ge=1, le=100_000)
    pdf_parse_timeout_seconds: float = Field(default=10.0, ge=0.1, le=120.0)
    pdf_parse_memory_bytes: int = Field(
        default=512 * 1024 * 1024,
        ge=64 * 1024 * 1024,
        le=4 * 1024 * 1024 * 1024,
    )
    answer_mode: Literal[
        "extractive-local-hash", "extractive-local-onnx", "grounded-local-v3"
    ] = "grounded-local-v3"
    embedding_model_path: Path = Path("models/paraphrase-multilingual-minilm-l12-v2")
    embedding_manifest_path: Path = Path(
        "infra/models/paraphrase-multilingual-minilm-l12-v2.json"
    )
    evaluation_manifest: Path = Path("datasets/development_v2/evaluation_cases.json")
    evaluation_corpus_manifest: Path = Path("datasets/development_v2/corpus_manifest.json")
    corpus_manifest: Path = Path("datasets/corpus_manifest.json")
    public_demo_allowlist: Path = Path("datasets/public_demo_uploads.json")
    demo_admin_password: str
    demo_analyst_password: str
    demo_reader_password: str


@lru_cache
def get_settings() -> Settings:
    return Settings()
