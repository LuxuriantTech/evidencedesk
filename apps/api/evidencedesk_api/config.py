from functools import lru_cache
from pathlib import Path

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
    answer_mode: str = "extractive-local"
    evaluation_manifest: Path = Path("datasets/evaluation_cases.json")
    corpus_manifest: Path = Path("datasets/corpus_manifest.json")
    demo_admin_password: str
    demo_analyst_password: str
    demo_reader_password: str


@lru_cache
def get_settings() -> Settings:
    return Settings()
