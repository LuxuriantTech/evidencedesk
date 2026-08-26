from typing import Any, ClassVar

from arq.connections import RedisSettings
from evidencedesk_api.config import get_settings
from evidencedesk_api.db import build_engine, build_session_factory
from evidencedesk_api.provider_registry import build_provider_bundle
from evidencedesk_api.storage import LocalDocumentStorage

from evidencedesk_worker.jobs import WorkerContext, process_document


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    engine = build_engine(settings)
    providers = build_provider_bundle(
        settings.answer_mode,
        model_path=settings.embedding_model_path,
        manifest_path=settings.embedding_manifest_path,
    )
    ctx["worker"] = WorkerContext(
        settings=settings,
        engine=engine,
        session_factory=build_session_factory(engine),
        storage=LocalDocumentStorage(settings.storage_root),
        embeddings=providers.embedding,
    )


async def shutdown(ctx: dict[str, Any]) -> None:
    worker: WorkerContext = ctx["worker"]
    await worker.engine.dispose()


class WorkerSettings:
    functions: ClassVar[list[object]] = [process_document]
    on_startup = startup
    on_shutdown = shutdown
    max_tries = 3
    keep_result = 3600
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
